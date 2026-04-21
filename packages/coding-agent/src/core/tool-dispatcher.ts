import { randomUUID } from "node:crypto";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StreamFunction,
	type Tool,
	type Usage,
} from "@mariozechner/pi-ai";

export interface ToolCallEnvelope {
	tool: string;
	args: Record<string, unknown>;
}

const EMPTY_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		total: 0,
	},
};

interface StreamAttemptResult {
	message: AssistantMessage;
	terminalType: "done" | "error";
	rawToolArgsByIndex: Map<number, string>;
}

interface NormalizeToolMessageResult {
	ok: true;
	message: AssistantMessage;
	previousOutput: string;
}

interface NormalizeToolMessageFailure {
	ok: false;
	error: Error;
	previousOutput: string;
}

type NormalizeToolMessageOutcome = NormalizeToolMessageResult | NormalizeToolMessageFailure;

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stripCodeFences(text: string): string {
	const trimmed = text.trim();
	const match = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
	return match ? match[1].trim() : trimmed;
}

function extractFirstJsonObjectCandidate(text: string): string | undefined {
	let start = -1;
	let braceDepth = 0;
	let bracketDepth = 0;
	let inString = false;
	let escaped = false;

	for (let index = 0; index < text.length; index++) {
		const char = text[index];

		if (start === -1) {
			if (char !== "{") {
				continue;
			}
			start = index;
			braceDepth = 1;
			continue;
		}

		if (escaped) {
			escaped = false;
			continue;
		}

		if (inString) {
			if (char === "\\") {
				escaped = true;
			} else if (char === '"') {
				inString = false;
			}
			continue;
		}

		if (char === '"') {
			inString = true;
			continue;
		}

		if (char === "{") {
			braceDepth++;
		} else if (char === "}") {
			braceDepth--;
		} else if (char === "[") {
			bracketDepth++;
		} else if (char === "]") {
			bracketDepth--;
		}

		if (braceDepth < 0 || bracketDepth < 0) {
			throw new Error("Malformed JSON candidate");
		}

		if (braceDepth === 0 && bracketDepth === 0) {
			return text.slice(start, index + 1).trim();
		}
	}

	if (start !== -1) {
		return text.slice(start).trim();
	}

	return undefined;
}

export function extractJSON(text: string): string {
	const sources = [
		...Array.from(text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi), (match) => match[1].trim()),
		text.trim(),
	];

	for (const source of sources) {
		const candidate = extractFirstJsonObjectCandidate(source);
		if (candidate) {
			return candidate;
		}
	}

	throw new Error("No JSON object found in tool output");
}

function unwrapStringifiedJsonArtifacts(raw: string): string {
	let current = stripCodeFences(raw);

	for (let index = 0; index < 3; index++) {
		const trimmed = current.trim();
		if (!(trimmed.startsWith('"') && trimmed.endsWith('"'))) {
			break;
		}

		try {
			const parsed = JSON.parse(trimmed);
			if (typeof parsed !== "string") {
				break;
			}
			current = parsed;
		} catch {
			break;
		}
	}

	return current.trim();
}

function removeTrailingCommas(input: string): string {
	let output = "";
	let inString = false;
	let escaped = false;

	for (let index = 0; index < input.length; index++) {
		const char = input[index];

		if (escaped) {
			output += char;
			escaped = false;
			continue;
		}

		if (inString) {
			output += char;
			if (char === "\\") {
				escaped = true;
			} else if (char === '"') {
				inString = false;
			}
			continue;
		}

		if (char === '"') {
			output += char;
			inString = true;
			continue;
		}

		if (char === ",") {
			let lookahead = index + 1;
			while (lookahead < input.length && /\s/.test(input[lookahead])) {
				lookahead++;
			}
			if (input[lookahead] === "}" || input[lookahead] === "]") {
				continue;
			}
		}

		output += char;
	}

	return output;
}

function closeJsonDelimiters(input: string): string {
	const stack: string[] = [];
	let inString = false;
	let escaped = false;

	for (const char of input) {
		if (escaped) {
			escaped = false;
			continue;
		}

		if (inString) {
			if (char === "\\") {
				escaped = true;
			} else if (char === '"') {
				inString = false;
			}
			continue;
		}

		if (char === '"') {
			inString = true;
			continue;
		}

		if (char === "{" || char === "[") {
			stack.push(char);
			continue;
		}

		if (char === "}" || char === "]") {
			const expected = char === "}" ? "{" : "[";
			const actual = stack.pop();
			if (actual !== expected) {
				throw new Error("Uncertain repair: mismatched JSON delimiters");
			}
		}
	}

	if (inString) {
		throw new Error("Uncertain repair: unterminated string");
	}

	return (
		input +
		stack
			.reverse()
			.map((char) => (char === "{" ? "}" : "]"))
			.join("")
	);
}

function normalizeStringifiedJson(value: unknown): unknown {
	if (typeof value === "string") {
		const trimmed = value.trim();
		if (trimmed.length === 0) {
			return value;
		}

		const maybeJson =
			(trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"));
		if (!maybeJson) {
			return value;
		}

		try {
			return normalizeStringifiedJson(JSON.parse(trimmed));
		} catch {
			return value;
		}
	}

	if (Array.isArray(value)) {
		return value.map((item) => normalizeStringifiedJson(item));
	}

	if (!isObject(value)) {
		return value;
	}

	const normalized: Record<string, unknown> = {};
	for (const [key, entryValue] of Object.entries(value)) {
		normalized[key] = normalizeStringifiedJson(entryValue);
	}
	return normalized;
}

function renameCommonKeys(value: unknown): unknown {
	if (Array.isArray(value)) {
		return value.map((item) => renameCommonKeys(item));
	}

	if (!isObject(value)) {
		return value;
	}

	const renamed: Record<string, unknown> = {};
	for (const [key, entryValue] of Object.entries(value)) {
		const normalizedKey = key === "filePath" ? "path" : key;
		const normalizedValue = renameCommonKeys(entryValue);

		if (normalizedKey in renamed && JSON.stringify(renamed[normalizedKey]) !== JSON.stringify(normalizedValue)) {
			throw new Error(`Uncertain repair: conflicting keys for "${normalizedKey}"`);
		}

		renamed[normalizedKey] = normalizedValue;
	}

	return renamed;
}

export function repairToolCall(raw: string | unknown): unknown {
	if (isObject(raw)) {
		return renameCommonKeys(normalizeStringifiedJson(raw));
	}

	if (typeof raw !== "string") {
		throw new Error("Tool output must be JSON text or an object");
	}

	let candidate = unwrapStringifiedJsonArtifacts(raw);
	if (!(candidate.startsWith("{") && candidate.endsWith("}"))) {
		candidate = extractJSON(candidate);
	}

	candidate = removeTrailingCommas(candidate);
	candidate = closeJsonDelimiters(candidate);

	let parsed: unknown;
	try {
		parsed = JSON.parse(candidate);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`Unable to repair tool JSON: ${message}`);
	}

	const normalized = renameCommonKeys(normalizeStringifiedJson(parsed));
	if (!isObject(normalized)) {
		throw new Error("Repaired tool JSON must decode to an object");
	}

	return normalized;
}

export function validateToolSchema(obj: unknown): ToolCallEnvelope {
	if (!isObject(obj)) {
		throw new Error("Tool call must be a JSON object");
	}

	const keys = Object.keys(obj);
	if (keys.length !== 2 || !keys.includes("tool") || !keys.includes("args")) {
		throw new Error('Tool call must match exactly: { "tool": string, "args": object }');
	}

	if (typeof obj.tool !== "string" || obj.tool.trim().length === 0) {
		throw new Error('Tool call field "tool" must be a non-empty string');
	}

	if (!isObject(obj.args)) {
		throw new Error('Tool call field "args" must be an object');
	}

	return {
		tool: obj.tool.trim(),
		args: obj.args,
	};
}

function looksLikeToolEnvelope(text: string): boolean {
	const trimmed = text.trim();
	if (!trimmed) {
		return false;
	}

	return (
		trimmed.startsWith("{") ||
		trimmed.startsWith("```") ||
		/"tool"\s*:/.test(trimmed) ||
		/"args"\s*:/.test(trimmed) ||
		/\bfilePath\b/.test(trimmed)
	);
}

function mergeUsage(left: Usage | undefined, right: Usage | undefined): Usage {
	const a = left ?? EMPTY_USAGE;
	const b = right ?? EMPTY_USAGE;
	return {
		input: a.input + b.input,
		output: a.output + b.output,
		cacheRead: a.cacheRead + b.cacheRead,
		cacheWrite: a.cacheWrite + b.cacheWrite,
		totalTokens: a.totalTokens + b.totalTokens,
		cost: {
			input: a.cost.input + b.cost.input,
			output: a.cost.output + b.cost.output,
			cacheRead: a.cost.cacheRead + b.cost.cacheRead,
			cacheWrite: a.cost.cacheWrite + b.cost.cacheWrite,
			total: a.cost.total + b.cost.total,
		},
	};
}

function createRetryAssistantMessage(model: Model<Api>, previousOutput: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: previousOutput }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: EMPTY_USAGE,
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function createRetryUserMessage(error: Error, previousOutput: string, tools?: Tool[]): Message {
	const allowedTools = tools?.length ? `\nTool must be one of: ${tools.map((tool) => tool.name).join(", ")}` : "";
	return {
		role: "user",
		content: [
			{
				type: "text",
				text:
					"You returned invalid tool JSON. Fix it and return ONLY valid JSON matching:\n" +
					'{ "tool": string, "args": object }\n\n' +
					"Rules:\n" +
					"- no explanation\n" +
					"- no markdown\n" +
					"- no thinking text\n" +
					"- only JSON" +
					allowedTools +
					`\n\nValidation error:\n${error.message}\n\nPrevious output:\n${previousOutput}`,
			},
		],
		timestamp: Date.now(),
	};
}

function assistantMessageToRetryText(message: AssistantMessage, rawToolArgsByIndex: Map<number, string>): string {
	const chunks: string[] = [];

	message.content.forEach((block, contentIndex) => {
		if (block.type === "text") {
			chunks.push(block.text);
			return;
		}

		if (block.type !== "toolCall") {
			return;
		}

		const rawArgs = rawToolArgsByIndex.get(contentIndex)?.trim();
		const serializedArgs = rawArgs && rawArgs.length > 0 ? rawArgs : JSON.stringify(block.arguments ?? {});
		chunks.push(`{"tool":${JSON.stringify(block.name)},"args":${serializedArgs}}`);
	});

	return chunks.join("\n").trim();
}

function createRecoveredToolCallId(): string {
	return `tool_recovered_${randomUUID()}`;
}

function createFailedDispatchMessage(model: Model<Api>, usage: Usage, error: Error): AssistantMessage {
	const errorMessage = `Invalid tool JSON after 3 retries: ${error.message}`;
	return {
		role: "assistant",
		content: [{ type: "text", text: errorMessage }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage,
		stopReason: "error",
		errorMessage,
		timestamp: Date.now(),
	};
}

function normalizeToolMessage(
	message: AssistantMessage,
	context: Context,
	rawToolArgsByIndex: Map<number, string>,
): NormalizeToolMessageOutcome {
	const previousOutput = assistantMessageToRetryText(message, rawToolArgsByIndex);
	if (message.stopReason === "error" || message.stopReason === "aborted") {
		return { ok: true, message, previousOutput };
	}

	const nativeToolCalls = message.content.filter((block) => block.type === "toolCall");
	if (nativeToolCalls.length > 0) {
		try {
			const normalizedContent = message.content.map((block, contentIndex) => {
				if (block.type !== "toolCall") {
					return block;
				}

				const rawArgs = rawToolArgsByIndex.get(contentIndex)?.trim();
				const rawEnvelope =
					rawArgs && rawArgs.length > 0
						? `{"tool":${JSON.stringify(block.name)},"args":${rawArgs}}`
						: JSON.stringify({ tool: block.name, args: block.arguments ?? {} });
				const repaired = repairToolCall(rawEnvelope);
				const validated = validateToolSchema(repaired);
				return {
					...block,
					name: validated.tool,
					arguments: validated.args,
				};
			});

			return {
				ok: true,
				message: {
					...message,
					content: normalizedContent,
					stopReason: "toolUse",
				},
				previousOutput,
			};
		} catch (error) {
			return {
				ok: false,
				error: error instanceof Error ? error : new Error(String(error)),
				previousOutput,
			};
		}
	}

	if (!context.tools?.length) {
		return { ok: true, message, previousOutput };
	}

	const textOutput = message.content
		.filter((block): block is Extract<AssistantMessage["content"][number], { type: "text" }> => block.type === "text")
		.map((block) => block.text)
		.join("\n")
		.trim();

	if (!looksLikeToolEnvelope(textOutput)) {
		return { ok: true, message, previousOutput: textOutput || previousOutput };
	}

	try {
		const extracted = extractJSON(textOutput);
		const repaired = repairToolCall(extracted);
		const validated = validateToolSchema(repaired);
		const thinkingBlocks = message.content.filter((block) => block.type === "thinking");
		return {
			ok: true,
			message: {
				...message,
				content: [
					...thinkingBlocks,
					{
						type: "toolCall",
						id: createRecoveredToolCallId(),
						name: validated.tool,
						arguments: validated.args,
					},
				],
				stopReason: "toolUse",
			},
			previousOutput: textOutput,
		};
	} catch (error) {
		return {
			ok: false,
			error: error instanceof Error ? error : new Error(String(error)),
			previousOutput: textOutput || previousOutput,
		};
	}
}

async function runStreamAttempt(
	model: Model<Api>,
	context: Context,
	options: SimpleStreamOptions | undefined,
	streamFn: StreamFunction<Api, SimpleStreamOptions>,
	out: AssistantMessageEventStream,
	forwardEvents: boolean,
): Promise<StreamAttemptResult> {
	const stream = await Promise.resolve(streamFn(model, context, options));
	const rawToolArgsByIndex = new Map<number, string>();
	let terminalType: "done" | "error" = "done";

	for await (const event of stream) {
		if (event.type === "toolcall_delta") {
			const current = rawToolArgsByIndex.get(event.contentIndex) ?? "";
			rawToolArgsByIndex.set(event.contentIndex, current + event.delta);
		}

		if (event.type === "done" || event.type === "error") {
			terminalType = event.type;
			continue;
		}

		if (forwardEvents) {
			out.push(event);
		}
	}

	const message = await stream.result();
	return { message, terminalType, rawToolArgsByIndex };
}

export async function retryWithLLM(
	model: Model<Api>,
	context: Context,
	options: SimpleStreamOptions | undefined,
	error: Error,
	previousOutput: string,
	streamFn: StreamFunction<Api, SimpleStreamOptions>,
): Promise<StreamAttemptResult> {
	const retryContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: [
			...context.messages,
			createRetryAssistantMessage(model, previousOutput),
			createRetryUserMessage(error, previousOutput, context.tools),
		],
	};

	return runStreamAttempt(model, retryContext, options, streamFn, createAssistantMessageEventStream(), false);
}

export function createRobustToolDispatchStream(
	model: Model<Api>,
	context: Context,
	options: SimpleStreamOptions | undefined,
	streamFn: StreamFunction<Api, SimpleStreamOptions>,
): AssistantMessageEventStream {
	const out = createAssistantMessageEventStream();

	void (async () => {
		let totalUsage = EMPTY_USAGE;
		let attempt = await runStreamAttempt(model, context, options, streamFn, out, true);
		totalUsage = mergeUsage(totalUsage, attempt.message.usage);

		let normalized = normalizeToolMessage(attempt.message, context, attempt.rawToolArgsByIndex);
		if (normalized.ok) {
			const message = { ...normalized.message, usage: totalUsage };
			out.push(
				message.stopReason === "error" || message.stopReason === "aborted"
					? { type: "error", reason: message.stopReason, error: message }
					: { type: "done", reason: message.stopReason, message },
			);
			return;
		}

		let lastError = normalized.error;
		let previousOutput = normalized.previousOutput;

		for (let retryIndex = 0; retryIndex < 3; retryIndex++) {
			attempt = await retryWithLLM(model, context, options, lastError, previousOutput, streamFn);
			totalUsage = mergeUsage(totalUsage, attempt.message.usage);
			normalized = normalizeToolMessage(attempt.message, context, attempt.rawToolArgsByIndex);
			if (normalized.ok) {
				const message = { ...normalized.message, usage: totalUsage };
				out.push(
					message.stopReason === "error" || message.stopReason === "aborted"
						? { type: "error", reason: message.stopReason, error: message }
						: { type: "done", reason: message.stopReason, message },
				);
				return;
			}

			lastError = normalized.error;
			previousOutput = normalized.previousOutput;
		}

		out.push({ type: "error", reason: "error", error: createFailedDispatchMessage(model, totalUsage, lastError) });
	})().catch((error) => {
		const failure = createFailedDispatchMessage(
			model,
			EMPTY_USAGE,
			error instanceof Error ? error : new Error(String(error)),
		);
		out.push({ type: "error", reason: "error", error: failure });
	});

	return out;
}
