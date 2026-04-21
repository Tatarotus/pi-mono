import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { extractJSON, repairToolCall, validateToolSchema } from "../src/core/tool-dispatcher.js";

function createModel(provider: string): Model<Api> {
	return {
		id: `${provider}-model`,
		name: `${provider} Model`,
		api: "openai-completions",
		provider,
		baseUrl: `https://${provider}.example.com/v1`,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};
}

function createAssistantMessage(
	model: Model<Api>,
	text: string,
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	};
}

function createDoneStream(message: AssistantMessage) {
	const stream = createAssistantMessageEventStream();
	stream.end(message);
	return stream;
}

describe("tool dispatcher helpers", () => {
	it("extracts the first JSON object from mixed text", () => {
		expect(extractJSON('Before\n```json\n{"tool":"read","args":{"path":"/tmp/demo"}}\n```\nAfter')).toBe(
			'{"tool":"read","args":{"path":"/tmp/demo"}}',
		);
	});

	it("returns a partial JSON candidate for truncated output", () => {
		expect(extractJSON('Noise {"tool":"read","args":{"path":"/tmp/demo"')).toBe(
			'{"tool":"read","args":{"path":"/tmp/demo"',
		);
	});

	it("repairs trailing commas, stringified JSON, and filePath aliases", () => {
		const repaired = validateToolSchema(
			repairToolCall('"{\\"tool\\":\\"read\\",\\"args\\":{\\"filePath\\":\\"/tmp/demo\\",}}"'),
		);

		expect(repaired).toEqual({
			tool: "read",
			args: { path: "/tmp/demo" },
		});
	});

	it("rejects objects that do not match the exact tool schema", () => {
		expect(() => validateToolSchema({ tool: "read", args: {}, extra: true })).toThrow(
			'exactly: { "tool": string, "args": object }',
		);
	});
});

describe("tool dispatcher integration", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-tool-dispatcher-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	it("repairs text-wrapped tool JSON and dispatches the tool", async () => {
		const provider = "dispatch-repair";
		const model = createModel(provider);
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey(provider, "test-key");
		const modelRegistry = ModelRegistry.create(authStorage, join(agentDir, "models.json"));
		const settingsManager = SettingsManager.create(cwd, agentDir);
		let callCount = 0;
		let executedPath: string | undefined;

		modelRegistry.registerProvider(provider, {
			api: model.api,
			streamSimple: (_model, _context, _options?: SimpleStreamOptions) => {
				callCount++;
				if (callCount === 1) {
					return createDoneStream(
						createAssistantMessage(
							model,
							'Let me call the tool.\n```json\n{"tool":"capture","args":{"filePath":"/tmp/repaired",},}\n```',
						),
					);
				}

				return createDoneStream(createAssistantMessage(model, "done"));
			},
		});

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model,
			authStorage,
			modelRegistry,
			settingsManager,
			sessionManager: SessionManager.inMemory(cwd),
			customTools: [
				{
					name: "capture",
					label: "Capture",
					description: "Captures a path",
					parameters: Type.Object({ path: Type.String() }),
					execute: async (_toolCallId, params: { path: string }) => {
						executedPath = params.path;
						return {
							content: [{ type: "text", text: params.path }],
							details: {},
						};
					},
				},
			],
		});

		try {
			await session.prompt("Use the capture tool.");

			expect(callCount).toBe(2);
			expect(executedPath).toBe("/tmp/repaired");

			const assistantToolCall = session.messages.find(
				(message) =>
					message.role === "assistant" &&
					message.content.some((block) => block.type === "toolCall" && block.name === "capture"),
			) as AssistantMessage | undefined;
			const toolCall = assistantToolCall?.content.find(
				(block): block is Extract<AssistantMessage["content"][number], { type: "toolCall" }> =>
					block.type === "toolCall",
			);

			expect(toolCall?.arguments).toEqual({ path: "/tmp/repaired" });
		} finally {
			session.dispose();
			modelRegistry.unregisterProvider(provider);
		}
	});

	it("retries with the model when repair is uncertain", async () => {
		const provider = "dispatch-retry";
		const model = createModel(provider);
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey(provider, "test-key");
		const modelRegistry = ModelRegistry.create(authStorage, join(agentDir, "models.json"));
		const settingsManager = SettingsManager.create(cwd, agentDir);
		let callCount = 0;
		let executedPath: string | undefined;
		const retryPromptChecks: string[] = [];
		const retryToolsStates: Array<unknown> = [];

		modelRegistry.registerProvider(provider, {
			api: model.api,
			streamSimple: (_model, context, _options?: SimpleStreamOptions) => {
				callCount++;
				if (callCount === 1) {
					return createDoneStream(
						createAssistantMessage(model, '{"tool":"capture","args":{"path":"unterminated}'),
					);
				}

				if (callCount === 2) {
					const lastMessage = context.messages[context.messages.length - 1];
					if (lastMessage?.role === "user" && Array.isArray(lastMessage.content)) {
						retryPromptChecks.push(
							lastMessage.content.map((part) => (part.type === "text" ? part.text : "")).join("\n"),
						);
					}
					retryToolsStates.push(context.tools);
					return createDoneStream(
						createAssistantMessage(model, '{"tool":"capture","args":{"path":"/tmp/retried"}}'),
					);
				}

				return createDoneStream(createAssistantMessage(model, "done"));
			},
		});

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model,
			authStorage,
			modelRegistry,
			settingsManager,
			sessionManager: SessionManager.inMemory(cwd),
			customTools: [
				{
					name: "capture",
					label: "Capture",
					description: "Captures a path",
					parameters: Type.Object({ path: Type.String() }),
					execute: async (_toolCallId, params: { path: string }) => {
						executedPath = params.path;
						return {
							content: [{ type: "text", text: params.path }],
							details: {},
						};
					},
				},
			],
		});

		try {
			await session.prompt("Use the capture tool.");

			expect(callCount).toBe(3);
			expect(executedPath).toBe("/tmp/retried");
			expect(retryPromptChecks[0]).toContain("You returned invalid tool JSON");
			expect(retryPromptChecks[0]).toContain('{ "tool": string, "args": object }');
			expect(retryToolsStates[0]).toBeUndefined();
		} finally {
			session.dispose();
			modelRegistry.unregisterProvider(provider);
		}
	});
});
