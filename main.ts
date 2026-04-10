/**
 * Qwen API to OpenAI Standard - Single File Deno Deploy/Playground Script
 *
 * @version 5.0.8
 * @description 完全按照官方 payload + CORS 支持 + 搜索模式修复 + 工具调用支持
 */

import {
	Application,
	Router,
	Context,
	Middleware,
} from "https://deno.land/x/oak@v12.6.1/mod.ts";

class Logger {
	private formatTimestamp(): string { return new Date().toISOString(); }

	info(message: string, data?: any) {
		console.log(`[${this.formatTimestamp()}] INFO: ${message}`, data ? JSON.stringify(data, null, 2) : "");
	}

	error(message: string, error?: any, data?: any) {
		console.error(`[${this.formatTimestamp()}] ERROR: ${message}`, { error: error?.message || error, ...data });
	}

	debug(message: string, data?: any) {
		if ((Deno.env.get("DEBUG") || "").toLowerCase() === "true") {
			console.log(`[${this.formatTimestamp()}] DEBUG: ${message}`, data ? JSON.stringify(data, null, 2) : "");
		}
	}

	request(ctx: Context, startTime: number) {
		const duration = `${Date.now() - startTime}ms`;
		const level = (ctx.response.status || 0) >= 400 ? "ERROR" : "INFO";
		console.log(`[${this.formatTimestamp()}] ${level}: ${ctx.request.method} ${ctx.request.url.pathname} - ${ctx.response.status} (${duration})`);
	}
}

const logger = new Logger();

const config = {
	salt: Deno.env.get("SALT") || "",
	useDenoEnv: (Deno.env.get("USE_DENO_ENV") || "").toLowerCase() === "true",
	qwenTokenEnv: Deno.env.get("QWEN_TOKEN") || "",
	ssxmodItnaEnv: Deno.env.get("SSXMOD_ITNA_VALUE") || "",
	debug: (Deno.env.get("DEBUG") || "").toLowerCase() === "true",
	sessionTemp: (Deno.env.get("QWEN_SESSION_TEMP") || "true").toLowerCase() === "true",
};

const QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2/chat/completions";
const QWEN_CHAT_NEW_URL = "https://chat.qwen.ai/api/v2/chats/new";
const QWEN_CHAT_INFO_URL = "https://chat.qwen.ai/api/v2/chats";

type OpenAITool = {
	type?: string;
	function?: {
		name?: string;
		description?: string;
		parameters?: any;
	};
};

type ParsedToolCall = {
	id: string;
	name: string;
	input: any;
};

function normalizeOpenAITools(tools: any[]): OpenAITool[] {
	if (!Array.isArray(tools)) return [];
	return tools
		.filter((t) => t && (t.type === "function" || t.function))
		.map((t) => ({
			type: "function",
			function: {
				name: t?.function?.name || t?.name,
				description: t?.function?.description || "",
				parameters: t?.function?.parameters || { type: "object", properties: {} },
			},
		}))
		.filter((t) => !!t.function?.name);
}

function extractTextContent(content: any): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		let text = "";
		for (const item of content) {
			if (item?.type === "text") text += item?.text || "";
		}
		return text;
	}
	return "";
}

function safeJsonParse(input: string, fallback: any) {
	try {
		return JSON.parse(input);
	} catch {
		return fallback;
	}
}

function buildPromptWithTools(messages: any[], tools: OpenAITool[]): string {
	const lines: string[] = [];
	let systemPrompt = "You are a helpful assistant.";

	for (const msg of messages || []) {
		const role = msg?.role;
		if (role === "system") {
			const text = extractTextContent(msg?.content);
			if (text) systemPrompt = text;
		}
	}

	lines.push(`[System]\n${systemPrompt}\n[/System]`);

	if (tools.length > 0) {
		const toolSpec = tools.map((t) => ({
			name: t.function?.name,
			description: t.function?.description || "",
			parameters: t.function?.parameters || { type: "object", properties: {} },
		}));
		lines.push(
			"[Tools]\n" +
			JSON.stringify(toolSpec, null, 2) +
			"\n[/Tools]\n" +
			"If a tool is needed, output exactly one tool call block in this format:\n" +
			"##TOOL_CALL##\n{\"name\":\"tool_name\",\"input\":{...}}\n##END_CALL##\n" +
			"Do not add markdown fences around the JSON."
		);
	}

	for (const msg of messages || []) {
		const role = msg?.role;
		if (role === "system") continue;

		if (role === "assistant" && Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0) {
			for (const tc of msg.tool_calls) {
				const fn = tc?.function || {};
				const argsStr = typeof fn?.arguments === "string" ? fn.arguments : JSON.stringify(fn?.arguments || {});
				const argsObj = safeJsonParse(argsStr, { raw: argsStr });
				lines.push(`Assistant tool call:\n##TOOL_CALL##\n${JSON.stringify({ name: fn?.name || "", input: argsObj })}\n##END_CALL##`);
			}
			continue;
		}

		if (role === "tool") {
			const toolResult = extractTextContent(msg?.content) || (typeof msg?.content === "string" ? msg.content : JSON.stringify(msg?.content || {}));
			const toolCallId = msg?.tool_call_id ? ` id=${msg.tool_call_id}` : "";
			lines.push(`[Tool Result${toolCallId}]\n${toolResult}\n[/Tool Result]`);
			continue;
		}

		const text = extractTextContent(msg?.content);
		if (!text) continue;
		const tag = role === "assistant" ? "Assistant" : "User";
		lines.push(`[${tag}]\n${text}\n[/${tag}]`);
	}

	return lines.join("\n\n");
}

function parseToolCallsFromText(answer: string): ParsedToolCall[] {
	const blocks: ParsedToolCall[] = [];
	const add = (name: string, input: any) => {
		if (!name) return;
		blocks.push({ id: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`, name, input: input ?? {} });
	};

	const regexes = [
		/##TOOL_CALL##\s*([\s\S]*?)\s*##END_CALL##/gi,
		/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi,
	];

	for (const re of regexes) {
		let m: RegExpExecArray | null;
		while ((m = re.exec(answer)) !== null) {
			const obj = safeJsonParse((m[1] || "").trim(), null);
			if (!obj || typeof obj !== "object") continue;
			const name = obj.name || obj.function?.name || "";
			const input = obj.input ?? obj.arguments ?? obj.parameters ?? obj.args ?? {};
			add(name, input);
		}
	}

	if (blocks.length === 0) {
		const trimmed = answer.trim().replace(/^```(?:json)?\s*/i, "").replace(/```$/, "").trim();
		if (trimmed.startsWith("{") && trimmed.includes("\"name\"")) {
			const obj = safeJsonParse(trimmed, null);
			if (obj && typeof obj === "object") {
				const name = obj.name || obj.function?.name || "";
				const input = obj.input ?? obj.arguments ?? obj.parameters ?? obj.args ?? {};
				add(name, input);
			}
		}
	}

	return blocks;
}

function parseOpenAIStreamToCompletion(rawSseText: string, model: string) {
	let content = "";
	const toolCallsByIndex: Record<number, any> = {};

	for (const line of rawSseText.split("\n")) {
		const text = line.trim();
		if (!text.startsWith("data:")) continue;
		const dataStr = text.slice(5).trim();
		if (!dataStr || dataStr === "[DONE]") continue;
		const chunk = safeJsonParse(dataStr, null);
		if (!chunk || !chunk?.choices?.[0]) continue;
		const delta = chunk.choices[0].delta || {};
		if (typeof delta.content === "string") content += delta.content;
		if (Array.isArray(delta.tool_calls)) {
			for (const tc of delta.tool_calls) {
				const idx = Number(tc?.index ?? 0);
				if (!toolCallsByIndex[idx]) toolCallsByIndex[idx] = { id: "", type: "function", function: { name: "", arguments: "" } };
				if (tc?.id) toolCallsByIndex[idx].id = tc.id;
				if (tc?.type) toolCallsByIndex[idx].type = tc.type;
				if (tc?.function?.name) toolCallsByIndex[idx].function.name = tc.function.name;
				if (tc?.function?.arguments) toolCallsByIndex[idx].function.arguments += tc.function.arguments;
			}
		}
	}

	const toolCalls = Object.keys(toolCallsByIndex)
		.map((k) => toolCallsByIndex[Number(k)])
		.filter((x) => x?.function?.name);

	const message = toolCalls.length > 0
		? { role: "assistant", content: null, tool_calls: toolCalls }
		: { role: "assistant", content };

	return {
		id: `chatcmpl-${crypto.randomUUID()}`,
		object: "chat.completion",
		created: Math.floor(Date.now() / 1000),
		model,
		choices: [
			{
				index: 0,
				message,
				finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
			},
		],
	};
}

function normalizeModelAlias(model: string): string {
	const suffixMatch = model.match(/-(search|thinking|image|image_edit|video|research)$/);
	const suffix = suffixMatch?.[0] || "";
	let base = suffix ? model.slice(0, -suffix.length) : model;
	const map: Record<string, string> = {
		"qwen3.6-plus": "qwen3.6-plus",
		"qwen3-plus": "qwen3.6-plus",
		"qwen-plus-latest": "qwen3.6-plus",
		"qwen-plus": "qwen3.6-plus",
	};
	base = map[base] || base;
	return `${base}${suffix}`;
}

function resolveChatType(model: string): string {
	if (model.endsWith("-video")) return "t2v";
	if (model.endsWith("-image_edit")) return "image_edit";
	if (model.endsWith("-image")) return "t2i";
	if (model.endsWith("-search")) return "search";
	if (model.endsWith("-research")) return "deep_research";
	return "t2t";
}

async function createNewChat(token: string, model: string, chatType: string, isTemp: boolean, ssxmodItna?: string): Promise<string | null> {
	try {
		const headers: Record<string, string> = {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
			"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
			source: "web",
			Referer: isTemp ? "https://chat.qwen.ai/?temporary-chat=true" : "https://chat.qwen.ai/",
			Origin: "https://chat.qwen.ai",
		};
		if (ssxmodItna) headers["Cookie"] = `ssxmod_itna=${ssxmodItna}`;

		const isSearch = chatType === "search";
		const isResearch = chatType === "deep_research";

		const body: any = {
			title: isTemp ? "(temp)" : "Conversation",
			models: [model],
			chat_mode: isSearch ? "search" : (isResearch ? "deep_research" : "normal"),
			chat_type: chatType,
			timestamp: Date.now()
		};

		if (isSearch) {
			body.search_mode = "enable";
		}

		const res = await fetch(QWEN_CHAT_NEW_URL, { method: "POST", headers, body: JSON.stringify(body) });
		if (!res.ok) return null;
		const data = await res.json();
		return data?.data?.id || null;
	} catch (e) { logger.error("createNewChat error", e); return null; }
}

async function deleteChat(chatId: string, token: string, ssxmodItna?: string): Promise<boolean> {
	try {
		const headers: Record<string, string> = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
		if (ssxmodItna) headers["Cookie"] = `ssxmod_itna=${ssxmodItna}`;
		const res = await fetch(`${QWEN_CHAT_INFO_URL}/${chatId}`, { method: "DELETE", headers });
		return res.ok;
	} catch { return false; }
}

function extractImagesFromMessages(messages: any[]): string[] {
	const images: string[] = [];
	for (const m of messages) {
		if (m.role === "user" && Array.isArray(m.content)) {
			for (const item of m.content) {
				if (item.type === "image_url" && item.image_url?.url) images.push(item.image_url.url);
			}
		}
	}
	return images;
}

function buildQwenMessage(content: string, files: any[], chatType: string, thinkingEnabled: boolean, qwenModel: string, parentId: string | null = null) {
	const isSearch = chatType === "search";
	const isResearch = chatType === "deep_research";

	return {
		fid: crypto.randomUUID(),
		parentId,
		childrenIds: [],
		role: "user",
		content,
		user_action: "chat",
		files,
		timestamp: Date.now(),
		models: [qwenModel],
		chat_type: chatType,
		feature_config: {
			thinking_enabled: thinkingEnabled,
			output_schema: "phase",
			research_mode: isResearch ? "deep_research" : "normal",
			auto_thinking: false,
			thinking_mode: "Thinking",
			thinking_format: "summary",
			auto_search: isSearch || isResearch,
			search_enabled: isSearch,
		},
		extra: {
			meta: {
				subChatType: isResearch ? "deep_thinking" : chatType,
				searchMode: isSearch ? "enable" : "disable"
			}
		},
		sub_chat_type: isResearch ? "deep_thinking" : chatType,
		parent_id: parentId,
	};
}

async function transformOpenAIRequestToQwen(openAIRequest: any, token: string, ssxmodItna?: string) {
	const requestedModel = openAIRequest.model || "qwen3.6-plus";
	const model = normalizeModelAlias(requestedModel);
	const resolvedType = resolveChatType(model);
	const qwenModel = model.replace(/-(search|thinking|image|image_edit|video|research)$/, "");
	const tools = normalizeOpenAITools(openAIRequest?.tools || []);
	const hasCustomTools = tools.length > 0;
	const thinkingEnabled = hasCustomTools ? false : model.includes("-thinking");

	const isResearch = resolvedType === "deep_research";
	const useTemp = !isResearch && config.sessionTemp;

	const allMessages = Array.isArray(openAIRequest?.messages) ? openAIRequest.messages : [];
	const lastUser = allMessages.filter((m: any) => m.role === "user").pop() || { content: "" };

	let chatTypeForCreation = resolvedType;
	if (resolvedType === "image_edit") {
		const imgs = extractImagesFromMessages(openAIRequest.messages);
		if (imgs.length === 0) chatTypeForCreation = "t2i";
	}

	if (resolvedType === "t2v") {
		const chatId = await createNewChat(token, qwenModel, "t2v", true, ssxmodItna);
		if (!chatId) throw new Error("create video chat failed");
		const text = typeof lastUser.content === "string" ? lastUser.content : "";
		const req = { stream: false, version: "2.1", incremental_output: true, chat_id: chatId, chat_mode: "normal", model: qwenModel, parent_id: null, messages: [buildQwenMessage(text || "Generate a video", [], "t2v", false, qwenModel)], timestamp: Date.now(), size: openAIRequest.size || "9:16" };
		return { request: req, chatId, isVideo: true, shouldAutoDelete: true };
	}

	if (resolvedType === "image_edit" || chatTypeForCreation === "t2i") {
		const chatId = await createNewChat(token, qwenModel, chatTypeForCreation, true, ssxmodItna);
		if (!chatId) throw new Error("create image chat failed");
		const imgs = extractImagesFromMessages(openAIRequest.messages);
		const text = typeof lastUser.content === "string" ? lastUser.content : "";
		const files = imgs.slice(-3).map(u => ({ type: "image", url: u }));
		const subType = resolvedType === "image_edit" && files.length > 0 ? "image_edit" : "t2i";
		const req = { stream: true, version: "2.1", incremental_output: true, chat_id: chatId, chat_mode: "normal", model: qwenModel, parent_id: null, messages: [buildQwenMessage(text || (subType === "image_edit" ? "Edit these images" : "Generate an image"), files, subType, false, qwenModel)], timestamp: Date.now(), ...(subType === "t2i" ? { size: openAIRequest.size || "1:1" } : {}) };
		return { request: req, chatId, isVideo: false, shouldAutoDelete: true };
	}

	const finalChatType = hasCustomTools ? "t2t" : (isResearch ? "deep_research" : resolvedType === "search" ? "search" : "t2t");
	let chatId: string, parentId: string | null = null;
	if (useTemp) {
		chatId = (await createNewChat(token, qwenModel, finalChatType, true, ssxmodItna)) || "";
	} else {
		const ctx = openAIRequest.qwen_context || {};
		if (ctx.chat_id) { chatId = ctx.chat_id; parentId = ctx.parent_id; } else {
			chatId = (await createNewChat(token, qwenModel, finalChatType, false, ssxmodItna)) || "";
		}
	}

	let text = "";
	const files: any[] = [];
	if (hasCustomTools) {
		text = buildPromptWithTools(allMessages, tools);
	} else if (typeof lastUser.content === "string") text = lastUser.content;
	else if (Array.isArray(lastUser.content)) {
		for (const i of lastUser.content) {
			if (i.type === "text") text += i.text || "";
			else if (i.type === "image_url") files.push({ type: "image", url: i.image_url.url });
		}
	}

	const message = buildQwenMessage(text, files, finalChatType, thinkingEnabled, qwenModel, parentId);

	let chatMode = "normal";
	if (finalChatType === "search") chatMode = "search";
	if (finalChatType === "deep_research") chatMode = "deep_research";

	const req: any = {
		stream: true,
		version: "2.1",
		incremental_output: true,
		chat_id: chatId,
		chat_mode: chatMode,
		model: qwenModel,
		parent_id: parentId,
		messages: [message],
		timestamp: Date.now(),
	};

	if (hasCustomTools) {
		req.messages[0].feature_config = {
			...req.messages[0].feature_config,
			thinking_enabled: false,
			auto_thinking: false,
			auto_search: false,
			code_interpreter: false,
			function_calling: true,
			plugins_enabled: false,
		};
	}

	return { request: req, chatId, isVideo: false, shouldAutoDelete: useTemp, hasCustomTools };
}

function createQwenToOpenAIStreamTransformer(options?: {
	hasCustomTools?: boolean;
	onComplete?: () => void | Promise<void>;
}) {
	const hasCustomTools = !!options?.hasCustomTools;
	const onComplete = options?.onComplete;
	const decoder = new TextDecoder();
	const encoder = new TextEncoder();
	let buffer = "";
	const messageId = crypto.randomUUID();
	let answerText = "";
	const nativeToolById: Record<string, { name: string; args: string }> = {};
	let roleSent = false;
	let finalFlushed = false;

	const enqueueJson = (controller: any, obj: any) => {
		controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
	};

	const mkChunk = (delta: any, finish: string | null = null) => ({
		id: `chatcmpl-${messageId}`,
		object: "chat.completion.chunk",
		created: Math.floor(Date.now() / 1000),
		model: "qwen-proxy",
		choices: [{ index: 0, delta, finish_reason: finish }],
	});

	const flushBufferedResult = (controller: any) => {
		if (finalFlushed) return;
		finalFlushed = true;

		if (!roleSent) {
			enqueueJson(controller, mkChunk({ role: "assistant" }, null));
			roleSent = true;
		}

		let parsedCalls: ParsedToolCall[] = [];
		if (Object.keys(nativeToolById).length > 0) {
			parsedCalls = Object.keys(nativeToolById).map((id) => {
				const tc = nativeToolById[id];
				return {
					id: id || `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
					name: tc.name,
					input: safeJsonParse(tc.args, { raw: tc.args }),
				};
			}).filter((tc) => !!tc.name);
		}

		if (parsedCalls.length === 0 && answerText.trim()) {
			parsedCalls = parseToolCallsFromText(answerText);
		}

		if (hasCustomTools && parsedCalls.length > 0) {
			parsedCalls.forEach((tc, idx) => {
				enqueueJson(controller, mkChunk({
					tool_calls: [{
						index: idx,
						id: tc.id,
						type: "function",
						function: { name: tc.name, arguments: "" },
					}],
				}, null));
				enqueueJson(controller, mkChunk({
					tool_calls: [{
						index: idx,
						function: { arguments: JSON.stringify(tc.input ?? {}, null, 0) },
					}],
				}, null));
			});
			enqueueJson(controller, mkChunk({}, "tool_calls"));
			return;
		}

		if (answerText) enqueueJson(controller, mkChunk({ content: answerText }, null));
		enqueueJson(controller, mkChunk({}, "stop"));
	};

	return new TransformStream({
		transform(chunk, controller) {
			const raw = decoder.decode(chunk, { stream: true });
			buffer += raw;

			logger.debug("Received raw chunk from Qwen", {
				length: raw.length,
				preview: raw.substring(0, 300),
			});

			const lines = buffer.split("\n\n");
			buffer = lines.pop() || "";

			for (const line of lines) {
				if (!line.trim()) continue;

				let dataStr = line.replace(/^data:\s*/, "").trim();
				logger.debug("Parsed line", { dataStr: dataStr.substring(0, 200) });

				if (dataStr === "[DONE]") {
					if (hasCustomTools) flushBufferedResult(controller);
					controller.enqueue(encoder.encode("data: [DONE]\n\n"));
					if (onComplete) onComplete();
					continue;
				}

				try {
					const qwenChunk = JSON.parse(dataStr);
					logger.debug("Parsed qwenChunk", { id: qwenChunk.id, hasChoices: !!qwenChunk.choices });

					let content = "";
					let phase = "";
					let phaseStatus = "";
					let phaseExtra: any = {};

					if (qwenChunk.choices && qwenChunk.choices[0]) {
						const delta = qwenChunk.choices[0].delta || qwenChunk.choices[0].message;
						content = delta?.content || "";
						phase = delta?.phase || "";
						phaseStatus = delta?.status || "";
						phaseExtra = delta?.extra || {};
					} else if (qwenChunk.content) {
						content = qwenChunk.content;
						phase = qwenChunk.phase || "";
						phaseStatus = qwenChunk.status || "";
						phaseExtra = qwenChunk.extra || {};
					}

					logger.debug("Extracted content", {
						length: content.length,
						preview: content.substring(0, 150),
						hasContent: !!content,
					});

					if (!hasCustomTools) {
						if (content) {
							const openAIChunk = {
								id: `chatcmpl-${messageId}`,
								object: "chat.completion.chunk",
								created: Math.floor(Date.now() / 1000),
								model: "qwen-proxy",
								choices: [{ index: 0, delta: { content }, finish_reason: null }],
							};
							controller.enqueue(encoder.encode(`data: ${JSON.stringify(openAIChunk)}\n\n`));
							logger.debug("✅ Enqueued OpenAI chunk", { contentLength: content.length });
						}
						continue;
					}

					if (content) {
						if (phase === "tool_call") {
							const tcId = phaseExtra?.tool_call_id || "tc_0";
							if (!nativeToolById[tcId]) nativeToolById[tcId] = { name: "", args: "" };
							const obj = safeJsonParse(content, null);
							if (obj && typeof obj === "object") {
								if (obj.name) nativeToolById[tcId].name = obj.name;
								if (obj.arguments) nativeToolById[tcId].args += String(obj.arguments);
							} else {
								nativeToolById[tcId].args += content;
							}
						} else {
							answerText += content;
						}
					}

					if (phaseStatus === "finished" && phase === "answer") {
						flushBufferedResult(controller);
					}
				} catch (e) {
					logger.debug("JSON parse failed", { error: (e as Error).message });
				}
			}
		},
		flush(controller) {
			logger.debug("Stream flush called");
			if (hasCustomTools) flushBufferedResult(controller);
			controller.enqueue(encoder.encode("data: [DONE]\n\n"));
		},
	});
}

const app = new Application();
const router = new Router();

// CORS 中间件
app.use(async (ctx, next) => {
	ctx.response.headers.set("Access-Control-Allow-Origin", "*");
	ctx.response.headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
	ctx.response.headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

	if (ctx.request.method === "OPTIONS") {
		ctx.response.status = 204;
		return;
	}

	await next();
});

app.use(async (ctx, next) => {
	const start = Date.now();
	try { await next(); } catch (e: any) {
		logger.error("Unhandled error", e);
		ctx.response.status = 500;
	}
	logger.request(ctx, start);
});

const authMiddleware: Middleware = async (ctx, next) => {
	if (ctx.request.url.pathname === "/") return await next();
	if (config.useDenoEnv) {
		ctx.state.qwenToken = config.qwenTokenEnv;
		ctx.state.ssxmodItna = config.ssxmodItnaEnv;
	} else {
		const header = ctx.request.headers.get("Authorization")?.replace(/^Bearer /, "") || "";
		if (!header) return ctx.throw(401, { error: "No Qwen token available." });
		const parts = header.split(";");
		ctx.state.qwenToken = config.salt ? (parts[1] || "").trim() : (parts[0] || "").trim();
	}
	await next();
};

app.use(authMiddleware);

router.get("/", (ctx) => {
	const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Qwen Proxy v5.0.8</title></head><body style="font-family:sans-serif;text-align:center;padding:80px;background:#0f172a;color:#fff"><h1>✅ 服务运行正常</h1><p>v5.0.8 支持工具调用（Function Calling）</p><p>API 文档请参考 README</p></body></html>`;
	ctx.response.body = html;
	ctx.response.headers.set("Content-Type", "text/html");
});

const handleModels = async (ctx: Context) => {
	const token = ctx.state.qwenToken;
	if (!token) return ctx.throw(401);
	ctx.response.body = { object: "list", data: [{ id: "qwen3.6-plus", object: "model" }] };
};
router.get("/v1/models", handleModels);
router.get("/models", handleModels);

const handleChatCompletions = async (ctx: Context) => {
	const token = ctx.state.qwenToken;
	if (!token) return ctx.throw(401, { error: "No Qwen token available." });

	try {
		const openAIRequest = await ctx.request.body({ type: "json" }).value;
		const { request: qwenRequest, chatId, isVideo, shouldAutoDelete, hasCustomTools } = await transformOpenAIRequestToQwen(openAIRequest, token, ctx.state.ssxmodItna);

		const url = `${QWEN_API_BASE_URL}?chat_id=${chatId}`;
		const headers = { Authorization: `Bearer ${token}`, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0" };

		logger.info("Sending to Qwen", { chatId, model: qwenRequest.model });

		const upstream = await fetch(url, { method: "POST", headers, body: JSON.stringify(qwenRequest) });

		if (!upstream.ok) {
			const text = await upstream.text();
			logger.error("Upstream error", { status: upstream.status, body: text });
			if (shouldAutoDelete && chatId) await deleteChat(chatId, token);
			ctx.response.status = upstream.status;
			ctx.response.body = { error: "Upstream failed", details: text };
			return;
		}

		logger.info("Upstream response OK, starting stream");

		ctx.response.headers.set("Content-Type", "text/event-stream");
		ctx.response.headers.set("Cache-Control", "no-cache");
		ctx.response.headers.set("Connection", "keep-alive");

		const onComplete = shouldAutoDelete ? async () => { if (chatId) await deleteChat(chatId, token); } : undefined;

		const transformed = upstream.body!.pipeThrough(createQwenToOpenAIStreamTransformer({ hasCustomTools, onComplete }));

		if (openAIRequest?.stream === false) {
			const rawSseText = await new Response(transformed).text();
			ctx.response.headers.set("Content-Type", "application/json");
			ctx.response.body = parseOpenAIStreamToCompletion(rawSseText, qwenRequest.model || "qwen-proxy");
		} else {
			ctx.response.body = transformed;
		}

	} catch (e: any) {
		logger.error("handleChatCompletions error", e);
		ctx.response.status = 500;
		ctx.response.body = { error: e.message };
	}
};

router.post("/v1/chat/completions", handleChatCompletions);
router.post("/chat/completions", handleChatCompletions);

router.get("/health", (ctx) => { ctx.response.body = { status: "healthy", version: "5.0.8" }; });

app.use(router.routes());
app.use(router.allowedMethods());

app.use((ctx) => { ctx.response.status = 404; ctx.response.body = { error: "Not Found" }; });

console.log("🚀 Qwen Proxy v5.0.8 启动 - 支持工具调用");
Deno.serve((req) => app.handle(req));
