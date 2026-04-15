/**
 * Qwen API to OpenAI Standard - Single File Deno Deploy/Playground Script
 *
 * @version 5.0.9
 * @description 完全按照官方 payload + CORS 支持 + 搜索模式修复 + 增强工具调用支持
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

type InferredIntent = "read" | "create" | "delete" | "run" | "edit" | "list" | "search" | "unknown";

type PathHints = {
	homeHint?: string;
	knownPaths: string[];
};

const WINDOWS_PATH_REGEX = /[A-Za-z]:[\\/](?:[^\s"'`<>]+[\\/])*[^\s"'`<>]+/g;
const UNIX_PATH_REGEX = /\/(?:[^\s"'`<>]+\/)*[^\s"'`<>]+/g;
const TOOL_NAME_ALIASES: Record<string, string[]> = {
	read: ["view", "cat", "open_file"],
	list: ["ls", "dir", "tree"],
	bash: ["run", "shell", "exec", "command"],
	apply_patch: ["patch"],
	edit: ["multiedit", "str_replace_editor"],
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

function normalizeToolArguments(argumentsText: string): any {
	if (!argumentsText) return {};
	const parsed = safeJsonParse(argumentsText, null);
	if (parsed && typeof parsed === "object") return parsed;
	return { raw: argumentsText };
}

function resolveForcedToolName(toolChoice: any): string | null {
	if (!toolChoice) return null;
	if (typeof toolChoice === "string") {
		if (toolChoice === "none" || toolChoice === "auto" || toolChoice === "required") return null;
	}
	if (typeof toolChoice === "object") {
		return toolChoice?.function?.name || null;
	}
	return null;
}

function buildPromptWithTools(messages: any[], tools: OpenAITool[], forcedToolName?: string | null): string {
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
		const normalizedForcedTool = forcedToolName && toolSpec.some((t) => t.name === forcedToolName)
			? forcedToolName
			: null;
		lines.push(
			"[Tools]\n" +
			JSON.stringify(toolSpec, null, 2) +
			"\n[/Tools]\n" +
			"You are operating as a coding agent with access to workspace tools.\n" +
			"For any request involving local files, directories, code search, shell commands, or code changes, you MUST use the provided tools instead of answering from memory.\n" +
			"Never claim you inspected, created, edited, or deleted a local file unless you emitted a tool call and later received a tool result confirming it.\n" +
			"Prefer one tool call at a time. After a tool result, either emit the next tool call or provide the final answer if the task is complete.\n" +
			"When the user asks to inspect a file, use a read/view tool first. When the user asks to inspect a directory or project structure, use a list/glob/grep tool first.\n" +
			"When the user asks to modify files, use edit/write/apply_patch tools. Do not answer with hypothetical edits.\n" +
			"If a tool is needed, output exactly one tool call block in this format:\n" +
			"##TOOL_CALL##\n{\"name\":\"tool_name\",\"input\":{...}}\n##END_CALL##\n" +
			"Do not add markdown fences around the JSON.\n" +
			"Never claim a listed tool does not exist; choose only from the listed tools."
		);
		if (normalizedForcedTool) {
			lines.push(
				"[ToolChoice]\n" +
				`You MUST call this tool now: ${normalizedForcedTool}.` +
				"\nReturn only the tool call block; do not output normal assistant text.\n" +
				"[/ToolChoice]"
			);
		}
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

function collectPathHintsFromMessages(messages: any[]): PathHints {
	const pathSet = new Set<string>();
	for (const msg of messages || []) {
		const raw = typeof msg?.content === "string"
			? msg.content
			: Array.isArray(msg?.content)
				? msg.content.map((x: any) => (x?.type === "text" ? x?.text || "" : "")).join("\n")
				: "";
		const text = String(raw || "");
		const matches = [
			...(text.match(UNIX_PATH_REGEX) || []),
			...(text.match(WINDOWS_PATH_REGEX) || []),
		];
		for (const p of matches) pathSet.add(p);
	}
	const knownPaths = Array.from(pathSet);
	const homeFromKnown = knownPaths
		.map((p) => p.match(/^(\/home\/[^\/]+)/)?.[1] || p.match(/^([A-Za-z]:[\\/][^\\/]+)/)?.[1] || "")
		.find(Boolean);
	return { homeHint: homeFromKnown || undefined, knownPaths };
}

function extractPathFromText(text: string, hints?: PathHints): string | null {
	const input = (text || "").trim();
	if (!input) return null;
	const knownPaths = hints?.knownPaths || [];
	const home = hints?.homeHint;
	const tilde = input.match(/~\/[^\s"'`]+/);
	if (tilde?.[0]) {
		if (home) return `${home}/${tilde[0].slice(2)}`;
		const tail = `/${tilde[0].slice(2)}`;
		const hit = knownPaths.find((p) => p.endsWith(tail));
		if (hit) return hit;
		return null;
	}
	const winAbs = input.match(/[A-Za-z]:[\\/](?:[^\s"'`<>]+[\\/])*[^\s"'`<>]+/);
	if (winAbs?.[0]) return winAbs[0];
	const abs = input.match(/\/(?:[^\s"'`<>]+\/)*[^\s"'`<>]+/);
	if (abs?.[0]) return abs[0];
	const rel = input.match(/(?:\.\/|\.\\)?[A-Za-z0-9_.\-\\/]+\.[A-Za-z0-9_\-]+/);
	if (rel?.[0]) return rel[0].replace(/^\.\//, "").replace(/^\.\\/, "");
	if (/(\.bashrc|bashrc)/i.test(input)) {
		const knownBashrc = knownPaths.find((p) => p.endsWith("/.bashrc"));
		if (knownBashrc) return knownBashrc;
		if (home) return `${home}/.bashrc`;
	}
	return null;
}

function inferIntentFromText(text: string): InferredIntent {
	const lower = (text || "").toLowerCase();
	if (!lower) return "unknown";
	if (/(list|ls|dir|tree|目录|文件列表|列出|遍历|结构)/.test(lower)) return "list";
	if (/(search|find|grep|glob|ripgrep|查找|搜索|检索|匹配)/.test(lower)) return "search";
	if (/(read|show|view|open|cat|读取|阅读|查看|打开|显示)/.test(lower)) return "read";
	if (/(create|new|touch|mk|write file|创建|新建)/.test(lower)) return "create";
	if (/(delete|remove|rm|unlink|删除|移除|删掉)/.test(lower)) return "delete";
	if (/(edit|modify|update|replace|修改|编辑|替换)/.test(lower)) return "edit";
	if (/(run|execute|bash|shell|cmd|命令|执行)/.test(lower)) return "run";
	return "unknown";
}

function getToolSchema(tool: OpenAITool): any {
	return tool?.function?.parameters && typeof tool.function.parameters === "object"
		? tool.function.parameters
		: { type: "object", properties: {} };
}

function getToolRequiredKeys(tool: OpenAITool): string[] {
	const schema = getToolSchema(tool);
	return Array.isArray(schema?.required) ? schema.required : [];
}

function getToolPropertyMap(tool: OpenAITool): Record<string, any> {
	const schema = getToolSchema(tool);
	return schema?.properties && typeof schema.properties === "object" ? schema.properties : {};
}

function pickFirstKey(keys: string[], re: RegExp): string | null {
	for (const key of keys) {
		if (re.test(key)) return key;
	}
	return null;
}

function escapeRegExp(input: string): string {
	return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildArgsForTool(tool: OpenAITool, intent: InferredIntent, userText: string, hints?: PathHints): any {
	const props = getToolPropertyMap(tool);
	const required = getToolRequiredKeys(tool);
	const keys = Object.keys(props);
	const lowerText = (userText || "").toLowerCase();
	const path = extractPathFromText(userText, hints);
	const args: Record<string, any> = {};

	const pathKey = pickFirstKey(keys, /(filepath|file_path|path|filename|file|target)/i);
	const contentKey = pickFirstKey(keys, /(content|text|body|data)/i);
	const commandKey = pickFirstKey(keys, /(command|cmd|script)/i);
	const descriptionKey = pickFirstKey(keys, /(description|desc|reason)/i);
	const oldStringKey = pickFirstKey(keys, /(oldstring|old_string)/i);
	const newStringKey = pickFirstKey(keys, /(newstring|new_string)/i);

	if (path && pathKey) args[pathKey] = path;
	if (intent === "list" && pathKey && args[pathKey] === undefined) args[pathKey] = ".";

	if (intent === "delete" && commandKey && path) {
		const escaped = path.replace(/"/g, '\\"');
		args[commandKey] = `rm -f "${escaped}"`;
	}

	if ((intent === "create" || intent === "edit") && contentKey && args[contentKey] === undefined) {
		args[contentKey] = "";
	}
	if (intent === "edit" && oldStringKey && args[oldStringKey] === undefined) args[oldStringKey] = "";
	if ((intent === "create" || intent === "edit") && newStringKey && args[newStringKey] === undefined) args[newStringKey] = "";

	if (descriptionKey && args[descriptionKey] === undefined) {
		if (intent === "delete" && path) args[descriptionKey] = `Delete file ${path}`;
		else if (intent === "read" && path) args[descriptionKey] = `Read file ${path}`;
		else if (intent === "create" && path) args[descriptionKey] = `Create file ${path}`;
		else if (intent === "list") args[descriptionKey] = `List directory ${path || "."}`;
		else if (intent === "search") args[descriptionKey] = "Search the codebase";
		else if (intent === "run") args[descriptionKey] = "Run command";
	}

	if (commandKey && args[commandKey] === undefined && intent === "run") {
		if (/\bls\b|list|目录|文件列表/.test(lowerText)) args[commandKey] = "ls";
		else if (/\bpwd\b|当前目录/.test(lowerText)) args[commandKey] = "pwd";
	}

	for (const key of required) {
		if (args[key] !== undefined) continue;
		const def = props[key] || {};
		const type = def?.type;
		if (type === "string") args[key] = "";
		else if (type === "number" || type === "integer") args[key] = 0;
		else if (type === "boolean") args[key] = false;
		else if (type === "array") args[key] = [];
		else args[key] = {};
	}

	return args;
}

function scoreToolForIntent(tool: OpenAITool, intent: InferredIntent, userText: string, hints?: PathHints): number {
	const name = (tool?.function?.name || "").toLowerCase();
	const desc = (tool?.function?.description || "").toLowerCase();
	const text = (userText || "").toLowerCase();
	const props = Object.keys(getToolPropertyMap(tool)).join(" ").toLowerCase();
	let score = 0;

	if (name && new RegExp(`\\b${escapeRegExp(name)}\\b`, "i").test(text)) score += 8;

	if (intent === "read" && /(read|view|open|cat|读取|查看)/.test(`${name} ${desc} ${props}`)) score += 6;
	if (intent === "create" && /(write|create|new|touch|创建|新建)/.test(`${name} ${desc} ${props}`)) score += 6;
	if (intent === "delete" && /(delete|remove|rm|bash|shell|删除|移除)/.test(`${name} ${desc} ${props}`)) score += 6;
	if (intent === "edit" && /(edit|modify|update|replace|修改|编辑)/.test(`${name} ${desc} ${props}`)) score += 6;
	if (intent === "run" && /(bash|shell|command|exec|run|执行|命令)/.test(`${name} ${desc} ${props}`)) score += 6;
	if (intent === "list" && /(list|ls|dir|tree|目录|列出)/.test(`${name} ${desc} ${props}`)) score += 7;
	if (intent === "search" && /(grep|glob|search|find|查找|搜索)/.test(`${name} ${desc} ${props}`)) score += 7;

	if (intent === "create") {
		if (name === "write") score += 12;
		if (name === "edit") score -= 10;
	}
	if (intent === "read" && (name === "read" || name === "view")) score += 10;
	if (intent === "list" && name === "list") score += 10;
	if (intent === "search" && (name === "grep" || name === "glob")) score += 10;
	if (intent === "run" && name === "bash") score += 10;
	if (intent === "edit" && name === "edit") score += 8;

	if (extractPathFromText(userText, hints) && /(filepath|file_path|path|filename|file|target)/.test(props)) score += 3;

	return score;
}

function findToolByName(tools: OpenAITool[], name: string): OpenAITool | null {
	for (const tool of tools || []) {
		if (tool?.function?.name === name) return tool;
	}
	return null;
}

function findAllowedToolName(tools: OpenAITool[], requestedName: string): string {
	const wanted = (requestedName || "").trim();
	if (!wanted) return "";

	const exact = findToolByName(tools, wanted);
	if (exact?.function?.name) return exact.function.name;

	const lower = wanted.toLowerCase();
	for (const tool of tools || []) {
		if ((tool?.function?.name || "").toLowerCase() === lower) return tool.function!.name!;
	}

	for (const [canonical, aliases] of Object.entries(TOOL_NAME_ALIASES)) {
		if (lower === canonical || aliases.includes(lower)) {
			for (const tool of tools || []) {
				if ((tool?.function?.name || "").toLowerCase() === canonical) return tool.function!.name!;
			}
		}
	}

	return wanted;
}

function syncKnownArgumentAliases(input: Record<string, any>, tool: OpenAITool | null) {
	if (!tool || !input || typeof input !== "object") return input;

	const keys = new Set(Object.keys(getToolPropertyMap(tool)));
	const pathValue = input.filePath ?? input.file_path ?? input.path ?? input.filename ?? input.file ?? input.target;
	const oldValue = input.oldString ?? input.old_string;
	const newValue = input.newString ?? input.new_string ?? input.content ?? input.text ?? input.body ?? input.data;
	const commandValue = input.command ?? input.cmd ?? input.script;
	const patchValue = input.patchText ?? input.patch_text ?? input.diff ?? input.patch;

	if (pathValue !== undefined) {
		for (const key of keys) {
			if (/(filepath|file_path|path|filename|file|target)/i.test(key)) input[key] = pathValue;
		}
	}
	if (oldValue !== undefined) {
		for (const key of keys) {
			if (/(oldstring|old_string)/i.test(key)) input[key] = oldValue;
		}
	}
	if (newValue !== undefined) {
		for (const key of keys) {
			if (/(newstring|new_string|content|text|body|data)/i.test(key)) input[key] = newValue;
		}
	}
	if (commandValue !== undefined) {
		for (const key of keys) {
			if (/(command|cmd|script)/i.test(key)) input[key] = commandValue;
		}
	}
	if (patchValue !== undefined) {
		for (const key of keys) {
			if (/(patchtext|patch_text|diff)/i.test(key)) input[key] = patchValue;
		}
	}

	return input;
}

function normalizeParsedToolCalls(calls: ParsedToolCall[], tools: OpenAITool[], userText: string, hints?: PathHints): ParsedToolCall[] {
	if (!Array.isArray(calls) || calls.length === 0) return [];
	const intent = inferIntentFromText(userText);
	const writeTool = findToolByName(tools, "write");
	const editTool = findToolByName(tools, "edit");
	const allowedToolNames = new Set((tools || []).map((t) => t?.function?.name).filter(Boolean) as string[]);

	const normalized = calls.map((call) => {
		const callName = findAllowedToolName(tools, call?.name || "");
		const baseTool = findToolByName(tools, callName);
		const baseArgs = baseTool ? buildArgsForTool(baseTool, intent, userText, hints) : {};
		const input = call?.input && typeof call.input === "object" ? { ...baseArgs, ...call.input } : { ...baseArgs };
		syncKnownArgumentAliases(input, baseTool);
		const filePath = input.filePath || input.file_path || input.path || extractPathFromText(userText, hints) || "";

		if (intent === "create" && callName === "edit" && writeTool) {
			const writeArgs = buildArgsForTool(writeTool, "create", userText, hints);
			const replacementText = String(input.newString ?? input.new_string ?? "");
			if (replacementText) {
				const contentKey = pickFirstKey(Object.keys(getToolPropertyMap(writeTool)), /(content|text|body|data|newstring|new_string)/i);
				if (contentKey) writeArgs[contentKey] = replacementText;
			}
			syncKnownArgumentAliases(writeArgs, writeTool);
			return { ...call, name: "write", input: writeArgs };
		}

		if (callName === "write" && writeTool) {
			syncKnownArgumentAliases(input, writeTool);
			return { ...call, name: callName, input };
		}

		if (callName === "edit" && editTool) {
			syncKnownArgumentAliases(input, editTool);
			return { ...call, name: callName, input };
		}

		if (callName === "read") {
			syncKnownArgumentAliases(input, baseTool);
			return { ...call, name: callName, input };
		}

		if (callName === "list") {
			syncKnownArgumentAliases(input, baseTool);
			return { ...call, name: callName, input };
		}

		if (callName === "bash") {
			syncKnownArgumentAliases(input, baseTool);
			return { ...call, name: callName, input };
		}

		if (callName === "apply_patch") {
			syncKnownArgumentAliases(input, baseTool);
			return { ...call, name: callName, input };
		}

		return { ...call, name: callName, input };
	});

	const validCalls = normalized.filter((c) => allowedToolNames.has(c.name));
	if (validCalls.length > 0) return validCalls;

	const inferred = inferToolCallFromIntent(userText, tools, hints);
	return inferred ? [inferred] : [];
}

function inferToolCallFromIntent(userText: string, tools: OpenAITool[], hints?: PathHints): ParsedToolCall | null {
	const text = (userText || "").trim();
	if (!text || !Array.isArray(tools) || tools.length === 0) return null;

	const intent = inferIntentFromText(text);
	const scored = tools
		.map((tool) => ({ tool, score: scoreToolForIntent(tool, intent, text, hints) }))
		.sort((a, b) => b.score - a.score);

	const best = scored[0];
	if (!best || best.score <= 0 || !best.tool?.function?.name) return null;

	const args = buildArgsForTool(best.tool, intent, text, hints);
	const required = getToolRequiredKeys(best.tool);
	const allRequiredPresent = required.every((k) => args[k] !== undefined);
	if (!allRequiredPresent) return null;
	if ((intent === "read" || intent === "create" || intent === "delete") && !extractPathFromText(text, hints)) return null;

	return {
		id: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
		name: best.tool.function.name,
		input: args,
	};
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
	const forcedToolName = resolveForcedToolName(openAIRequest?.tool_choice);
	const thinkingEnabled = hasCustomTools ? false : model.includes("-thinking");

	const isResearch = resolvedType === "deep_research";
	const useTemp = !isResearch && config.sessionTemp;

	const allMessages = Array.isArray(openAIRequest?.messages) ? openAIRequest.messages : [];
	const lastUser = allMessages.filter((m: any) => m.role === "user").pop() || { content: "" };
	const lastTool = allMessages.filter((m: any) => m.role === "tool").pop() || { content: "" };
	const lastAssistant = allMessages.filter((m: any) => m.role === "assistant" && Array.isArray(m?.tool_calls) && m.tool_calls.length > 0).pop() || { tool_calls: [] };
	const lastUserText = extractTextContent(lastUser?.content);
	const lastToolResultText = extractTextContent(lastTool?.content) || (typeof lastTool?.content === "string" ? lastTool.content : "");
	const lastAssistantToolName = lastAssistant?.tool_calls?.[0]?.function?.name || "";
	const lastAssistantToolArgsText = typeof lastAssistant?.tool_calls?.[0]?.function?.arguments === "string"
		? lastAssistant.tool_calls[0].function.arguments
		: JSON.stringify(lastAssistant?.tool_calls?.[0]?.function?.arguments || {});
	const lastMessageRole = String(allMessages[allMessages.length - 1]?.role || "");
	const hadRecentToolSuccess = lastMessageRole === "tool" && !!lastToolResultText && !/(file\s+not\s+found|permission\s+denied|not\s+found|failed|error|unexpected\s+eof)/i.test(lastToolResultText);
	const pathHints = collectPathHintsFromMessages(allMessages);
	logger.debug("Tool request context", {
		hasCustomTools,
		forcedToolName,
		toolNames: tools.map((tool) => tool?.function?.name).filter(Boolean),
		lastMessageRole,
		lastUserText: lastUserText.slice(0, 200),
	});

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
		text = buildPromptWithTools(allMessages, tools, forcedToolName);
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

	return { request: req, chatId, isVideo: false, shouldAutoDelete: useTemp, hasCustomTools, forcedToolName, lastUserText, lastToolResultText, lastAssistantToolName, lastAssistantToolArgsText, lastMessageRole, hadRecentToolSuccess, pathHints, tools };
}

function createQwenToOpenAIStreamTransformer(options?: {
	hasCustomTools?: boolean;
	forcedToolName?: string | null;
	fallbackUserText?: string;
	fallbackTools?: OpenAITool[];
	lastToolResultText?: string;
	lastAssistantToolName?: string;
	lastAssistantToolArgsText?: string;
	lastMessageRole?: string;
	hadRecentToolSuccess?: boolean;
	pathHints?: PathHints;
	onComplete?: () => void | Promise<void>;
}) {
	const hasCustomTools = !!options?.hasCustomTools;
	const forcedToolName = options?.forcedToolName || null;
	const fallbackUserText = options?.fallbackUserText || "";
	const fallbackTools = options?.fallbackTools || [];
	const lastToolResultText = options?.lastToolResultText || "";
	const lastAssistantToolName = options?.lastAssistantToolName || "";
	const lastAssistantToolArgsText = options?.lastAssistantToolArgsText || "";
	const lastMessageRole = options?.lastMessageRole || "";
	const hadRecentToolSuccess = !!options?.hadRecentToolSuccess;
	const pathHints = options?.pathHints || { knownPaths: [] };
	const hadToolError = /(file\s+not\s+found|permission\s+denied|not\s+found|failed|error)/i.test(lastToolResultText);
	const lastArgs = normalizeToolArguments(lastAssistantToolArgsText);
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

		if (parsedCalls.length > 0 && hasCustomTools) {
			parsedCalls = normalizeParsedToolCalls(parsedCalls, fallbackTools, fallbackUserText, pathHints);
		}

		if (parsedCalls.length === 0 && hasCustomTools) {
			const looksLikeMissingToolText = /tool\s+\w+\s+does\s+not\s+exists?/i.test(answerText);
			const shouldInferByEmpty = !answerText.trim() && lastMessageRole === "user" && !hadRecentToolSuccess;
			if (looksLikeMissingToolText || shouldInferByEmpty) {
				const inferred = inferToolCallFromIntent(fallbackUserText, fallbackTools, pathHints);
				if (inferred) {
					const sameFailedCall = hadToolError &&
						!!lastAssistantToolName &&
						inferred.name === lastAssistantToolName &&
						JSON.stringify(inferred.input ?? {}) === JSON.stringify(lastArgs ?? {});
					const sameAsLastCall = !!lastAssistantToolName &&
						inferred.name === lastAssistantToolName &&
						JSON.stringify(inferred.input ?? {}) === JSON.stringify(lastArgs ?? {});
					if (!sameFailedCall && !(lastMessageRole === "tool" && sameAsLastCall)) {
						parsedCalls = [inferred];
					}
				}
			}
		}

		if (parsedCalls.length === 0 && hasCustomTools && forcedToolName) {
			const forcedExists = fallbackTools.some((t) => t?.function?.name === forcedToolName);
			if (forcedExists) {
				parsedCalls = [{
					id: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
					name: forcedToolName,
					input: {},
				}];
			}
		}

		if (parsedCalls.length > 0 && hasCustomTools) {
			parsedCalls = normalizeParsedToolCalls(parsedCalls, fallbackTools, fallbackUserText, pathHints);
		}

		if (hasCustomTools && lastMessageRole === "tool" && parsedCalls.length > 0) {
			parsedCalls = parsedCalls.filter((tc) => !(
				!!lastAssistantToolName &&
				tc.name === lastAssistantToolName &&
				JSON.stringify(tc.input ?? {}) === JSON.stringify(lastArgs ?? {})
			));
		}

		if (hasCustomTools && parsedCalls.length > 0) {
			logger.debug("Resolved tool calls", {
				parsedCalls,
				answerPreview: answerText.slice(0, 200),
				lastMessageRole,
			});
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
	const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Qwen Proxy v5.0.9</title></head><body style="font-family:sans-serif;text-align:center;padding:80px;background:#0f172a;color:#fff"><h1>✅ 服务运行正常</h1><p>v5.0.9 支持增强工具调用（Function Calling）</p><p>API 文档请参考 README</p></body></html>`;
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
		const { request: qwenRequest, chatId, isVideo, shouldAutoDelete, hasCustomTools, forcedToolName, lastUserText, lastToolResultText, lastAssistantToolName, lastAssistantToolArgsText, lastMessageRole, hadRecentToolSuccess, pathHints, tools } = await transformOpenAIRequestToQwen(openAIRequest, token, ctx.state.ssxmodItna);

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

		const transformed = upstream.body!.pipeThrough(createQwenToOpenAIStreamTransformer({
			hasCustomTools,
			forcedToolName,
			fallbackUserText: lastUserText,
			lastToolResultText,
			lastAssistantToolName,
			lastAssistantToolArgsText,
			lastMessageRole,
			hadRecentToolSuccess,
			pathHints,
			fallbackTools: tools,
			onComplete,
		}));

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

router.get("/health", (ctx) => { ctx.response.body = { status: "healthy", version: "5.0.9" }; });

app.use(router.routes());
app.use(router.allowedMethods());

app.use((ctx) => { ctx.response.status = 404; ctx.response.body = { error: "Not Found" }; });

console.log("🚀 Qwen Proxy v5.0.9 启动 - 支持增强工具调用");
Deno.serve((req) => app.handle(req));
