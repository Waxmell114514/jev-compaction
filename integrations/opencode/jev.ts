/**
 * Jev admission gate for OpenCode.
 *
 * Talks to a `python -m jevctx.serve` sidecar. Every text result from a read-only
 * tool goes through `admit()` in `tool.execute.after`, before OpenCode stores it,
 * so the transcript sent to the model stays an append-only prefix. Low-scoring runs
 * become `[[elided id=...]]` pointers; the model gets `expand` for an exact id and
 * `recall` for a description.
 *
 *   ~/.config/opencode/plugins/jev.ts   (or .opencode/plugins/ in a project)
 *
 * It imports `@opencode-ai/plugin` (and through it `zod`), so the directory it is
 * loaded from needs those packages in a reachable node_modules.
 *
 * Environment:
 *   JEV_MODE     on (default) | shadow (score and log, change nothing) | off
 *   JEV_URL      sidecar URL (required; start `python -m jevctx.serve` yourself)
 *   JEV_SESSION  store/log name on the sidecar (default: oc-<OpenCode session id>)
 *   JEV_TOOLS    tools to gate (default: bash,read,grep,glob,list)
 *   JEV_TASK     task digest; defaults to the session's first user message
 *   JEV_MAX_ELIDE_FRACTION  override the sidecar's tripwire; 1 disables it
 *   JEV_PROFILE  1 asks every segment's type, role, lifetime and injection risk
 *   JEV_WORKAREA 1 also offers the transcript's tail for compaction before each request
 *                (jevctx.workarea): tool outputs since the last commit point that have
 *                served their purpose become pointers, when the sidecar's price
 *                arithmetic says the cache break pays. Replacements are request-local
 *                (OpenCode's stored session is untouched) and reapplied on every request.
 *   JEV_INTENT   1 judges each output against what the model said it was doing when it made
 *                the call (the text, else the reasoning, of the same assistant message) as
 *                well as the task
 *
 * Every tool call's arguments go to the sidecar too -- with the output for gated tools,
 * on their own (/observe) for the rest -- so it knows what each output viewed, ran or
 * searched and which files were written since (jevctx.supersede): outputs a later call
 * made obsolete are marked when expanded or recalled, and compacted first.
 *
 * The gate fails open: if the sidecar or Jev is unreachable, the original result
 * goes through unchanged and a line is written to stderr.
 */

import { type Plugin, tool } from "@opencode-ai/plugin";

type Mode = "off" | "shadow" | "on";

const mode = (process.env.JEV_MODE ?? "on") as Mode;
const baseUrl = process.env.JEV_URL;
const gatedTools = new Set((process.env.JEV_TOOLS ?? "bash,read,grep,glob,list").split(",").map((s) => s.trim()));
const maxElideFraction = process.env.JEV_MAX_ELIDE_FRACTION ? Number(process.env.JEV_MAX_ELIDE_FRACTION) : undefined;
const profile = process.env.JEV_PROFILE ? process.env.JEV_PROFILE === "1" : undefined;
const workarea = process.env.JEV_WORKAREA === "1";
const withIntent = process.env.JEV_INTENT === "1";

// OpenCode wraps a read as "<path>…</path>\n<type>file</type>\n<content>\n…\n\n(Showing lines
// 1-2000 of 3000. Use offset=2001 to continue.)\n</content>". The wrapper and the notice tell the
// model where it is and how to read on, so only the body between them is offered to the gate.
const READ_WRAPPER = /^(<path>[\s\S]*?<content>\n)([\s\S]*?)((?:\n\n\([^\n]*\))?\n?<\/content>\s*)$/;
// Other tools may end with a one-line "(...)" or "[...]" notice after a blank line.
const TRAILING_NOTICE = /\n\n[([][^\n]*[)\]]\s*$/;

function split(text: string): [string, string, string] {
	const wrapped = text.match(READ_WRAPPER);
	if (wrapped) return [wrapped[1], wrapped[2], wrapped[3]];
	const notice = text.match(TRAILING_NOTICE)?.[0] ?? "";
	return ["", text.slice(0, text.length - notice.length), notice];
}

const tasks = new Map<string, string>();
// Every compaction the sidecar ever returned, per session: reapplied on every request so
// the transcript is rewritten identically each time, even if the sidecar is unreachable.
const compacted = new Map<string, Record<string, string>>();
const estimate = (text: string) => Math.ceil(text.length / 4);
const turns = new Map<string, number>();
const sessionName = (sessionID: string) => process.env.JEV_SESSION ?? `oc-${sessionID}`;

// What the model said before each call. Parts stream in as events: the latest text and
// reasoning of every assistant message, and, when a tool part first appears, a snapshot
// of its message's words taken for that call. A tool part only ever belongs to an
// assistant message, so a user's text never becomes an intent.
const said = new Map<string, { text?: string; reasoning?: string }>();
const intents = new Map<string, string>();
const INTENT_CHARS = 2000;
function remember(part: any): void {
	if (part.type === "text" || part.type === "reasoning") {
		if (part.synthetic || typeof part.text !== "string") return;
		const words = said.get(part.messageID) ?? {};
		words[part.type as "text" | "reasoning"] = part.text;
		said.delete(part.messageID); // re-insert: the map stays in last-updated order
		said.set(part.messageID, words);
		if (said.size > 256) said.delete(said.keys().next().value as string);
	} else if (part.type === "tool" && !intents.has(part.callID)) {
		const words = said.get(part.messageID);
		const intent = (words?.text?.trim() || words?.reasoning?.trim() || "").slice(-INTENT_CHARS);
		if (intent) intents.set(part.callID, intent);
		if (intents.size > 256) intents.delete(intents.keys().next().value as string);
	}
}

async function call(path: string, body: Record<string, unknown>, signal?: AbortSignal): Promise<any> {
	if (!baseUrl) throw new Error("JEV_URL is not set");
	const response = await fetch(baseUrl + path, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
		signal,
	});
	const payload = await response.json();
	if (!response.ok) throw new Error(payload?.error ?? `HTTP ${response.status}`);
	return payload;
}

// Mirrors jevctx.profile.TYPE_QUESTION / ROLE_QUESTION; the sidecar rejects anything else.
const TYPES = ["source_code", "test_code", "test_output", "error", "search_results", "file_listing",
	"diff", "configuration", "documentation", "log", "other"] as const;
const ROLES = ["change_site", "evidence", "reference", "verification", "navigation", "background",
	"noise"] as const;

const GUIDANCE =
	"Some tool results contain [[elided id=... \"summary\"]] lines standing in for output judged " +
	"low-value; the summary says what kind of output it was. Call `expand` with that id before relying " +
	"on, quoting or editing what the pointer stands for. When you need something you saw earlier but it " +
	"is no longer in the conversation, or you only roughly remember it, call `recall` with a description " +
	"(and a name, type or role filter if you know one) instead of re-running the command.";

export const JevPlugin: Plugin = async ({ directory }) => {
	if (mode === "off") return {};
	if (mode !== "on" && mode !== "shadow") throw new Error(`JEV_MODE must be on, shadow or off, not ${mode}`);

	const tools =
		mode === "on"
			? {
					expand: tool({
						description:
							"Retrieve the full original text behind an [[elided id=... ]] pointer in a tool result.",
						args: { id: tool.schema.string().describe("The id from the pointer, e.g. r:7f3a91e2") },
						async execute(args, context) {
							const { text } = await call("/expand", {
								session: sessionName(context.sessionID),
								id: args.id,
								turn: turns.get(context.sessionID) ?? 0,
							}, context.abort);
							return text;
						},
					}),
					recall: tool({
						description:
							"Search earlier tool output from this session, including output that was elided or is no " +
							"longer in the conversation. Describe what you are looking for; optionally narrow by a " +
							"file/function/error name it mentions, what kind of output it was, or what it was for. " +
							"Returns original text.",
						args: {
							query: tool.schema.string().describe("What you are looking for, in words"),
							name: tool.schema.string().optional().describe("A path, function, class, error or test name it mentions"),
							type: tool.schema.enum(TYPES).optional().describe("What kind of output it was"),
							role: tool.schema.enum(ROLES).optional()
								.describe("What it was for: change_site = code to change, evidence = shows the problem, ..."),
						},
						async execute(args, context) {
							const { text } = await call("/recall", {
								session: sessionName(context.sessionID),
								...args,
								turn: turns.get(context.sessionID) ?? 0,
							}, context.abort);
							return text;
						},
					}),
				}
			: {};

	return {
		tool: tools,

		event: async ({ event }) => {
			if (withIntent && event.type === "message.part.updated") remember(event.properties.part);
		},

		"chat.message": async (input, output) => {
			if (!tasks.has(input.sessionID)) {
				const text = output.parts
					.map((part) => (part.type === "text" ? part.text : ""))
					.join("\n")
					.trim();
				if (text) tasks.set(input.sessionID, process.env.JEV_TASK ?? text);
			}
		},

		// One call per model request: the turn number the gate logs decisions under.
		"chat.params": async (input) => {
			turns.set(input.sessionID, (turns.get(input.sessionID) ?? 0) + 1);
		},

		"experimental.chat.messages.transform": async (_input, output) => {
			if (!workarea || mode !== "on" || output.messages.length === 0) return;
			const sessionID = output.messages[0].info.sessionID;
			const known = compacted.get(sessionID) ?? {};
			const items: Record<string, unknown>[] = [];
			let recent = "";
			for (const message of output.messages) {
				const texts: string[] = [];
				for (const part of message.parts as any[]) {
					if (part.type === "tool" && part.state?.status === "completed" && typeof part.state.output === "string") {
						const current = known[part.id] ?? part.state.output;
						items.push({
							id: part.id,
							kind: "tool",
							tokens: estimate(current),
							text: part.id in known ? undefined : part.state.output,
							tool: part.tool,
							call_id: part.callID,
						});
					} else {
						const text = typeof part.text === "string" ? part.text : "";
						items.push({ id: part.id, kind: "other", tokens: estimate(text) });
						if (message.info.role === "assistant" && text) texts.push(text);
					}
				}
				if (texts.length) recent = texts.join("\n");
			}
			try {
				const result = await call("/workarea", {
					session: sessionName(sessionID),
					turn: turns.get(sessionID) ?? 0,
					recent,
					task: tasks.get(sessionID),
					items,
				});
				Object.assign(known, result.replacements);
				compacted.set(sessionID, known);
			} catch (error) {
				console.error(`[jev] workarea failed, reapplying earlier compactions only: ${(error as Error).message}`);
			}
			for (const message of output.messages) {
				for (const part of message.parts as any[]) {
					if (part.type === "tool" && part.id in known && part.state?.status === "completed") {
						part.state.output = known[part.id];
					}
				}
			}
		},

		"experimental.chat.system.transform": async (_input, output) => {
			if (mode === "on") output.system.push(GUIDANCE);
		},

		"tool.execute.after": async (input, output) => {
			const task = tasks.get(input.sessionID) ?? process.env.JEV_TASK;
			const observed = { session: sessionName(input.sessionID), tool: input.tool, call_id: input.callID,
				args: input.args ?? {}, turn: turns.get(input.sessionID) ?? 0, cwd: directory };
			// A call the gate does not score is still observed: an edit dates every earlier
			// read of that file.
			const observe = () => call("/observe", observed).catch((error) =>
				console.error(`[jev] observe failed: ${(error as Error).message}`));
			if (!gatedTools.has(input.tool) || !task || typeof output.output !== "string") return void (await observe());
			const [head, body, tail] = split(output.output);
			const intent = intents.get(input.callID);
			intents.delete(input.callID);
			if (!body.trim()) return void (await observe());
			try {
				const result = await call("/admit", {
					...observed,
					text: body,
					task,
					mode,
					max_elide_fraction: maxElideFraction,
					profile,
					intent,
				});
				if (mode === "on" && result.text !== body) output.output = head + result.text + tail;
			} catch (error) {
				// Fail open, but say so: a silently disabled gate looks like a gate that never fires.
				console.error(`[jev] admit failed, passing the original through: ${(error as Error).message}`);
			}
		},
	};
};
