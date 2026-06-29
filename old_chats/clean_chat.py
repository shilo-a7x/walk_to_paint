"""
Extract a VS Code Copilot chat export (baselines.json) into a lean, readable
Markdown transcript.

Unlike a naive prose-only dump, this captures the parts that actually document
*what was done* — terminal commands + their output, file edits (which file, and
the new content), subagent calls, todo lists, and read/list tool actions — in
addition to the user/assistant prose and the assistant's thinking. That is where
the provenance of things like baselines/prepare_splits.py and the train/test
split decisions lives.

Usage (from this directory):
    python clean_chat.py
    python clean_chat.py --full            # no truncation of outputs/edits
    python clean_chat.py -i other.json -o other.md
"""

import argparse
import json
import os

# ── truncation budgets (chars). --full disables all of them. ─────────────────
LIMITS = {
    "thinking": 4000,
    "assistant": 100000,   # prose is the point; keep essentially all of it
    "terminal_out": 1600,
    "edit_text": 3000,
    "subagent_prompt": 1000,
    "subagent_result": 2500,
    "tool_msg": 300,
}


def _clip(text, key, full):
    text = "" if text is None else str(text)
    if full:
        return text
    n = LIMITS.get(key)
    if n is None or len(text) <= n:
        return text
    head = text[: int(n * 0.75)]
    tail = text[-int(n * 0.2):]
    return f"{head}\n… [clipped {len(text) - len(head) - len(tail)} chars] …\n{tail}"


def _msg_text(val):
    """invocationMessage / message can be a str or a {'value': ...} dict."""
    if isinstance(val, dict):
        return val.get("text", val.get("value", ""))
    return val or ""


def _emit_response_item(x, out, full):
    """Render one item of a request's `response` list."""
    if not isinstance(x, dict):
        return
    kind = x.get("kind")

    # plain assistant markdown (these have no 'kind')
    if kind is None and isinstance(x.get("value"), str):
        v = x["value"].strip()
        if v:
            out.write(_clip(v, "assistant", full) + "\n\n")
        return

    if kind == "thinking":
        v = _msg_text(x.get("value")) or x.get("value")
        v = (v or "").strip() if isinstance(v, str) else ""
        if v:
            out.write("> 🧠 **thinking:** " + _clip(v, "thinking", full).replace("\n", "\n> ") + "\n\n")
        return

    if kind == "toolInvocationSerialized":
        tsd = x.get("toolSpecificData")
        tkind = tsd.get("kind") if isinstance(tsd, dict) else None

        if tkind == "terminal":
            cmd = tsd.get("commandLine")
            cmd = cmd.get("original", cmd.get("userEdited", "")) if isinstance(cmd, dict) else cmd
            cwd = tsd.get("cwd")
            outp = tsd.get("terminalCommandOutput")
            outp = outp.get("text") if isinstance(outp, dict) else outp
            out.write(f"🖥️  **terminal**{f' (cwd: {cwd})' if cwd else ''}:\n")
            out.write("```bash\n" + (cmd or "").strip() + "\n```\n")
            if outp and str(outp).strip():
                out.write("<details><summary>output</summary>\n\n```\n"
                          + _clip(str(outp).strip(), "terminal_out", full)
                          + "\n```\n</details>\n\n")
            else:
                out.write("\n")
            return

        if tkind == "subagent":
            out.write(f"🤖 **subagent** [{tsd.get('agentName')}] — {tsd.get('description','')}\n")
            p = (tsd.get("prompt") or "").strip()
            r = (tsd.get("result") or "").strip()
            if p:
                out.write("  - prompt: " + _clip(p, "subagent_prompt", full).replace("\n", " ") + "\n")
            if r:
                out.write("  - result: " + _clip(r, "subagent_result", full).replace("\n", " ") + "\n")
            out.write("\n")
            return

        if tkind == "todoList":
            items = tsd.get("todoList") or []
            if isinstance(items, list) and items:
                out.write("📋 **todo list:**\n")
                for it in items:
                    if isinstance(it, dict):
                        out.write(f"  - [{it.get('status','')}] {it.get('title', it.get('description',''))}\n")
                out.write("\n")
            return

        # other tools (read file, list dir, search, …): one-line action note
        msg = _msg_text(x.get("invocationMessage")).strip()
        if msg:
            out.write(f"🔧 {x.get('toolId','tool')}: " + _clip(msg, "tool_msg", full).replace("\n", " ") + "\n\n")
        return

    if kind == "textEditGroup":
        uri = x.get("uri") or {}
        path = uri.get("path") or uri.get("fsPath") or "<unknown file>"
        out.write(f"✏️  **edit**: `{path}`\n")
        # edits: list[ list[ {text, range} ] ] — concatenate the inserted text
        texts = []
        for grp in x.get("edits") or []:
            if isinstance(grp, list):
                for e in grp:
                    if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip():
                        texts.append(e["text"])
        if texts:
            joined = "\n".join(texts)
            out.write("```\n" + _clip(joined, "edit_text", full) + "\n```\n\n")
        else:
            out.write("\n")
        return

    # inlineReference / codeblockUri / undoStop / mcpServersStarting / etc.: skip
    return


def extract_requests(data, out, full):
    out.write(f"# Copilot Chat — {data.get('responderUsername','export')}\n\n")
    out.write(f"_{len(data['requests'])} interactions_\n\n")
    for i, turn in enumerate(data["requests"], 1):
        out.write(f"\n## Interaction {i}\n\n")
        user_msg = _msg_text(turn.get("message", "")).strip()
        if user_msg:
            out.write("**👤 User:**\n\n" + user_msg + "\n\n")
        out.write("**🤖 Copilot:**\n\n")
        resp = turn.get("response", [])
        if isinstance(resp, list):
            for x in resp:
                _emit_response_item(x, out, full)
        elif isinstance(resp, str) and resp.strip():
            out.write(resp.strip() + "\n\n")
        out.write("\n---\n")


# ── legacy fallbacks for other export schemas ────────────────────────────────

def _recursive_dump(node, out):
    if isinstance(node, dict):
        for k in ["prompt", "user", "message", "text"]:
            if k in node and isinstance(node[k], str) and node[k].strip():
                out.write(f"**Text Content ({k}):**\n{node[k].strip()}\n\n")
        for k in ["response", "assistant", "value", "content"]:
            if k in node and isinstance(node[k], str) and node[k].strip():
                out.write(f"**AI Content ({k}):**\n{node[k].strip()}\n\n")
        for v in node.values():
            if isinstance(v, (dict, list)):
                _recursive_dump(v, out)
    elif isinstance(node, list):
        for item in node:
            _recursive_dump(item, out)


def extract_chat(input_path, output_path, full=False):
    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}'")
        return
    print("Reading and parsing JSON...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out:
        if isinstance(data, dict) and "requests" in data:
            extract_requests(data, out, full)
        else:
            out.write("# Copilot Chat History Export\n\n")
            print("Unrecognized schema; using structural fallback parser...")
            items = data if isinstance(data, list) else data.get("history", [])
            for i, item in enumerate(items, 1):
                out.write(f"## Block {i}\n\n")
                _recursive_dump(item, out)
                out.write("\n---\n\n")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Finished! Output saved to {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="baselines.json")
    ap.add_argument("-o", "--output", default="baselines_cleaned.md")
    ap.add_argument("--full", action="store_true", help="disable truncation of outputs/edits")
    args = ap.parse_args()
    extract_chat(args.input, args.output, full=args.full)
