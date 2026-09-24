# Jev for OpenCode

[`jev.ts`](jev.ts) is an [OpenCode](https://opencode.ai) plugin that puts the
`jevctx` gate between OpenCode's tools and its context. The gate itself is Python.
The plugin talks to a small sidecar, `python -m jevctx.serve`, which owns the
stores, logs and Jev clients.

What it does:

- **Admission.** In `tool.execute.after`, every text result from `bash`, `read`,
  `grep`, `glob` and `list` goes through `admit()` before OpenCode stores it.
  Low-value runs of lines become `[[elided id=... "role, type: summary"]]`
  pointers, and the originals stay in the sidecar. With `JEV_PROFILE=1` the same
  Jev request also labels each segment:
  - its type: source code, test output, traceback…
  - its role: the code to change, evidence of the bug, navigation, noise…
  - how long it stays relevant
  - whether it is a prompt injection, which is quarantined, not passed through
- **`expand`** returns the original text behind a pointer, byte for byte.
- **`recall`** finds earlier output from a description, when the model only roughly
  remembers it or it has left the context. It takes optional filters: a name the
  output mentions, a `type` and a `role`. It returns original text.
- **Supersession.** Every tool call's arguments go to the sidecar too; calls it
  does not gate (`edit`, `write`…) go to `/observe`. The sidecar then knows which
  outputs a later call made obsolete: the same command run again, the same lines
  read again, or the file edited since. `expand` and `recall` flag them.
- **Work area** (`JEV_WORKAREA=1`). Before each model request, the tail of the
  transcript since the last commit point is offered for compaction. Outputs Jev
  says have served their purpose, and outputs a later call made obsolete, become
  pointers. That happens only when the price arithmetic says breaking the
  provider's prompt cache pays for itself. The rewrite is request-local (OpenCode's
  stored session is untouched) and is reapplied identically on every request.

OpenCode wraps a `read` result in `<path>`, `<type>` and `<content>` tags and
ends a truncated one with `(Showing lines 1-2000 of 3000. Use offset=2001 to
continue.)`. The plugin gates only the body between them, so the model always
sees where it is and how to read on.

## Run it

```bash
# 1. the sidecar (TYPESAFE_API_KEY is the Jev key)
export TYPESAFE_API_KEY=...
.venv/bin/python -m jevctx.serve --port 8765 --data-dir runs/opencode \
    --profile --gate-on role:change_site --max-elide-fraction 1.0 \
    --price-input 3 --price-cache-read 0.3        # your model's prices, USD per 1M tokens

# 2. install the plugin
mkdir -p ~/.config/opencode/plugins
cp integrations/opencode/jev.ts ~/.config/opencode/plugins/
#    it imports @opencode-ai/plugin: OpenCode installs that for its config
#    directory, or link a node_modules that has it next to plugins/

# 3. run
JEV_URL=http://127.0.0.1:8765 JEV_PROFILE=1 JEV_WORKAREA=1 opencode
```

| variable | meaning |
|---|---|
| `JEV_URL` | sidecar URL (required; the plugin does not start one) |
| `JEV_MODE` | `on` (default); `shadow` scores and logs but changes nothing; `off` |
| `JEV_SESSION` | store name on the sidecar (default `oc-<OpenCode session id>`) |
| `JEV_TOOLS` | tools to gate (default `bash,read,grep,glob,list`) |
| `JEV_TASK` | task digest (default: the session's first user message) |
| `JEV_PROFILE` | `1` asks the type, role, lifetime and injection questions with every admit |
| `JEV_MAX_ELIDE_FRACTION` | override the sidecar's tripwire; `1` disables it |
| `JEV_WORKAREA` | `1` turns on work-area compaction |

The plugin fails open. If the sidecar or Jev is unreachable, the original output
goes through unchanged and a line is written to stderr.

For the numbers on SWE-bench Verified, see [RESULTS.md](../../RESULTS.md).
