"""Supersession: which earlier tool outputs a later action made obsolete.

The profile (:mod:`jevctx.profile`) says what each output *is*. This module says
how outputs *relate*, from what the agent did rather than from what the text says,
so it needs no Jev call and cannot be wrong about the facts it records:

- **superseded** -- a later output shows the same thing: the same command or
  search run again, or the same file viewed again over at least the same lines.
  The earlier one adds nothing the later one does not.
- **stale** -- a later write touched the file an earlier output showed. The earlier
  text may no longer match the file.

Relations are keyed by tool-call id, which every harness hands the gate with the
output. An output's *footprint* comes from its tool and arguments:

=========  ===============================================================
view       ``read``; or bash ``cat``, ``nl``, ``head -n N``, ``sed -n 'A,Bp'`` of one file
write      ``edit``, ``write``, ``patch``...; or bash ``sed -i``, ``>``, ``tee``, ``cp``,
           ``mv``, ``git checkout``/``restore`` (``git stash``/``reset``/``apply``
           and ``patch`` touch every file)
run        any bash command, by its normalised text
search     ``grep``, ``glob``, ``list``..., by their arguments
=========  ===============================================================

What uses them: the work area compacts superseded and stale outputs without asking
Jev whether they are still needed (:mod:`jevctx.workarea`); ``recall`` shows the
reranker and the agent which hits are out of date; ``expand`` says so on top of the
text it returns.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Footprint", "Relation", "SupersessionIndex", "footprints", "note_for"]

RelationKind = Literal["superseded", "stale"]
#: Every file: the write could have changed anything (``git stash``, ``patch``).
ANY_FILE = "*"
_WHOLE = float("inf")

VIEW_TOOLS = frozenset({"read", "view", "cat"})
WRITE_TOOLS = frozenset({"edit", "write", "multiedit", "patch", "apply_patch", "str_replace"})
SEARCH_TOOLS = frozenset({"grep", "glob", "list", "ls", "find", "search"})
RUN_TOOLS = frozenset({"bash", "shell", "powershell"})


@dataclass(frozen=True)
class Footprint:
    kind: Literal["view", "write", "run", "search"]
    #: The file (view, write), the normalised command (run) or the call (search).
    key: str
    #: Lines viewed, 1-based and inclusive; ``(1, inf)`` is the whole file.
    span: tuple[float, float] | None = None


@dataclass(frozen=True)
class Relation:
    older: str
    newer: str
    kind: RelationKind
    turn: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# -- footprints ------------------------------------------------------------------ #

def _path(value: Any, cwd: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = value.strip()
    if not path.startswith("/") and cwd:
        path = posixpath.join(cwd, path)
    return posixpath.normpath(path)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_span(args: Mapping[str, Any]) -> tuple[float, float]:
    # OpenCode and Pi: 1-based ``offset``, ``limit`` lines (both default to the head;
    # a read without a limit stops at the harness's cap, so it covers "the start").
    start = max(1, _int(args.get("offset")) or 1)
    limit = _int(args.get("limit"))
    return (float(start), start + limit - 1.0 if limit else _WHOLE)


_CD_PREFIX = re.compile(r"^\s*cd\s+(\S+)\s*(?:&&|;)\s*")
_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+))?p$")


_STDERR = re.compile(r"\s(?:2>&1|2>\s*/dev/null)(?=\s|$)")
_FILTER = re.compile(r"\s*\|\s*(?:head|tail|grep|sed|sort|uniq|wc|cat|less)\b[^|\n]*$")


def _normalise(command: str) -> str:
    """Whitespace collapsed, stderr merges and trailing output filters (``| tail -20``,
    ``| grep FAIL``) dropped: they change how much of the check is shown, not which
    check it is. A heredoc's body is kept whole."""
    head, sep, body = command.partition("\n")
    if "<<" not in head:
        head, sep, body = command, "", ""
    head = _STDERR.sub(" ", " " + head)
    while _FILTER.search(head):
        head = _FILTER.sub("", head)
    return " ".join((head + sep + body).split())


def _first_line(command: str) -> str:
    # A heredoc's body is data, not shell: parse the line that starts it only.
    head = command.split("\n", 1)[0]
    return head if "<<" in head else command


def _simple_commands(command: str) -> list[list[str]] | None:
    """The command split on ``&&``, ``||``, ``;`` and ``|``, each tokenised; None if
    it does not tokenise (unbalanced quotes, a heredoc across lines...)."""
    try:
        lexer = shlex.shlex(_first_line(command), posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    parts: list[list[str]] = [[]]
    for token in tokens:
        if token in {"&&", "||", ";", "|", "&", ";;"}:
            parts.append([])
        else:
            parts[-1].append(token)
    return [p for p in parts if p]


def _view_of(words: list[str], cwd: str | None) -> Footprint | None:
    """``cat f``, ``nl -ba f``, ``head -n N f``, ``sed -n 'A,Bp' f``: a view of one file."""
    name, rest = words[0], words[1:]
    files = [w for w in rest if not w.startswith("-")]
    if name == "cat" and len(files) == 1 and all(w in {"-n", "-A"} for w in rest if w.startswith("-")):
        path = _path(files[0], cwd)
        return Footprint("view", path, (1.0, _WHOLE)) if path else None
    if name == "nl" and len(files) == 1:
        path = _path(files[0], cwd)
        return Footprint("view", path, (1.0, _WHOLE)) if path else None
    if name == "head":
        count, targets, i = 10, [], 0
        while i < len(rest):
            word = rest[i]
            if word == "-n" and i + 1 < len(rest) and _int(rest[i + 1]):
                count, i = _int(rest[i + 1]) or count, i + 2
                continue
            if re.fullmatch(r"-n?\d+", word):
                count = int(word.lstrip("-n"))
            elif not word.startswith("-"):
                targets.append(word)
            i += 1
        path = _path(targets[0], cwd) if len(targets) == 1 else None
        return Footprint("view", path, (1.0, float(count))) if path else None
    if name == "sed" and rest[:1] == ["-n"] and len(rest) == 3:
        match = _SED_RANGE.match(rest[1])
        path = _path(rest[2], cwd)
        if match and path:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            return Footprint("view", path, (float(start), float(end)))
    return None


def _writes_of(words: list[str], cwd: str | None) -> list[str]:
    """Files one simple command may have changed."""
    name, rest = words[0], words[1:]
    paths: list[str | None] = []
    if name == "sed" and any(w.startswith("-i") or w == "--in-place" for w in rest):
        operands = [w for w in rest if not w.startswith("-")]
        # The first operand is the script unless -e/-f gave it.
        script_given = any(w in {"-e", "-f"} for w in rest)
        paths += [_path(w, cwd) for w in (operands if script_given else operands[1:])]
    elif name == "tee":
        paths += [_path(w, cwd) for w in rest if not w.startswith("-")]
    elif name in {"cp", "mv", "install"}:
        operands = [w for w in rest if not w.startswith("-")]
        if len(operands) >= 2:
            paths.append(_path(operands[-1], cwd))
            if name == "mv":
                paths += [_path(w, cwd) for w in operands[:-1]]
    elif name in {"rm", "touch", "truncate"}:
        paths += [_path(w, cwd) for w in rest if not w.startswith("-")]
    elif name == "git" and rest[:1] in (["checkout"], ["restore"]):
        operands = [w for w in rest[1:] if not w.startswith("-")]
        if "--" in rest:
            operands = rest[rest.index("--") + 1:]
        if operands == ["."] or not operands:
            return [ANY_FILE]
        paths += [_path(w, cwd) for w in operands]
    elif name == "git" and rest[:1] == ["stash"] and rest[1:2] in ([], ["push"], ["pop"],
                                                                     ["apply"], ["-q"], ["save"]):
        return [ANY_FILE]
    elif (name == "git" and rest[:1] in (["reset"], ["apply"], ["am"], ["pull"], ["merge"],
                                          ["rebase"], ["cherry-pick"])) or name == "patch":
        # Dry runs and reports apply nothing.
        if not {"--stat", "--numstat", "--summary", "--check", "--dry-run"} & set(rest):
            return [ANY_FILE]
    # Redirections: ">" and ">>" followed by a target that is not a descriptor.
    for i, word in enumerate(words[:-1]):
        if word in {">", ">>"} and not words[i + 1].startswith("&"):
            paths.append(_path(words[i + 1], cwd))
    return [p for p in paths if p and p != "/dev/null"]


def footprints(tool: str, args: Mapping[str, Any] | None, *, cwd: str | None = None
               ) -> list[Footprint]:
    """What one tool call viewed, wrote, ran or searched."""
    args = args or {}
    tool = (tool or "").lower()
    if tool in VIEW_TOOLS:
        path = _path(args.get("filePath") or args.get("path") or args.get("file_path"), cwd)
        return [Footprint("view", path, _read_span(args))] if path else []
    if tool in WRITE_TOOLS:
        raw = args.get("filePath") or args.get("path") or args.get("file_path")
        path = _path(raw, cwd)
        return [Footprint("write", path)] if path else [Footprint("write", ANY_FILE)]
    if tool in SEARCH_TOOLS:
        return [Footprint("search", tool + json.dumps(dict(args), sort_keys=True, default=str))]
    if tool in RUN_TOOLS:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return []
        workdir = _path(args.get("workdir") or args.get("cwd"), cwd) or cwd
        cd = _CD_PREFIX.match(command)
        body = command[cd.end():] if cd else command
        if cd:
            workdir = _path(cd.group(1), workdir) or workdir
        found = [Footprint("run", f"{workdir or ''}$ {_normalise(body)}")]
        parts = _simple_commands(body)
        if parts is None:
            return found
        if len(parts) == 1:
            view = _view_of(parts[0], workdir)
            if view is not None:
                found.append(view)
        # "git stash && run the tests && git stash pop" checks the old code and puts the
        # change back: no file ends up different.
        stashes = [w for w in parts if w[:2] == ["git", "stash"]]
        restored = bool(stashes) and stashes[-1][2:3] in (["pop"], ["apply"]) \
            and any(w[2:3] in ([], ["push"], ["-q"], ["save"]) for w in stashes)
        for words in parts:
            if restored and words[:2] == ["git", "stash"]:
                continue
            found += [Footprint("write", p) for p in _writes_of(words, workdir)]
        return found
    return []


# -- the index ------------------------------------------------------------------- #

@dataclass
class _Output:
    call_id: str
    turn: int
    prints: list[Footprint]
    #: Line ranges of this output's file that later views have shown again.
    covered: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class SupersessionIndex:
    """Relations among one session's tool calls, built one call at a time.

    ``observe`` every tool call in order -- outputs the gate saw and edits it did
    not -- and it returns the relations that call created.
    """

    cwd: str | None = None
    relations: dict[str, list[Relation]] = field(default_factory=dict)
    _outputs: list[_Output] = field(default_factory=list)

    def observe(self, call_id: str, tool: str, args: Mapping[str, Any] | None, turn: int
                ) -> list[Relation]:
        prints = footprints(tool, args, cwd=self.cwd)
        new: list[Relation] = []
        writes = {p.key for p in prints if p.kind == "write"}
        for earlier in self._outputs:
            if earlier.call_id == call_id or self.status(earlier.call_id, "superseded"):
                continue
            reason = _supersedes(prints, earlier)
            if reason:
                new.append(self._relate(earlier.call_id, call_id, "superseded", turn, reason))
                continue
            if writes and not self.status(earlier.call_id, "stale"):
                for view in (p for p in earlier.prints if p.kind == "view"):
                    if view.key in writes or ANY_FILE in writes:
                        new.append(self._relate(earlier.call_id, call_id, "stale", turn,
                                                f"{view.key} was changed after this output"))
                        break
        if any(p.kind in {"view", "run", "search"} for p in prints):
            self._outputs.append(_Output(call_id, turn, prints))
        return new

    def status(self, call_id: str, kind: RelationKind | None = None) -> Relation | None:
        """The relation that matters most for ``call_id``: superseded before stale."""
        found = self.relations.get(call_id, [])
        for wanted in ((kind,) if kind else ("superseded", "stale")):
            for relation in found:
                if relation.kind == wanted:
                    return relation
        return None

    def load(self, relations: Iterable[Mapping[str, Any]]) -> None:
        """Restore relations saved with :meth:`Relation.to_dict` (outputs are not
        restored: new calls relate only to calls observed since)."""
        for data in relations:
            relation = Relation(**data)
            self.relations.setdefault(relation.older, []).append(relation)

    def _relate(self, older: str, newer: str, kind: RelationKind, turn: int, reason: str
                ) -> Relation:
        relation = Relation(older=older, newer=newer, kind=kind, turn=turn, reason=reason)
        self.relations.setdefault(older, []).append(relation)
        return relation


def _supersedes(later: list[Footprint], earlier: _Output) -> str | None:
    for old in earlier.prints:
        for new in later:
            if old.kind != new.kind or old.key != new.key:
                continue
            if old.kind in {"run", "search"}:
                return "the same command was run again" if old.kind == "run" \
                    else "the same search was run again"
            if old.kind == "view" and old.span and new.span:
                # Several narrower views can between them show all an earlier one did.
                earlier.covered.append(new.span)
                if _covers(earlier.covered, old.span):
                    return f"{old.key} was viewed again over the same lines"
    return None


def _covers(spans: list[tuple[float, float]], target: tuple[float, float]) -> bool:
    reach = target[0]
    for start, end in sorted(spans):
        if start > reach:
            break
        if end >= target[1]:
            return True
        reach = max(reach, end + 1)
    return False


def note_for(relation: Relation | None) -> str:
    """One line to show the agent above an out-of-date output, or ''."""
    if relation is None:
        return ""
    if relation.kind == "superseded":
        return f"[note: superseded at turn {relation.turn}: {relation.reason}; prefer the later output]"
    return f"[note: possibly out of date: {relation.reason} (turn {relation.turn})]"
