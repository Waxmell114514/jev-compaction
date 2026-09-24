"""recall(): find earlier tool output from a loose description.

``expand`` serves an agent that knows exactly what it wants: it holds a pointer
id. Late in a long task the agent more often half-remembers -- "the traceback
from when the date tests first failed", "where the parser handled time zones" --
and the output it means may have been elided, or compacted out of context by the
harness altogether. ``recall`` serves that case, in three steps:

1. **Filter by structure.** Profiled records (:mod:`jevctx.profile`) carry
   type and role distributions, the names they mention and where they came
   from. Filters are soft -- ``role="evidence"`` keeps a record in which some
   segment is evidence with probability at least ``min_prob`` (its *presence*,
   not its token-weighted share, so a long log with a short traceback still
   counts) -- and exact on names, so a vague query can still be pinned to
   ``tests/test_dates.py``.
2. **Shortlist lexically.** BM25 over the query, plus a bonus for each name the
   query mentions, plus recency, keeps at most ``shortlist`` candidates. Keep it
   wide: a vague query shares few words with its target. Replayed with ~200
   stored outputs, widening it from 30 to 90 took top-3 hits on identifier-free
   queries from 53% to 69%, for about 70k Jev input tokens a recall.
3. **Rerank with Jev.** Each candidate becomes a card -- its labels, source,
   turn, names, and the lines around the query's terms -- and Jev scores all of
   them against the query in one batched request. Cards, not whole records: the
   rerank needs to know what a record is, not every line of it.

What comes back is original text: whole records while they fit the budget,
otherwise the excerpt around the matching lines with the record id, so the
agent can ``expand`` the rest. Every rerank decision is logged as a
``retrieve`` decision, and returned records count as hits for calibration.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jevctx.scorer import score_items
from jevctx.shadow import ShadowLog
from jevctx.tokens import estimate_tokens
from jevctx.types import JevClient, MemoryStore, Noul, Origin, Record, ScoreItem

__all__ = ["RECALL_QUESTION", "RecallHit", "card", "excerpt", "recall", "render_hits"]

RECALL_QUESTION = Noul(
    instructions=(
        "`task` holds what the agent is looking for, and the work it is doing. Is this stored "
        "item the thing it is looking for, or does it contain it? Answer true only if the item "
        "would answer the request; related but different output is false."
    ),
    true="The item is, or contains, what the agent is looking for.",
    false="The item is not what the agent is looking for.",
)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PART = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


@dataclass(frozen=True)
class RecallHit:
    record: Record
    score: float
    text: str
    truncated: bool
    #: The ``name`` filter matched nothing and was dropped to find this.
    relaxed: bool = False
    #: Why this output is out of date, if a later action made it so (:mod:`jevctx.supersede`).
    outdated: str = ""


def _terms(query: str) -> set[str]:
    """Words, and the parts of identifiers: ``parse_month`` and ``parseMonth`` both
    also yield ``parse`` and ``month``, so a plain-language query meets code."""
    terms: set[str] = set()
    for word in _WORD.findall(query):
        terms.add(word.lower())
        terms.update(p.lower() for p in _PART.findall(word) if len(p) >= 3)
    return terms


def excerpt(text: str, query: str, budget_tokens: int, context: int = 6) -> tuple[str, bool]:
    """The lines around the query's terms, within ``budget_tokens``.

    Whole text if it fits; otherwise windows of ``context`` lines around the
    lines that mention the most query terms, best first, in original order, with
    ``...`` between gaps. Deterministic, verbatim, never rewritten.
    """
    if estimate_tokens(text) <= budget_tokens:
        return text, False
    lines = text.splitlines()
    terms = _terms(query)
    ranked = sorted(range(len(lines)),
                    key=lambda i: (-len(terms & _terms(lines[i])), i))
    chosen: set[int] = set()
    for index in ranked:
        if not terms & _terms(lines[index]) and chosen:
            break
        window = set(range(max(0, index - context), min(len(lines), index + context + 1)))
        trial = sorted(chosen | window)
        if estimate_tokens("\n".join(lines[i] for i in trial)) > budget_tokens:
            if not chosen:  # even one window is too big: take the head of it
                chosen = set(sorted(window)[:max(1, budget_tokens // 20)])
            break
        chosen |= window
    out: list[str] = []
    previous = -2
    for i in sorted(chosen):
        if i != previous + 1:
            out.append(f"... [line {i + 1}]")
        out.append(lines[i])
        previous = i
    return "\n".join(out), True


def card(record: Record, query: str, budget_tokens: int = 250, outdated: str = "") -> str:
    """What the reranker sees of one record: labels, provenance, names, best lines,
    and whether a later action made it out of date."""
    profile = record.meta.get("profile") or {}
    names = ", ".join(record.meta.get("names", [])[:12])
    header = (f"[{profile.get('role', '?')}, {profile.get('type', '?')}] "
              f"{record.origin.source} turn {record.created_turn}"
              + (f"; mentions {names}" if names else "")
              + (f"; OUT OF DATE: {outdated}" if outdated else ""))
    body, _ = excerpt(record.text, query, budget_tokens, context=2)
    return f"{header}\n{body}"


def _matches(record: Record, *, type_: str | None, role: str | None, name: str | None,
             source: str | None, min_prob: float) -> bool:
    profile = record.meta.get("profile") or {}
    types = profile.get("type_presence") or profile.get("type_probs", {})
    roles = profile.get("role_presence") or profile.get("role_probs", {})
    if type_ is not None and types.get(type_, 0.0) < min_prob:
        return False
    if role is not None and roles.get(role, 0.0) < min_prob:
        return False
    if source is not None and record.origin.source != source:
        return False
    if name is not None:
        lowered = name.lower()
        if not any(lowered in n.lower() for n in record.meta.get("names", [])) \
                and lowered not in record.text.lower():
            return False
    return True


def recall(
    query: str,
    *,
    store: MemoryStore,
    client: JevClient,
    log: ShadowLog,
    turn: int,
    task: str = "",
    k: int = 3,
    budget_tokens: int = 4000,
    type_: str | None = None,
    role: str | None = None,
    name: str | None = None,
    source: str | None = None,
    min_prob: float = 0.3,
    threshold: float = 0.4,
    shortlist: int = 90,
    outdated: Callable[[Record], str] | None = None,
) -> list[RecallHit]:
    """Up to ``k`` records matching ``query``, best first, within ``budget_tokens``.

    ``outdated`` says why a record is out of date ('' if it is not); the reranker
    sees it on the card, so a current copy of the same thing can win, and the agent
    sees it on the hit."""
    if not query.strip():
        raise ValueError("query must be nonempty")
    if not 1 <= k <= 10 or not 200 <= budget_tokens <= 16_000:
        raise ValueError("k must be 1-10 and budget_tokens 200-16000")
    records = store.all_records()

    def fragment_of_stored_output(record: Record) -> bool:
        # The whole output is the better answer; an unprofiled gate stores none,
        # and then the fragment is all there is.
        parent = record.meta.get("full_output_id")
        return bool(parent) and store.get(parent) is not None

    candidates = [r for r in records
                  if not fragment_of_stored_output(r)
                  and _matches(r, type_=type_, role=role, name=name, source=source,
                               min_prob=min_prob)]
    relaxed = False
    if not candidates and name is not None:
        # A name the agent misremembers (in a live run: a harness-internal call id)
        # should narrow the search, not end it. Drop it and say so.
        candidates = [r for r in records
                      if not fragment_of_stored_output(r)
                      and _matches(r, type_=type_, role=role, name=None, source=source,
                                   min_prob=min_prob)]
        relaxed, name = True, None
    if not candidates:
        return []

    ranked = store.search(query, limit=len(records))
    lexical = {r.id: len(ranked) - i for i, r in enumerate(ranked)}
    terms = _terms(query)
    newest = max(r.created_turn for r in candidates) or 1

    def prior(r: Record) -> float:
        named = sum(1 for n in r.meta.get("names", []) if n.lower() in query.lower()
                    or terms & _terms(n))
        return lexical.get(r.id, 0) + 5 * named + r.created_turn / newest

    shortlisted = sorted(candidates, key=prior, reverse=True)[:shortlist]
    # A name filter is part of what the agent is looking for.
    focus = f"{query} {name}" if name else query
    stale = {r.id: outdated(r) for r in shortlisted} if outdated else {}
    cards = {r.id: card(r, focus, outdated=stale.get(r.id, "")) for r in shortlisted}
    items = [ScoreItem(id=i, text=text, tokens=estimate_tokens(text)) for i, text in cards.items()]
    digest = f"Looking for: {query}" + (f"\n\nCurrent task: {task[:2000]}" if task else "")
    results = score_items(client, digest, items, RECALL_QUESTION)
    by_id = {r.id: r for r in shortlisted}

    hits: list[RecallHit] = []
    spent = 0
    for result in sorted(results, key=lambda r: -r.score):
        if result.failed or result.score < threshold or len(hits) == k:
            continue
        record = by_id[result.item_id]
        remaining = budget_tokens - spent
        if remaining < 100:
            break
        text, truncated = excerpt(record.text, focus, min(remaining, budget_tokens // k * 2))
        spent += estimate_tokens(text)
        hits.append(RecallHit(record=record, score=result.score, text=text, truncated=truncated,
                              relaxed=relaxed, outdated=stale.get(record.id, "")))
        store.touch(record.id, hit=True)

    chosen = {h.record.id for h in hits}
    for result in results:
        log.decision(kind="retrieve", item_id=result.item_id, score=result.score,
                     threshold=threshold,
                     action="injected" if result.item_id in chosen else "skipped",
                     tokens=by_id[result.item_id].tokens,
                     origin=Origin(source="recall", ref=None, turn=turn),
                     text=query, turn=turn)
    return hits


def render_hits(hits: Sequence[RecallHit]) -> str:
    """The text the agent sees. Headers are metadata; everything else is verbatim."""
    if not hits:
        return "No stored output matched. Try a broader query or fewer filters."
    parts = ["(No stored output mentions that name; showing the best matches without it.)"
             ] if hits[0].relaxed else []
    for hit in hits:
        profile = hit.record.meta.get("profile") or {}
        note = (f"; excerpt of {hit.record.tokens} tokens, expand id={hit.record.id} for all"
                if hit.truncated else "")
        warning = f"; out of date: {hit.outdated}" if hit.outdated else ""
        parts.append(f"[[record id={hit.record.id} from {hit.record.origin.source} "
                     f"turn {hit.record.created_turn}; {profile.get('role', '?')}, "
                     f"{profile.get('type', '?')}; match {hit.score:.2f}{warning}{note}]]\n{hit.text}")
    return "\n\n".join(parts)
