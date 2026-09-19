# awesome-jev-compaction

Cache-preserving agent context compaction and memory, with
[TypeSafe Jev](https://docs.typesafe.ai) as the semantic judgement layer.

> An agent's context is `[frozen prefix] + [work area]`. The frozen prefix is append-only, so the
> KV cache over it is never invalidated. Raw tool output is **relocated**, never deleted: what the
> gate removes goes to an external store and leaves a one-line pointer the agent can `expand()`.
> Jev decides what to relocate at write time and what to pull back at read time — the same scoring
> primitive at both ends.

See **[SPEC.md](SPEC.md)** for the full design and the hard Jev constraints it is derived from.

Status: Phase 0 (staging area) and Phase 1 (scorer + store) under construction.
