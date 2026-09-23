# Memory core specification

This branch contains the harness-independent implementation. Pi integrations,
Pi bridge and their integration tests are deliberately excluded.

## Retrieval

Merge up to 50 lexical matches with up to 50 recent digest entries. Resolve
fragment links to full output records, filter by content type, lifecycle or
entity presence (probability >=0.5), and rerank summaries with Jev. Score budgets
use summary size, not original text size. Return whole records, deduplicated by
text; skip oversized records and continue. The default output budget is 4000
estimated tokens including JSON metadata, with a limit of five records. Empty
candidate sets make no API calls. Shadow retrieval logs no injected records;
scoring failures propagate. No automatic deletion or expiry is introduced.

## Multi-dimensional labels

Existing batched labelling supplies content type, lifetime and three independent
presence probabilities for paths, URLs and identifiers. These are not extracted
entity strings. Successful labels persist via label_records(store=...); failed
labels preserve the existing record. The core API does not automatically label
all outputs: integrations choose when to call it.

## Prefix bookkeeping

advance_prefix reuses ContextBuffer to commit a sequence of message hashes.
Appending or retrying identical context preserves the epoch. Changed/shortened
prefixes and explicit reset reasons begin a new epoch and report the common
prefix length. The caller owns hash creation and state persistence. No original
messages are restored or rewritten, and stable hashes do not guarantee provider
cache hits.

## Model runner and costs

The OpenAI-compatible runner supplies off/shadow/on admission modes, exact
expansion, bounded execution and provider usage accounting. It exposes only
explicitly selected files in the CLI. Host and Jev prices are separate; absent
prices/usage remain unknown. CacheLedger is a simulator, not measured billing.

## Verification

The Python suite covers candidate merging, filtering, summary/result budgets,
deduplication, prefix changes, labelling failures, wire protocol and expansion.
No live model efficiency claim is made. OpenCode model probes returned HTTP 402
(insufficient account funds); official free models returned HTTP 403 (available
only within OpenCode). Jev connectivity succeeded. No benchmark comparison ran.
