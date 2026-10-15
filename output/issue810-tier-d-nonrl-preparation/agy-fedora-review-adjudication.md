# Fedora AGY source review evidence

The Tailscale Fedora plugin executed AGY CLI 1.3.5 with the configured model `claude-opus-5-5-high` and effort `high`. The CLI returned nonempty reviews for two bounded source packets. These receipts establish configured harness/model, conversation and response provenance, not provider weights or a complete repository approval.

- Function-only conversation: f3153dec-0b2b-404f-a4df-c9e3a47e60ac; disposition BLOCKED; retained in agy-fedora-cleanup-focused-review.json.
- Caller-only follow-up: e3cda30d-ac42-44d7-af70-23142976b493; disposition PASS limited to supplied caller/tracking excerpts; retained in agy-fedora-caller-adjudication.json.
- The full supervisor packet and earlier Gemini packets returned empty responses after print timeouts. Empty SUCCESS responses are incomplete and unusable as approval.
- No new whole-supervisor approval, Generic Review Gate receipt, research authorization, economic evidence, or live permission is issued by these partial reviews.

Root source reconciliation: terminate_owned_tree catches a failed refresh and uses last_sample when available, then kills creation-bound known descendants before reaping Popen. The caller also has a Popen fallback. The caller-only review's suggestion that this tested path depends solely on an assumed CPython launcher Job Object is not supported by the supplied helper implementation and is not adopted as evidence. Generic helper reuse (initial snapshot unavailable, kill subprocess failure, future respawning children) and failure diagnostic retention remain hardening topics.

The current consumed-development attempt market-20261014-process-tree-r2 was independently approved earlier with gpt-6-astra, explicitly accepted by the user for this retry. It was stopped for throughput at 1905.938 seconds, with 12 fitted vintages, a frozen 48-arm matrix, one INTERRUPTED arm and 47 UNATTEMPTED arms. No completed receipt/result exists. Output manifest matches all 12 retained files; all eight runner source pins remain unchanged; the four known attempt PIDs are absent. The same consumed-development data remains reusable. No profitability conclusion follows.

