# 3-minute demo script (screen + voice, no slides)

Target: under 3:00. Rehearse once with a timer. Record in one take, host unlisted (YouTube / Loom / Drive "anyone with link").

**0:00–0:20 — What it is.** Show the README top. "This triages one support ticket into category, priority, summary and reason. Small on purpose: one job, validated, measured."

**0:20–1:00 — Happy path.** Terminal 1: `uvicorn app.main:app --port 8000`. Terminal 2: send a ticket with `Invoke-RestMethod` (e.g. a duplicate charge). Point out `request_id`, category, priority, `repaired: false`. Open `logs/triage.jsonl`, show the line: prompt_version, tokens, latency_ms, stop_reason, outcome.

**1:00–1:50 — One failure handled.** Pick ONE (rehearse it, it must be real):
- Run `pytest tests/test_triage.py -k "repair" -v` and show the invalid output being repaired by exactly one second call; or
- Temporarily set `ANTHROPIC_API_KEY=bad` and show the 401 is NOT retried and returns a safe 503 with a request_id; or
- Set `MAX_OUTPUT_TOKENS=50` and show the truncated response (`stop_reason: max_tokens`) is reported as a failure, not a result.
Restore your settings afterwards.

**1:50–2:30 — Evaluation.** Show the README results table from your real run, the per-category breakdown, and one miss in "Cases not fully correct". Say it plainly.

**2:30–3:00 — One trade-off.** "Strict parsing: fenced JSON costs a repair call instead of being silently accepted. I chose clarity of 'valid' over saving a call." Done.
