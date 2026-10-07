# Prompts

Prompts are code: they live in files, are versioned, and every change is explained in `CHANGELOG.md`.

| File | Used for |
|------|----------|
| `triage_v1.txt` | System prompt for the first (classification) call |
| `repair_v1.txt` | System prompt for the single repair call after invalid output |

## How they are loaded
`app/triage.py` reads `prompts/<name>.txt`, where `<name>` comes from the `TRIAGE_PROMPT` / `REPAIR_PROMPT` settings (default `triage_v1` / `repair_v1`).
The file stem is logged as `prompt_version` on every model call, together with `prompt_sha256` (first 12 hex chars of the file hash) so a log line can be matched to the exact text that produced it.

## How to change a prompt
1. Copy the file to a new version (`triage_v1.txt` -> `triage_v2.txt`). Do not edit an old version in place.
2. Add an entry to `CHANGELOG.md`: what changed and why (ideally with the eval numbers before and after).
3. Set `TRIAGE_PROMPT=triage_v2` in `.env`, run `python evals/run_eval.py --update-readme`, and compare with the previous results.
4. Keep the old file so past log lines remain reconstructable.
