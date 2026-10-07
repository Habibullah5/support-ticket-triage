# AI Support Ticket Triage Agent

Takes one customer support ticket and returns a **category** (`billing`, `technical`, `account`, `delivery`, `refund`, `other`), a **priority** (`low`, `medium`, `high`), a one-sentence **summary**, and a concise **reason** using Claude Haiku (`claude-haiku-4-5-20251001`) behind a small FastAPI service. Model output is strictly validated against a Pydantic schema, repaired at most once on validation errors, retried with jittered exponential backoff on transient network failures, and recorded with full token, latency, and prompt telemetry for every call. It is deliberately bounded: one job, executed cleanly end-to-end.

---

## Demo Video

[![Watch the 3-Minute Capstone Demo Walkthrough](https://img.youtube.com/vi/F57qfdPUey8/0.jpg)](https://youtu.be/F57qfdPUey8)

**Direct Link:** [https://youtu.be/F57qfdPUey8](https://youtu.be/F57qfdPUey8)  
*(Duration: < 3 minutes. Screen recording with microphone voiceover showing live CLI triage, 1-attempt repair fallback handling, and the complete evaluation suite run.)*

---

## 1. Quickstart (Windows PowerShell, Under 4 Minutes)

Requires Python 3.11+ and an Anthropic API key.

### Clone and Environment Setup

```powershell
git clone [https://github.com/Habibullah5/ai-support-ticket-triage.git](https://github.com/Habibullah5/ai-support-ticket-triage.git) support-triage-agent
cd support-triage-agent

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell script execution is restricted, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
Copy-Item .env.example .env
notepad .env