# Troubleshooting

Real issues encountered building and running this project, with fixes.

### "NumPy requires GCC >= 8.4" / pip tries to compile from source
Cause: Python 3.13/3.14 has no prebuilt numpy wheel for the pinned version
chromadb depends on, so pip falls back to compiling from source — which
fails on an old/missing compiler.
**Fix: use Python 3.11 or 3.12.** Create the venv with `py -3.11 -m venv venv`
specifically.

### `RuntimeError: Set GROQ_API_KEY` even though .env exists
1. Confirm `.env` (not just `.env.example`) actually exists:
   `Get-Content .env` (PowerShell) or `cat .env`.
2. **Windows BOM issue**: creating `.env` via PowerShell's
   `Out-File -Encoding utf8` silently adds a byte-order-mark that breaks
   python-dotenv's parsing, even though the file *looks* correct. Fix:
   `[System.IO.File]::WriteAllText(".env", "GROQ_API_KEY=your_key")` instead.

### `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
A newer `httpx` version removed a parameter the `groq` SDK still expects.
Pinned in `requirements.txt` (`httpx==0.27.2`) — if you see this, run
`pip install "httpx==0.27.2"` again.

### `ModuleNotFoundError: No module named 'chromadb'` after a successful install
The venv isn't activated in your current terminal session. Run
`venv\Scripts\activate` again and confirm your prompt shows `(venv)`.

### Latency numbers vary wildly between runs (e.g. 400ms then 9,000ms)
Groq's free-tier rate limiting, not the pipeline. A burst of fast requests
is allowed, then throttling kicks in. Report the unthrottled floor
separately from throttled averages if measuring this yourself.

### `Failed to send telemetry event ClientStartEvent: capture() takes...`
Harmless — ChromaDB's anonymous telemetry failing due to a `posthog`
version mismatch. Doesn't affect functionality.
