# ============================================================================
# ONE-CELL RESTART — pull latest, restart server + tunnel, verify end to end.
# Safe to re-run any number of times. Paste into a new Colab cell and run.
# ============================================================================
import os, re, stat, subprocess, sys, time, urllib.request

APP, SLOG, TLOG = "/content/app", "/content/server.log", "/content/tunnel.log"

# --- 0. latest code -------------------------------------------------------
os.chdir(APP)
subprocess.run(["git", "fetch", "--all", "-q"])
subprocess.run(["git", "reset", "--hard", "origin/main", "-q"])
print("code:", subprocess.run(["git", "log", "--oneline", "-1"],
                              capture_output=True, text=True).stdout.strip())

# --- 1. stop anything still running --------------------------------------
for name in ("uvicorn", "cloudflared"):
    subprocess.run(["pkill", "-f", name])
time.sleep(3)

# --- 2. server ------------------------------------------------------------
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=open(SLOG, "w"), stderr=subprocess.STDOUT, cwd=APP)

ready = False
for _ in range(160):                      # models load on startup
    time.sleep(3)
    if "Application startup complete" in open(SLOG).read():
        ready = True
        break
    if server.poll() is not None:
        print("SERVER DIED:\n", open(SLOG).read()[-3000:])
        raise SystemExit
if not ready:
    print("server slow/stuck:\n", open(SLOG).read()[-2500:])
    raise SystemExit
print("server: ready")

# --- 3. tunnel ------------------------------------------------------------
BIN = "/usr/local/bin/cloudflared"
if not os.path.exists(BIN):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64", BIN)
    os.chmod(BIN, os.stat(BIN).st_mode | stat.S_IEXEC)

subprocess.Popen([BIN, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                 stdout=open(TLOG, "w"), stderr=subprocess.STDOUT)

url = None
for _ in range(45):
    time.sleep(2)
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open(TLOG).read())
    if m:
        url = m.group(0)
        break
if not url:
    print("TUNNEL FAILED:\n", open(TLOG).read()[-2000:])
    raise SystemExit
print("tunnel:", url)

# --- 4. wait for DNS, then verify for real --------------------------------
import requests
health = None
for attempt in range(24):
    try:
        health = requests.get(f"{url}/health", timeout=60).json()
        break
    except requests.exceptions.RequestException:
        print(f"\r  waiting for DNS... {attempt*5}s", end="", flush=True)
        time.sleep(5)

print()
if not health:
    print("tunnel up but not resolving yet — wait a minute and open the URL manually")
else:
    ui = requests.get(url, timeout=60).status_code
    q = requests.post(f"{url}/query",
                      json={"text": "कॉर्पोरेशन क्या है?", "language": "hi"},
                      timeout=300).json()
    print("=" * 68)
    print("  OPEN THIS LINK:", url)
    print("=" * 68)
    print(f"  UI page      : HTTP {ui}")
    print(f"  Qdrant       : {health.get('points'):,} points")
    print(f"  providers    : {health.get('providers')}   STT: {health.get('stt_enabled')}")
    print(f"  test query   : conf={q.get('confidence')} "
          f"cites={len(q.get('citations', []))} "
          f"{q.get('timing', {}).get('total_ms', 0):.0f}ms")
    print(f"  answer       : {q.get('answer','')[:80]}")
    print("=" * 68)
    print("  Hard-refresh the page after opening (Cmd/Ctrl + Shift + R).")
