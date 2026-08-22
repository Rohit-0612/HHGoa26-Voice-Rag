# ============================================================================
# ONE-CELL RESTART — pull, restart server, open a STABLE ngrok tunnel, verify.
#
# Prereqs (one time, see notebooks/NGROK_SETUP.md):
#   Colab Secrets:  NGROK_TOKEN   = your ngrok authtoken
#                   NGROK_DOMAIN  = your-name.ngrok-free.app   (static domain)
#
# With a static domain the URL NEVER changes, so a restart does not invalidate
# the link you gave the judges. Falls back to cloudflared (random URL) if the
# ngrok secrets are absent.
#
# Safe to re-run any number of times.
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
for name in ("uvicorn", "cloudflared", "ngrok"):
    subprocess.run(["pkill", "-f", name])
time.sleep(3)

# --- 2. server ------------------------------------------------------------
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=open(SLOG, "w"), stderr=subprocess.STDOUT, cwd=APP)

for _ in range(160):
    time.sleep(3)
    if "Application startup complete" in open(SLOG).read():
        break
    if server.poll() is not None:
        print("SERVER DIED:\n", open(SLOG).read()[-3000:]); raise SystemExit
else:
    print("server stuck:\n", open(SLOG).read()[-2500:]); raise SystemExit
print("server: ready")

# --- 3. tunnel: ngrok static domain, else cloudflared ---------------------
def _secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

token, domain = _secret("NGROK_TOKEN"), _secret("NGROK_DOMAIN")
url = None

if token:
    if not os.path.exists("/usr/local/bin/ngrok"):
        subprocess.run("curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz "
                       "| tar xz -C /usr/local/bin", shell=True)
    subprocess.run(["/usr/local/bin/ngrok", "config", "add-authtoken", token],
                   capture_output=True)
    cmd = ["/usr/local/bin/ngrok", "http", "8000", "--log", "stdout"]
    if domain:
        cmd += [f"--domain={domain.replace('https://','').strip('/')}"]
    subprocess.Popen(cmd, stdout=open(TLOG, "w"), stderr=subprocess.STDOUT)

    for _ in range(30):
        time.sleep(2)
        m = re.search(r"url=(https://[-a-z0-9.]+\.ngrok[-a-z.]*\.app)", open(TLOG).read())
        if m:
            url = m.group(1); break
    if not url:
        print("ngrok failed, falling back to cloudflared:\n", open(TLOG).read()[-1200:])

if not url:
    BIN = "/usr/local/bin/cloudflared"
    if not os.path.exists(BIN):
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64", BIN)
        os.chmod(BIN, os.stat(BIN).st_mode | stat.S_IEXEC)
    subprocess.Popen([BIN, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                     stdout=open(TLOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(45):
        time.sleep(2)
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open(TLOG).read())
        if m:
            url = m.group(0); break

if not url:
    print("TUNNEL FAILED:\n", open(TLOG).read()[-2000:]); raise SystemExit
print("tunnel:", url)

# --- 4. verify for real before printing the link --------------------------
import requests
health = None
for attempt in range(24):
    try:
        health = requests.get(f"{url}/health", timeout=60,
                              headers={"ngrok-skip-browser-warning": "1"}).json()
        break
    except requests.exceptions.RequestException:
        print(f"\r  waiting for DNS... {attempt*5}s", end="", flush=True); time.sleep(5)
print()

if not health:
    print("tunnel up but not resolving yet — wait a minute, then open:", url)
else:
    q = requests.post(f"{url}/query",
                      json={"text": "कॉर्पोरेशन क्या है?", "language": "hi"},
                      timeout=300, headers={"ngrok-skip-browser-warning": "1"}).json()
    print("=" * 70)
    print("  OPEN THIS LINK:", url)
    print("=" * 70)
    print(f"  Qdrant     : {health.get('points'):,} points")
    print(f"  providers  : {health.get('providers')}   STT: {health.get('stt_enabled')}")
    print(f"  test query : conf={q.get('confidence')} cites={len(q.get('citations', []))} "
          f"{q.get('timing', {}).get('total_ms', 0):.0f}ms")
    print(f"  answer     : {q.get('answer','')[:80]}")
    if domain:
        print("\n  This is a STATIC domain — the URL stays the same across restarts.")
    print("=" * 70)
