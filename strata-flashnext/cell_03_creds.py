# CELL 3 — credentials, pasted inline (as you asked — not Kaggle secrets).
# Both are written to /tmp ONLY: they die with the session and are never
# saved to /kaggle/working, never committed anywhere.
import os, json, subprocess, urllib.request

HF_TOKEN    = "PASTE_YOUR_HF_TOKEN"     # https://huggingface.co/settings/tokens
NGROK_TOKEN = "PASTE_YOUR_NGROK_TOKEN"  # ngrok dashboard -> Your Authtoken

assert not HF_TOKEN.startswith("PASTE_"),    "paste your HF token"
assert not NGROK_TOKEN.startswith("PASTE_"), "paste your ngrok token"
for name, val in [("hf_token", HF_TOKEN), ("ngrok_token", NGROK_TOKEN)]:
    p = f"/tmp/{name}"
    open(p, "w").write(val)
    os.chmod(p, 0o600)

# verify the HF token (authenticated call must answer)
req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                             headers={"Authorization": f"Bearer {HF_TOKEN}"})
with urllib.request.urlopen(req, timeout=30) as r:
    who = json.loads(r.read())
print("HF token OK — user:", who.get("name"))

# ngrok binary + authtoken (static domain is configured later, in cell 7)
if not os.path.exists("/tmp/ngrok"):
    urllib.request.urlretrieve(
        "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
        "/tmp/ngrok.tgz")
    subprocess.run(["tar", "xzf", "/tmp/ngrok.tgz", "-C", "/tmp"], check=True)
r = subprocess.run(["/tmp/ngrok", "config", "add-authtoken", NGROK_TOKEN],
                   capture_output=True, text=True)
print("ngrok authtoken:", (r.stdout or r.stderr).strip()[:120])
print("CELL 3 OK — creds staged in /tmp")
