# ngrok static domain — one-time setup (~4 minutes)

A free ngrok account gives **one permanent domain**. The tunnel URL then stops
changing, so restarting Colab does not invalidate the link handed to judges —
which is the single biggest weakness of the cloudflared setup.

## 1. Create the account

1. https://dashboard.ngrok.com/signup — free, GitHub/Google sign-in works
2. Verify your email

## 2. Copy your authtoken

https://dashboard.ngrok.com/get-started/your-authtoken

Looks like: `2abcDEF...long-string...xyz`

## 3. Claim your free static domain

https://dashboard.ngrok.com/domains → **New Domain**

The free plan includes exactly one. You get something like:
```
lucky-mammal-42.ngrok-free.app
```
Copy it **without** `https://`.

## 4. Add both to Colab Secrets

🔑 icon in the Colab sidebar → add two, each with **Notebook access ON**:

| Name | Value |
|---|---|
| `NGROK_TOKEN` | the authtoken from step 2 |
| `NGROK_DOMAIN` | `lucky-mammal-42.ngrok-free.app` |

## 5. Run the restart cell

Paste `notebooks/RESTART_CELL.py` into a Colab cell and run it. It will use
ngrok with your static domain and print the URL — the same URL, every time.

If the secrets are missing it falls back to cloudflared with a random URL, so
it still works, just without the stable name.

---

## What this fixes, and what it does not

**Fixes:** the URL no longer changes. Restart Colab as often as you like; the
link in your submission stays valid.

**Does not fix:** Colab still stops after ~90 minutes idle and hard-stops at
~12 hours. You still need to re-run the restart cell after that — but now the
URL it prints is the one judges already have.

**Note on the ngrok interstitial:** free ngrok shows a browser warning page on
first visit. Click through it once. API calls from the page are unaffected —
the app sends the `ngrok-skip-browser-warning` header.
