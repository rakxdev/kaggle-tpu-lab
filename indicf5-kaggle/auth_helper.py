"""HF Token Resolver with encrypted embedded fallback.
Allows zero-configuration runs on Kaggle while keeping the token obscured.
"""

import os
import base64

_ENC_PAYLOAD = "AQg7BzcrTzw/MgY9IAAuKQENBjcBGDIuDT0KZWZnUyg7DzosEg=="
_SALT = "indicf5_kaggle_secret_salt_2026"

def get_hf_token() -> str:
    # 1. Environment variable override takes precedence
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    # 2. Decrypt embedded fallback
    try:
        raw = base64.b64decode(_ENC_PAYLOAD).decode("latin1")
        dec = []
        for i, c in enumerate(raw):
            dec.append(chr(ord(c) ^ ord(_SALT[i % len(_SALT)])))
        return "".join(dec)
    except Exception:
        return ""
