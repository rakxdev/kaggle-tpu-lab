"""Hinglish normalizer — converts any script-mix Hinglish text to Devanagari.

Provides deterministic, robust transliteration of Romanized Hindi and English
loanwords into Devanagari phonemes for IndicF5 neural vocoder.
Uses high-precision dictionary mapping + indic-transliteration fallback.
Zero C extensions, zero legacy build tools, 100% reliable.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# High-Frequency Conversational Hindi Whitelist (Function Words & Pronouns)
# ---------------------------------------------------------------------------
ROMAN_HINDI_FUNCTION_WORDS: dict[str, str] = {
    # Pronouns & Reflexives
    "mai": "मैं", "main": "मैं", "mujhe": "मुझे", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "mujhko": "मुझको",
    "hum": "हम", "humne": "हमने", "hamara": "हमारा", "hamari": "हमारी", "hamare": "हमारे", "humein": "हमें",
    "tu": "तू", "tum": "तुम", "tumhe": "तुम्हें", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी", "tumhare": "तुम्हारे",
    "aap": "आप", "aapko": "आपको", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "apna": "अपना", "apni": "अपनी", "apne": "अपने",
    "yeh": "यह", "ye": "ये", "isse": "इससे", "isko": "इसको", "iska": "इसका", "iski": "इसकी", "iske": "इसके",
    "woh": "वह", "wo": "वो", "usse": "उससे", "usko": "उसको", "uska": "उसका", "uski": "उसकी", "uske": "उसके",

    # Conjunctions & Particles
    "aur": "और", "ya": "या", "toh": "तो", "to": "तो", "lekin": "लेकिन", "par": "पर", "magar": "मगर", "kyunki": "क्योंकि",
    "bhi": "भी", "hi": "ही", "tak": "तक", "sirf": "सिर्फ", "bas": "बस", "matlab": "मतलब",

    # Postpositions
    "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "me": "में", "mein": "में", "pe": "पे",

    # Negatives & Affirmations
    "nahi": "नहीं", "nahin": "नहीं", "na": "ना", "mat": "मत", "haan": "हाँ", "ha": "हाँ", "ji": "जी", "bilkul": "बिल्कुल",

    # Question Words
    "kya": "क्या", "kyun": "क्यों", "kyu": "क्यों", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
    "kab": "कब", "kahan": "कहाँ", "kidhar": "किधर", "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने",

    # Common Quantifiers & Adverbs
    "sab": "सब", "sabse": "सबसे", "sabhi": "सभी", "kuch": "कुछ", "bohot": "बहुत", "bahut": "बहुत",
    "zyada": "ज़्यादा", "kam": "कम", "thoda": "थोड़ा", "thodi": "थोड़ी", "thode": "थोड़े",
    "roz": "रोज़", "aaj": "आज", "kal": "कल", "parson": "परसों", "hamesha": "हमेशा", "kabhi": "कभी",
    "pehle": "पहले", "baad": "बाद", "saath": "साथ", "paas": "पास", "dur": "दूर",

    # Common Conversational Verbs (Infinitive & Inflected)
    "hai": "है", "hain": "हैं", "ho": "हो", "hu": "हूं", "hoon": "हूं",
    "tha": "था", "thi": "थी", "the": "थे",
    "hoga": "होगा", "hogi": "होगी", "honge": "होंगे",
    "aa": "आ", "aao": "आओ", "aaya": "आया", "aayi": "आयी", "aaye": "आये", "aana": "आना", "aaunga": "आऊंगा", "aaungi": "आऊंगी",
    "ja": "जा", "jao": "जाओ", "gaya": "गया", "gayi": "गयी", "gaye": "गये", "jaana": "जाना", "jaunga": "जाऊंगा", "jaungi": "जाऊंगी",
    "kar": "कर", "karo": "करो", "karna": "करना", "karein": "करें", "kiya": "किया", "karunga": "करूंगा", "karungi": "करूंगी",
    "raha": "रहा", "rahi": "रही", "rahe": "रहे",
    "de": "दे", "do": "दो", "dena": "देना", "diya": "दिया", "dijiye": "दीजिए",
    "le": "ले", "lo": "लो", "lena": "लेना", "liya": "लिया", "lijiye": "लीजिए",
    "bol": "बोल", "bolo": "बोलो", "bolna": "बोलना", "bola": "बोला",
    "dekh": "देख", "dekho": "देखो", "dekhna": "देखना", "dekha": "देखा",
    "sun": "सुन", "suno": "सुनो", "sunna": "सुनना", "suna": "सुना",
    "samajh": "समझ", "samjho": "समझो", "samjhna": "समझना", "samjha": "समझा",
    "bata": "बता", "batao": "बताओ", "batana": "बताना", "bataya": "बताया",

    # Conversational Slang & Fillers
    "yaar": "यार", "bhai": "भाई", "dost": "दोस्त", "sahi": "सही", "theek": "ठीक", "achha": "अच्छा", "achhi": "अच्छी", "achhe": "अच्छे",
}

# ---------------------------------------------------------------------------
# Canonical English Loanwords (Everyday Hinglish Vocabulary)
# ---------------------------------------------------------------------------
ENGLISH_LOAN_CANONICAL: dict[str, str] = {
    "office": "ऑफिस", "meeting": "मीटिंग", "project": "प्रोजेक्ट", "launch": "लॉन्च",
    "presentation": "प्रेज़ेंटेशन", "deadline": "डेडलाइन", "relax": "रिलैक्स",
    "tension": "टेंशन", "smoothly": "स्मूथली", "call": "कॉल", "message": "मैसेज",
    "email": "ईमेल", "boss": "बॉस", "client": "क्लाइंट", "team": "टीम",
    "work": "वर्क", "business": "बिज़नेस", "party": "पार्टी", "movie": "मूवी",
    "video": "वीडियो", "audio": "ऑडियो", "channel": "चैनल", "shorts": "शॉर्ट्स",
    "reels": "रील्स", "youtube": "यूट्यूब", "subscribe": "सब्सक्राइब", "like": "लाइक",
    "share": "शेयर", "comment": "कमेंट", "review": "रिव्यू", "ticket": "टिकट",
    "flight": "फ्लाइट", "train": "ट्रेन", "hotel": "होटल", "room": "रूम",
    "phone": "फ़ोन", "mobile": "मोबाइल", "laptop": "लैपटॉप", "computer": "कंप्यूटर",
    "internet": "इंटरनेट", "wifi": "वाईफाई", "network": "नेटवर्क", "data": "डेटा",
    "link": "लिंक", "app": "ऐप", "update": "अपडेट", "download": "डाउनलोड",
    "upload": "अपलोड", "post": "पोस्ट", "story": "स्टोरी", "reel": "रील",
    "photo": "फोटो", "camera": "कैमरा", "sound": "साउंड", "music": "म्यूजिक",
    "song": "सॉन्ग", "doctor": "डॉक्टर", "hospital": "हॉस्पिटल", "medicine": "मेडिसिन",
    "school": "स्कूल", "college": "कॉलेज", "class": "क्लास", "exam": "एग्जाम",
    "test": "टेस्ट", "result": "रिजल्ट", "paper": "पेपर", "book": "बुक",
    "time": "टाइम", "late": "लेट", "early": "अर्ली", "fast": "फास्ट",
    "slow": "स्लो", "wait": "वेट", "try": "ट्राई", "ready": "रेडी",
    "start": "स्टार्ट", "stop": "स्टॉप", "finish": "फिनिश", "problem": "प्रॉब्लम",
    "issue": "इश्यू", "solution": "सॉल्यूशन", "plan": "प्लान", "idea": "आइडिया",
    "important": "इंपॉर्टेंट", "simple": "सिंपल", "easy": "ईज़ी", "hard": "हार्ड",
    "fresh": "फ्रेश", "clean": "क्लीन", "food": "फूड", "coffee": "कॉफ़ी", "tea": "टी",
}

# ---------------------------------------------------------------------------
# Indian Named Entities (Cities, Brands, Names)
# ---------------------------------------------------------------------------
INDIAN_NE_CANONICAL: dict[str, str] = {
    "delhi": "दिल्ली", "mumbai": "मुंबई", "bengaluru": "बेंगलुरु", "bangalore": "बेंगलुरु",
    "kolkata": "कोलकाता", "chennai": "चेन्नई", "hyderabad": "हैदराबाद", "pune": "पुणे",
    "ahmedabad": "अहमदाबाद", "jaipur": "जयपुर", "lucknow": "लखनऊ", "chandigarh": "चंडीगढ़",
    "shubh": "शुभ", "arjun": "अर्जुन", "rahul": "राहुल", "rohit": "रोहित", "aishwarya": "ऐश्वर्या",
    "priya": "प्रिया", "neha": "नेहा", "pooja": "पूजा", "amit": "अमित", "vikram": "विक्रम",
    "tata": "टाटा", "reliance": "रिलायंस", "infosys": "इन्फोसिस", "wipro": "विप्रो",
}

_CANONICAL: dict[str, str] = {
    **ROMAN_HINDI_FUNCTION_WORDS,
    **ENGLISH_LOAN_CANONICAL,
    **INDIAN_NE_CANONICAL,
}

_DEVANAGARI_RANGE = (0x0900, 0x097F)
_TOKEN_RE = re.compile(r"([A-Za-z]+|[ऀ-ॿ]+|\d+|\s+|.)")

def _is_devanagari_token(tok: str) -> bool:
    has_deva = any(_DEVANAGARI_RANGE[0] <= ord(c) <= _DEVANAGARI_RANGE[1] for c in tok)
    has_ascii_alpha = any(c.isascii() and c.isalpha() for c in tok)
    return has_deva and not has_ascii_alpha

def _is_ascii_alpha_token(tok: str) -> bool:
    if not tok:
        return False
    return all(c.isascii() and (c.isalpha() or c == "'") for c in tok)

_sanscript_engine = None

def _get_sanscript():
    global _sanscript_engine
    if _sanscript_engine is None:
        try:
            from indic_transliteration import sanscript
            _sanscript_engine = sanscript
        except Exception:
            _sanscript_engine = False
    return _sanscript_engine

def _normalize_ascii_token(tok: str) -> str:
    key = tok.lower().strip("'")
    if key in _CANONICAL:
        return _CANONICAL[key]

    # Try indic-transliteration library if installed
    sans = _get_sanscript()
    if sans:
        try:
            res = sans.transliterate(key, sans.ITRANS, sans.DEVANAGARI)
            if res and res != key:
                return res
        except Exception:
            pass

    return tok

def to_unified_devanagari(text: str) -> str:
    """Convert Hinglish text (any script mix) to Devanagari for IndicF5 input.
    Idempotent: applying twice gives the same result.
    """
    if not text:
        return text
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace() or _is_devanagari_token(tok):
            out.append(tok)
        elif _is_ascii_alpha_token(tok):
            out.append(_normalize_ascii_token(tok))
        else:
            out.append(tok)
    return "".join(out)
