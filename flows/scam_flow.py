# flows/scam_flow.py
import re
from typing import Dict, Any
from rag_backend import answer_query

# --- Intent detection ---
SCAM_INTENT_PATTERNS = [
    r"\b(scam|scammer|suspicious|fraud|cheat(ed)?|fake|impersonat(e|or)|phishing)\b",
    r"\b(loan shark|ah long|moneylender scam|job scam|love scam|investment scam)\b",
    r"\b(agent fee|upfront fee|processing fee|deposit|gift card|crypto|bitcoin)\b",
    r"\b(OTP|one[- ]time password|password|bank account|transfer now)\b",
    r"\b(MOM|ICA|police|bank) (call(ed)?|message(d)?|email(ed)?) me\b",
    r"\b(suspect|not sure|too good to be true)\b",
]

def is_scam_intent(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(pat, low) for pat in SCAM_INTENT_PATTERNS)

# --- Simple FSM states ---
ASK_SCENARIO   = "ask_scenario"     # what happened?
ASK_CHANNEL    = "ask_channel"      # where did it happen? (SMS/WhatsApp/Call/Web/Agent)
ASK_REQUESTS   = "ask_requests"     # what did they ask from you? (money/OTP/ID)
SUMMARIZE_RISK = "summarize_risk"   # call backend for tailored guidance
PROVIDE_STEPS  = "provide_steps"    # show concrete steps; offer reporting
DONE           = "done"

CHANNEL_MAP = {
    "sms": "SMS",
    "text": "SMS",
    "whatsapp": "WhatsApp",
    "wechat": "WeChat",
    "telegram": "Telegram",
    "call": "Phone call",
    "phone": "Phone call",
    "email": "Email",
    "site": "Website",
    "web": "Website",
    "website": "Website",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "agent": "In-person agent",
    "in person": "In-person agent",
}

REQUEST_KEYWORDS = [
    "upfront fee", "processing fee", "deposit", "gift card", "crypto", "bitcoin",
    "bank transfer", "paynow", "otp", "one-time password", "password",
    "nric", "passport", "work permit", "bank details", "account number"
]

def reset_scam_state(state: Dict[str, Any]) -> str:
    state["flow"] = "scam"
    state["flow_state"] = ASK_SCENARIO
    state["scam_scenario"] = None
    state["scam_channel"] = None
    state["scam_requests"] = None
    return (
        "I’m here to help you stay safe. 🛡️\n\n"
        "**What happened?** Please describe the message/call/offer in your own words."
    )

def _normalize_channel(text: str) -> str | None:
    low = (text or "").lower()
    for k, v in CHANNEL_MAP.items():
        if k in low:
            return v
    return None

def _extract_requests(text: str) -> list[str]:
    low = (text or "").lower()
    hits = []
    for k in REQUEST_KEYWORDS:
        if k in low:
            hits.append(k)
    # capture money amounts if any
    amounts = re.findall(r"\b(?:sgd|s\$|\$)?\s?\d{1,4}(?:[.,]\d{2})?\b", low)
    if amounts:
        hits.extend([a.strip() for a in amounts])
    return list(dict.fromkeys(hits))  # dedupe, keep order

def handle_scam_turn(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "text": str,
        "used_backend": bool,
        "used_rag": bool,
        "sources": list[str],
        "done": bool
      }
    """
    cur = state.get("flow_state", ASK_SCENARIO)
    user = (user_text or "").strip()

    # Step 1: Scenario
    if cur == ASK_SCENARIO:
        if not user:
            return {
                "text": "Please describe the situation (what was said/sent to you).",
                "used_backend": False, "used_rag": False, "sources": [], "done": False
            }
        state["scam_scenario"] = user
        state["flow_state"] = ASK_CHANNEL
        return {
            "text": "Where did this happen? (e.g., **SMS**, **WhatsApp**, **Phone call**, **Website**, **In-person agent**)",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    # Step 2: Channel
    if cur == ASK_CHANNEL:
        ch = _normalize_channel(user)
        if not ch:
            return {
                "text": "Please tell me the channel: **SMS**, **WhatsApp/WeChat/Telegram**, **Phone call**, **Email/Website**, or **In-person agent**.",
                "used_backend": False, "used_rag": False, "sources": [], "done": False
            }
        state["scam_channel"] = ch
        state["flow_state"] = ASK_REQUESTS
        return {
            "text": f"Thanks. Did they ask for anything like **money (upfront/fees)**, **bank details**, or your **OTP/passport**? "
                    "Feel free to paste exact wording. If nothing specific, you can say **not sure**.",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    # Step 3: Requests they asked for
    if cur == ASK_REQUESTS:
        hits = _extract_requests(user)
        state["scam_requests"] = hits or (["not sure"] if user.lower() == "not sure" else [])
        state["flow_state"] = SUMMARIZE_RISK

    # Step 4: Summarize risk (use backend to produce tailored guidance)
    if state.get("flow_state") == SUMMARIZE_RISK:
        scenario = state.get("scam_scenario") or ""
        channel  = state.get("scam_channel") or "Unknown channel"
        requests = ", ".join(state.get("scam_requests") or []) or "No specific requests"
        # Focus the backend on scam safety guidance. It will use RAG if present, else general fallback.
        query = (
            "Scam safety check for a migrant worker in Singapore. "
            f"Channel: {channel}. Key details: {scenario}. Requests mentioned: {requests}. "
            "Identify red flags in bullet points, then give clear DO/DON'T steps in simple English. "
            "Emphasize: do not share OTP/password, do not pay upfront fees or deposits to strangers, "
            "verify with official channels directly, and stop contact if pressured."
        )
        res = answer_query(
    query,
    require_keywords=(
        "scam","fraud","phishing","impersonation","anti-scam","report",
        "police","otp","password","upfront fee","deposit","processing fee",
        "bank details","account number"
    )
)

        state["flow_state"] = PROVIDE_STEPS
        text = (
            "**Let’s review this safely:**\n\n"
            f"{res.get('answer','')}\n\n"
            "If you like, I can also show **how to report** and where to get official help. Would you like that? (yes/no)"
        )
        return {
            "text": text,
            "used_backend": True, "used_rag": bool(res.get("used_rag")), "sources": res.get("sources", []),
            "done": False
        }

    # Step 5: Provide reporting steps (general guidance, no specific phone numbers here)
    if state.get("flow_state") == PROVIDE_STEPS:
        if user.lower() in ("yes", "y", "yeah", "ok", "okay", "sure"):
            # Keep high-level but practical, without hardcoding numbers/URLs
            tips = (
                "Here are safe next steps:\n\n"
                "1) **Stop contact** with the sender/caller. Do not click links or scan QR codes.\n"
                "2) **Do not share** OTP, passwords, banking details, or ID images.\n"
                "3) **Verify independently** with official sources (e.g., visit the agency/bank’s official site or hotline from their official page).\n"
                "4) **Document** the evidence (screenshots, phone numbers, usernames) in case you need to report.\n"
                "5) **Report** through official Singapore channels (e.g., national anti-scam resources or the police e-services portal). "
                "Use only contacts listed on the official websites.\n"
                "6) If you already sent money or shared details, **contact your bank immediately** to secure your account.\n"
            )
            state["flow_state"] = DONE
            state["flow"] = None
            return {
                "text": tips + "\nStay safe. You can ask me anything else anytime.",
                "used_backend": False, "used_rag": False, "sources": [], "done": True
            }
        else:
            state["flow_state"] = DONE
            state["flow"] = None
            return {
                "text": "No problem. Stay safe — and feel free to ask anything else.",
                "used_backend": False, "used_rag": False, "sources": [], "done": True
            }

    # Fallback
    state["flow_state"] = DONE
    state["flow"] = None
    return {"text": "Okay, ending this safety check. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
