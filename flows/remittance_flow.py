# flows/remittance_flow.py
import re
from typing import Dict, Any
from rag_backend import answer_query

# Simple intent patterns (expand freely)
REMITTANCE_INTENT_PATTERNS = [
    r"\b(remit|remittance|send money|transfer (money|funds)?)\b",
    r"\b(remesa|remitir)\b",  # optional non-English examples
]

# States for a tiny finite-state machine
ASK_COUNTRY   = "ask_country"
ASK_METHOD    = "ask_method"
ASK_AMOUNT    = "ask_amount_optional"
SHOW_OPTIONS  = "show_options"
OFFER_BUDGET  = "offer_budget"
DONE          = "done"

METHOD_MAP = {
    "bank": "bank transfer",
    "bank transfer": "bank transfer",
    "cash": "cash pickup",
    "pickup": "cash pickup",
    "cash pickup": "cash pickup",
    "wallet": "mobile wallet",
    "mobile": "mobile wallet",
    "mobile wallet": "mobile wallet",
}

def is_remittance_intent(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(pat, low) for pat in REMITTANCE_INTENT_PATTERNS)

def reset_remittance_state(state: Dict[str, Any]) -> str:
    """Initialize flow slots in Streamlit session_state-like dict."""
    state["flow"] = "remittance"
    state["flow_state"] = ASK_COUNTRY
    state["country"] = None
    state["method"] = None
    state["amount"] = None
    return (
        "Let’s sort out **remittance** ✨\n"
        "Which **country** do you usually send money to?"
    )

def _normalize_method(text: str) -> str | None:
    low = text.lower().strip()
    for key, val in METHOD_MAP.items():
        if key in low:
            return val
    return None

def handle_remittance_turn(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "text": str,               # assistant message to show
        "used_backend": bool,      # whether we called answer_query()
        "used_rag": bool,          # propagated from backend
        "sources": list[str],      # propagated from backend
        "done": bool               # flow finished
      }
    """
    cur = state.get("flow_state", ASK_COUNTRY)
    user = (user_text or "").strip()

    # ---- Step 1: Ask destination country
    if cur == ASK_COUNTRY:
        if not user:
            return {"text": "Which **country** do you usually send money to?", "used_backend": False, "used_rag": False, "sources": [], "done": False}
        state["country"] = user
        state["flow_state"] = ASK_METHOD
        return {
            "text": f"Got it — **{state['country']}**. Do you prefer **bank transfer**, **cash pickup**, or **mobile wallet**?",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    # ---- Step 2: Ask preferred method
    if cur == ASK_METHOD:
        method = _normalize_method(user)
        if not method:
            return {
                "text": "Please choose one: **bank transfer**, **cash pickup**, or **mobile wallet**.",
                "used_backend": False, "used_rag": False, "sources": [], "done": False
            }
        state["method"] = method
        state["flow_state"] = ASK_AMOUNT
        return {
            "text": f"Okay — **{method}**. (Optional) About how much do you usually send **per month**? You can reply with an amount like `200` or say **skip**.",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    # ---- Step 3: Optional amount
    if cur == ASK_AMOUNT:
        if user and user.lower() != "skip":
            # naive amount capture
            nums = re.findall(r"\d+(?:\.\d+)?", user.replace(",", ""))
            if nums:
                state["amount"] = nums[0]
        state["flow_state"] = SHOW_OPTIONS

    # ---- Step 4: Show options (call backend with focused query)
    if state.get("flow_state") == SHOW_OPTIONS:
        ctry = state.get("country") or "the destination country"
        meth = state.get("method") or "a suitable method"
        amt  = state.get("amount")

        focus = f" from Singapore to {ctry} via {meth}"
        if amt:
            focus += f" for about SGD {amt} per month"
        query = (
            "Remittance guidance" + focus +
            ". Cover fees range if available, typical transfer times, "
            "required documents/KYC, and safety tips. "
            "If context is insufficient, provide high-level safe guidance. "
            "Be clear and simple."
        )
        res = answer_query(query)
        state["flow_state"] = OFFER_BUDGET
        text = (
            f"Here’s what to expect when sending money to **{ctry}** via **{meth}**:"
            f"\n\n{res.get('answer','')}\n\n"
            "Would you also like **simple budgeting tips** to help plan your remittances each month? (yes/no)"
        )
        return {
            "text": text,
            "used_backend": True,
            "used_rag": bool(res.get("used_rag")),
            "sources": res.get("sources", []),
            "done": False
        }

    # ---- Step 5: Offer budgeting next
    if state.get("flow_state") == OFFER_BUDGET:
        if user.lower() in ("yes", "y", "yeah", "ok", "okay", "sure"):
            ctry = state.get("country") or "home country"
            q = (
                f"Budgeting tips for migrant workers in Singapore who remit monthly to {ctry}. "
                "Make it practical: % to save, small emergency fund, reminders for fee timing, "
                "and caution against scams."
            )
            res = answer_query(q)
            state["flow_state"] = DONE
            state["flow"] = None
            return {
                "text": f"Great — here are some budgeting tips:\n\n{res.get('answer','')}\n\nYou can ask another question anytime.",
                "used_backend": True, "used_rag": bool(res.get("used_rag")), "sources": res.get("sources", []), "done": True
            }
        else:
            state["flow_state"] = DONE
            state["flow"] = None
            return {
                "text": "No worries. You can ask another question anytime.",
                "used_backend": False, "used_rag": False, "sources": [], "done": True
            }

    # Fallback
    state["flow_state"] = DONE
    state["flow"] = None
    return {"text": "Okay, ending this remittance flow. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
