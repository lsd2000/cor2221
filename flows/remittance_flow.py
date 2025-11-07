# flows/remittance_flow.py
import re
from typing import Dict, Any
from rag_backend import answer_query

"""
Remittance flow aligned to your corpus:
- Prefers grounding from:
  - "eremittance guide to sending money home safely"
  - "your guide to paylah" / "transfer funds using dbs paylah"
  - "financial institution directory"
  - "documents required for account opening" / "posb payroll account..."
  - "mw handy guide english"
- Adds PayLah/PayNow explicitly as methods to increase retrieval hits.
"""

REMITTANCE_INTENT_PATTERNS = [
    r"\b(remit|remittance|send money|transfer (money|funds)?)\b",
    r"\b(remesa|remitir)\b",
]

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
    "paylah": "PayLah",
    "paynow": "PayNow",
}

# Doc/title cues + topic anchors to drive RAG selection
_DOC_KEYS = (
    "eremittance guide to sending money home safely",
    "your guide to paylah",
    "transfer funds using dbs paylah",
    "financial institution directory",
    "documents required for account opening",
    "posb payroll account for work permit holders in singapore",
    "mw handy guide english",
    "documents required for work pass",
    # Topic anchors
    "paylah", "paynow", "dbs paylah", "dbs paynow",
    "kyc", "required documents", "account opening", "work permit", "s pass", "employment pass",
    "fees", "charges", "exchange rate", "fx", "limits", "cash pickup", "bank transfer", "mobile wallet"
)

def is_remittance_intent(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(pat, low) for pat in REMITTANCE_INTENT_PATTERNS)

def reset_remittance_state(state: Dict[str, Any]) -> str:
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
    low = (text or "").lower().strip()
    for key, val in METHOD_MAP.items():
        if key in low:
            return val
    return None

def handle_remittance_turn(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns dict: { text, used_backend, used_rag, sources, done }
    """
    cur = state.get("flow_state", ASK_COUNTRY)
    user = (user_text or "").strip()

    if cur == ASK_COUNTRY:
        if not user:
            return {"text": "Which **country** do you usually send money to?", "used_backend": False, "used_rag": False, "sources": [], "done": False}
        state["country"] = user.title()
        state["flow_state"] = ASK_METHOD
        return {
            "text": f"Got it — **{state['country']}**. Do you prefer **bank transfer**, **cash pickup**, **mobile wallet**, **PayLah**, or **PayNow**?",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    if cur == ASK_METHOD:
        method = _normalize_method(user)
        if not method:
            return {
                "text": "Please choose one: **bank transfer**, **cash pickup**, **mobile wallet**, **PayLah**, or **PayNow**.",
                "used_backend": False, "used_rag": False, "sources": [], "done": False
            }
        state["method"] = method
        state["flow_state"] = ASK_AMOUNT
        return {
            "text": f"Okay — **{method}**. (Optional) About how much do you usually send **per month**? You can reply with an amount like `200` or say **skip**.",
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    if cur == ASK_AMOUNT:
        if user and user.lower() != "skip":
            nums = re.findall(r"\d+(?:\.\d+)?", user.replace(",", ""))
            if nums:
                state["amount"] = nums[0]
        state["flow_state"] = SHOW_OPTIONS

    if state.get("flow_state") == SHOW_OPTIONS:
        ctry = state.get("country") or "the destination country"
        meth = state.get("method") or "a suitable method"
        amt  = state.get("amount")

        focus = f" from Singapore to {ctry} via {meth}"
        if amt:
            focus += f" for about SGD {amt} per month"

        # Include explicit doc-title cues in the query text to help the retriever match metadata/content.
        query = (
            "Remittance guidance" + focus +
            ". Cover: step-by-step sending process, **required documents/KYC**, expected **fees/FX considerations**, "
            "**transfer times/limits**, and safety tips for migrants. "
            "Refer to uploaded materials such as 'eremittance guide to sending money home safely', "
            "'your guide to paylah', 'transfer funds using dbs paylah', 'financial institution directory', "
            "'documents required for account opening', 'posb payroll account for work permit holders in singapore', "
            "and 'mw handy guide english' where relevant."
        )
        res = answer_query(query, require_keywords=_DOC_KEYS)

        state["flow_state"] = OFFER_BUDGET
        text = (
            f"Here’s what to expect when sending money to **{ctry}** via **{meth}**:"
            f"\n\n{res.get('answer','')}\n\n"
            "Would you also like **simple budgeting tips** to help plan your remittances each month? (yes/no)"
        )
        return {
            "text": text,
            "used_backend": True, "used_rag": bool(res.get("used_rag")), "sources": res.get("sources", []),
            "done": False
        }

    if state.get("flow_state") == OFFER_BUDGET:
        if user.lower() in ("yes", "y", "yeah", "ok", "okay", "sure"):
            ctry = state.get("country") or "home country"
            q = (
                f"Budgeting tips for migrant workers in Singapore who remit monthly to {ctry}. "
                "Be practical and simple: a % split for needs/remittance/savings, small emergency fund, "
                "remittance fee timing/FX basics, and caution against scams. "
                "If available, ground guidance using 'mw handy guide english', 'financial institution directory', "
                "and PayLah/PayNow guides."
            )
            res = answer_query(q, require_keywords=_DOC_KEYS)
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

    state["flow_state"] = DONE
    state["flow"] = None
    return {"text": "Okay, ending this remittance flow. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
