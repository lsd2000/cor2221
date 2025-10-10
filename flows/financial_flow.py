# flows/financial_flow.py
import re
from typing import Dict, Any
from rag_backend import answer_query

FIN_INTENT_PATTERNS = [
    r"\b(budget|budgeting|save|savings|financial plan|planning|invest|investment)\b",
]

ASK_GOAL   = "ask_goal"
ASK_HORIZ  = "ask_horizon"
SHOW_TIPS  = "show_tips"
DONE       = "done"

def is_financial_intent(text: str) -> bool:
    if not text:
        return False
    return any(re.search(p, text.lower()) for p in FIN_INTENT_PATTERNS)

def reset_financial_state(state: Dict[str, Any]) -> str:
    state["flow"] = "financial"
    state["flow_state"] = ASK_GOAL
    state["goal"] = None
    state["horizon"] = None
    return "Let’s plan your money 📈 What’s your main **goal** (e.g., save for family, emergency fund, pay debt)?"

def handle_financial_turn(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    cur = state.get("flow_state", ASK_GOAL)
    user = (user_text or "").strip()

    if cur == ASK_GOAL:
        if not user:
            return {"text": "What’s your main **goal**?", "used_backend": False, "used_rag": False, "sources": [], "done": False}
        state["goal"] = user
        state["flow_state"] = ASK_HORIZ
        return {"text": "What’s your **time horizon**? (e.g., 3 months, 1 year)", "used_backend": False, "used_rag": False, "sources": [], "done": False}

    if cur == ASK_HORIZ:
        if not user:
            return {"text": "Please share a rough **time horizon** (e.g., 6 months).", "used_backend": False, "used_rag": False, "sources": [], "done": False}
        state["horizon"] = user
        state["flow_state"] = SHOW_TIPS

    if state.get("flow_state") == SHOW_TIPS:
        q = (
            f"Simple financial planning steps for a migrant worker in Singapore. "
            f"Goal: {state.get('goal')}. Time horizon: {state.get('horizon')}. "
            "Keep it practical: budgeting % split, emergency fund basics, safe remittance timing, avoid scams."
        )
        res = answer_query(q)
        state["flow_state"] = DONE
        state["flow"] = None
        return {
            "text": f"Here’s a simple plan:\n\n{res.get('answer','')}\n\nYou can ask another question anytime.",
            "used_backend": True,
            "used_rag": bool(res.get("used_rag")),
            "sources": res.get("sources", []),
            "done": True
        }

    state["flow_state"] = DONE
    state["flow"] = None
    return {"text": "Okay, ending this planning flow. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
