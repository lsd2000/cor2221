# flows/financial_flow.py
import re
from typing import Dict, Any
from rag_backend import answer_query

"""
Financial planning flow (aligned to your corpus):
- If planning guidance is thin, we explicitly guide users to bank account setup,
  so RAG can ground from:
  - "documents required for account opening"
  - "posb payroll account for work permit holders in singapore"
  - "financial institution directory"
  - "mw handy guide english"
  - PayLah/PayNow on-ramps if relevant
"""

FIN_INTENT_PATTERNS = [
    r"\b(budget|budgeting|save|savings|financial plan|planning|invest|investment)\b",
    r"\b(expense|expenses|spend|spending|emergency fund)\b",
]

ASK_GOAL     = "ask_goal"
ASK_HORIZ    = "ask_horizon"
ASK_INCOME   = "ask_income_optional"
ASK_BANKPATH = "ask_bankpath"      # NEW state: offer bank setup guidance
SHOW_TIPS    = "show_tips"
DONE         = "done"

# Keywords to bias RAG toward your uploaded docs (include exact title cues + topic terms)
_DOC_KEYS = (
    # Titles (lowercase forms) and distinctive phrases likely present in content
    "documents required for work pass",
    "your guide to paylah",
    "transfer funds using dbs paylah",
    "financial institution directory",
    "eremittance guide to sending money home safely",
    "documents required for account opening",
    "posb payroll account for work permit holders in singapore",
    "mw handy guide english",
    # Topic anchors
    "paylah", "paynow", "dbs paylah", "dbs paynow",
    "account opening", "required documents", "kyc", "work permit", "s pass", "employment pass",
    "posb payroll", "minimum balance", "bank fees", "limits", "transfer", "remittance"
)

# A smaller planning-oriented gate for general tips
_PLAN_KEYS = (
    "budget", "budgeting", "savings", "emergency fund",
    "remittance", "fees", "fx", "limits", "paylah", "paynow",
    "bank account", "account opening", "posb payroll", "financial institution directory",
    "migrant worker", "handy guide"
)

def is_financial_intent(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(p, low) for p in FIN_INTENT_PATTERNS)

def reset_financial_state(state: Dict[str, Any]) -> str:
    state["flow"] = "financial"
    state["flow_state"] = ASK_GOAL
    state["goal"] = None
    state["horizon"] = None
    state["income"] = None
    return (
        "Let’s plan your money 📈\n"
        "What’s your main **goal**? (e.g., save for family, emergency fund, pay debt)"
    )

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
        state["flow_state"] = ASK_INCOME
        return {"text": "Optional: What’s your **monthly income** in SGD? (e.g., 900) You can also reply **skip**.", "used_backend": False, "used_rag": False, "sources": [], "done": False}

    if cur == ASK_INCOME:
        if user and user.lower() != "skip":
            nums = re.findall(r"\d+(?:\.\d+)?", user.replace(",", ""))
            if nums:
                state["income"] = nums[0]
        # Before tips, offer a targeted bank-setup path to ground on account-opening docs.
        state["flow_state"] = ASK_BANKPATH
        return {
            "text": (
                "Would you like help **setting up a simple bank account** first? "
                "This can make saving and remitting easier (e.g., POSB payroll, low minimum balance, PayLah/PayNow). (yes/no)"
            ),
            "used_backend": False, "used_rag": False, "sources": [], "done": False
        }

    if cur == ASK_BANKPATH:
        if user.lower() in ("yes", "y", "yeah", "ok", "okay", "sure"):
            # Explicitly craft a query that names your corpus cues so retrieval prefers those docs.
            q = (
                "Step-by-step **bank account setup** in Singapore for a migrant worker. "
                "Cover: **documents required for account opening** (KYC), options for **low or no minimum balance**, "
                "**POSB payroll account for Work Permit holders**, and how **PayLah/PayNow** link to accounts. "
                "Keep it simple and practical. "
                "Use uploaded context such as 'documents required for account opening', "
                "'posb payroll account for work permit holders in singapore', "
                "'financial institution directory', and 'mw handy guide english' if available."
            )
            res = answer_query(q, require_keywords=_DOC_KEYS)
            # After bank path, still give planning tips next
            state["bank_path_answer"] = res.get("answer", "")
            state["bank_path_sources"] = res.get("sources", [])
        # Move on to tips regardless of yes/no
        state["flow_state"] = SHOW_TIPS

    if state.get("flow_state") == SHOW_TIPS:
        income_part = f" Monthly income: SGD {state.get('income')}." if state.get("income") else ""
        q = (
            "Financial planning tips tailored for a migrant worker in Singapore."
            f" Goal: {state.get('goal')}. Time horizon: {state.get('horizon')}.{income_part} "
            "Keep it practical and simple: budgeting % split (needs/remittance/savings), "
            "small emergency fund, **remittance fee/FX basics**, and **how PayLah/PayNow can help for day-to-day**. "
            "If available, ground guidance using 'mw handy guide english', 'your guide to paylah', "
            "'transfer funds using dbs paylah', and 'financial institution directory'."
        )
        res = answer_query(q, require_keywords=_PLAN_KEYS + _DOC_KEYS)

        # Stitch bank-path answer (if any) before tips
        bank_block = ""
        if state.get("bank_path_answer"):
            bank_block = f"**Bank account setup (quick guide):**\n\n{state['bank_path_answer']}\n\n"

        state["flow_state"] = DONE
        state["flow"] = None
        return {
            "text": f"{bank_block}**Simple plan:**\n\n{res.get('answer','')}\n\nYou can ask another question anytime.",
            "used_backend": True,
            "used_rag": bool(res.get("used_rag")) or bool(state.get("bank_path_sources")),
            "sources": list(dict.fromkeys((state.get("bank_path_sources") or []) + (res.get("sources") or []))),
            "done": True
        }

    state["flow_state"] = DONE
    state["flow"] = None
    return {"text": 'Okay, ending this planning flow. Ask me anything else.', "used_backend": False, "used_rag": False, "sources": [], "done": True}
