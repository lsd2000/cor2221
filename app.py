# app.py
import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import markdown as md

# app.py (top imports)
from flows.remittance_flow import (
    is_remittance_intent, reset_remittance_state, handle_remittance_turn
)
from flows.financial_flow import (
    is_financial_intent, reset_financial_state, handle_financial_turn
)

from flows.scam_flow import (
    is_scam_intent, reset_scam_state, handle_scam_turn
)

from rag_backend import answer_query  # backend function

def _apply_secrets_to_env():
    try:
        # st.secrets is a toml-like dict; flatten one level if needed
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[str(k)] = str(v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    os.environ[str(kk)] = str(vv)
    except Exception:
        pass  # running locally without st.secrets

_apply_secrets_to_env()

load_dotenv()
st.set_page_config(page_title="SG RAG Chat", page_icon="🇸🇬", layout="wide")

# push content down so the toolbar doesn't overlap
st.markdown("""
<style>
  .block-container {max-width: 1400px;}
  .stVerticalBlock {
      margin: 0 !important;
      padding: 0 !important;
      gap: 0 !important;   /* remove spacing between children */
  div[data-testid="stLayoutWrapper"] {
    padding-top: 4px;    /* adjust top padding */
    padding-bottom: 4px; /* adjust bottom padding */
}
    div[data-testid="stFormSubmitButton"] {
    display: none !important;        
}
    label[for="text_input_1"] {
    display: none;
    }
div[data-testid="stAppIframeResizerAnchor"] {
        display: none;
    }
            
</style>
""", unsafe_allow_html=True)

# app.py (top) – after set_page_config and before rendering
# Runner: if a backend call is pending, do it now (after we've already shown the loader on the previous run)
if st.session_state.get("run_backend"):
    st.session_state.run_backend = False

    # Plain Q&A path
    if st.session_state.get("pending_query"):
        from rag_backend import answer_query
        q = st.session_state.pending_query
        st.session_state.pending_query = ""
        res = answer_query(q)
        ans = (res.get("answer") or "").strip()
        used_rag = bool(res.get("used_rag"))
        sources = res.get("sources") or []
        meta = "Answer grounded in uploaded context ✅" if used_rag else "General knowledge fallback ⚠️"

        # Replace loader
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
            st.session_state.messages[-1] = {
                "role": "assistant", "content": ans, "meta": meta, "sources": sources
            }
        else:
            st.session_state.messages.append({"role": "assistant", "content": ans, "meta": meta, "sources": sources})

        st.rerun()

    # Flow path (remittance / financial), run one flow step now
    flow_name = st.session_state.get("pending_flow")
    flow_input = st.session_state.get("flow_pending_input", "")
    if flow_name and flow_input:
        # import here to avoid circulars at import time
        from flows.remittance_flow import handle_remittance_turn
        from flows.financial_flow import handle_financial_turn

        if flow_name == "remittance":
            out = handle_remittance_turn(flow_input, st.session_state)
        elif flow_name == "financial":
            out = handle_financial_turn(flow_input, st.session_state)
        elif flow_name == "scam":
            out = handle_scam_turn(flow_input, st.session_state)
        else:
            out = {"text": "Okay, ending current flow. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
            st.session_state.flow = None
            st.session_state.flow_state = None

        meta = "Flow: Guidance" if not out.get("used_backend") else (
            "Answer grounded in uploaded context ✅" if out.get("used_rag") else "General knowledge fallback ⚠️"
        )

        # Replace loader with flow output
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
            st.session_state.messages[-1] = {"role": "assistant", "content": out.get("text",""), "meta": meta, "sources": out.get("sources", [])}
        else:
            st.session_state.messages.append({"role": "assistant", "content": out.get("text",""), "meta": meta, "sources": out.get("sources", [])})

        # Clear pending flags
        st.session_state.pending_flow = None
        st.session_state.flow_pending_input = ""

        # End flow if done
        if out.get("done"):
            st.session_state.flow = None
            st.session_state.flow_state = None

        st.rerun()

# No sidebar; session messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta, sources}]

if "flow" not in st.session_state:
    st.session_state.flow = None          # "remittance" | "financial" | None
if "flow_state" not in st.session_state:
    st.session_state.flow_state = None

# Header + clear
c1, c2 = st.columns([6, 1])
with c1:
    st.markdown("### Singapore Integration Assistant (RAG-first)")
    st.markdown(
        """
        <hr style="margin:0; border:1px solid #888;">
        """,
        unsafe_allow_html=True
    )
with c2:
    if st.button("Clear", help="Clear this conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- conversation renderer ----------
def render_messages_html(messages):
    html_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        meta = m.get("meta", "")
        sources = m.get("sources", [])
        klass = "user" if role == "user" else "bot"
        html_content = md.markdown(content, extensions=["extra", "sane_lists"])
        bubble = f'<div class="bubble {klass}">{html_content}</div>'
        meta_line = f'<div class="meta">{meta}</div>' if meta else ""
        src_line = ""
        if role == "assistant" and sources:
            uniq = []
            for s in sources:
                name = str(s).split("/")[-1].split("\\")[-1]
                if name not in uniq:
                    uniq.append(name)
            src_line = f'<div class="srcs">Sources: {", ".join(uniq[:8])}</div>'
        html_parts.append(bubble + meta_line + src_line)
    return "\n".join(html_parts)

# Load template
template_path = os.path.join("templates", "chat.html")
with open(template_path, "r", encoding="utf-8") as f:
    template_html = f.read()

# Build conversation HTML
messages_html = render_messages_html(st.session_state.messages)
final_html = template_html.replace("{{MESSAGES_HTML}}", messages_html)

# Show conversation
components.html(final_html, height=500, scrolling=True)

# ---------- INPUT AT THE BOTTOM (Form = Enter submits) ----------
# ---------- QUICK ACTION BUTTONS ----------
st.markdown(
    """
    <style>
    /* Adjust button appearance */
    .stButton>button {
        width: 100%;
        border-radius: 100px;
        font-weight: 600;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

cols = st.columns(3, gap="small")

# Each button label, message, and optional emoji
quick_actions = [
    ("💰 Finance", "I want to ask about financial planning."),
    ("💸 Remittance", "I want to send money overseas."),
    ("🚨 Scam", "I think I might be a victim of a scam.")
]

cols = st.columns([0.1, 0.12, 0.12,0.7])  # last column is empty for spacing

for i, (label, msg) in enumerate(quick_actions):
    with cols[i]:
        if st.button(label, key=f"btn_{label}"):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": msg, "meta": "", "sources": []})

            # Reuse your existing flow detection logic
            from flows.remittance_flow import is_remittance_intent, reset_remittance_state
            from flows.financial_flow import is_financial_intent, reset_financial_state
            from flows.scam_flow import is_scam_intent, reset_scam_state

            if is_remittance_intent(msg):
                first_msg = reset_remittance_state(st.session_state)
                st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Remittance", "sources": []})
                st.rerun()
            elif is_financial_intent(msg):
                first_msg = reset_financial_state(st.session_state)
                st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Financial Planning", "sources": []})
                st.rerun()
            elif is_scam_intent(msg):
                first_msg = reset_scam_state(st.session_state)
                st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Scam Safety", "sources": []})
                st.rerun()
            else:
                # Fallback to normal backend Q&A
                st.session_state.pending_query = msg
                st.session_state.run_backend = True
                st.rerun()


# --- FORM SUBMIT (bottom) ---
import streamlit as st
import streamlit.components.v1 as components

col_input, col_button = st.columns([9, 1])

with col_input:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "", 
            key="chatbox",
            placeholder="Ask Anything..."
        )
        # Hidden submit
        submitted = st.form_submit_button("Send", type="primary", key="hidden_submit")

with col_button:
    components.html("""
    <div style="height:100%; display:flex; align-items:center;">
        <button id="customSend" style="
            padding: 8px 16px; 
            margin-top:15px;
            border-radius: 12px; 
            font-weight: 600;
            font-size: 14px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            width: 100%;
        ">Send</button>
    </div>

    <script>
    setTimeout(() => {
        const hiddenSubmit = window.parent.document.querySelector('div[data-testid="stFormSubmitButton"] button');
        const customButton = document.getElementById("customSend");

        customButton.onclick = () => {
            if(hiddenSubmit){
                hiddenSubmit.click();  // trigger the hidden Streamlit submit button
            }
        }
    }, 500);  // wait a bit to ensure the button exists
    </script>
    """, height=100)

if submitted and user_input.strip():
    ui_q = user_input.strip()

    # 1) Show user message immediately
    st.session_state.messages.append({"role": "user", "content": ui_q, "meta": "", "sources": []})

    # 2) FLOW ROUTING comes BEFORE any backend scheduling
    from flows.remittance_flow import is_remittance_intent, reset_remittance_state
    from flows.financial_flow import is_financial_intent, reset_financial_state

    # If there is no active flow, try to start one
    if not st.session_state.get("flow"):
        if is_remittance_intent(ui_q):
            first_msg = reset_remittance_state(st.session_state)
            st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Remittance", "sources": []})
            st.rerun()

        elif is_financial_intent(ui_q):
            first_msg = reset_financial_state(st.session_state)
            st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Financial Planning", "sources": []})
            st.rerun()

        elif is_scam_intent(ui_q):  
            first_msg = reset_scam_state(st.session_state)
            st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Scam Safety", "sources": []})
            st.rerun()

        # No flow matched → plain Q&A with loader
        else:
            typing_html = "<span class='dot-flashing' aria-label='typing'></span> Thinking…"
            st.session_state.messages.append({"role": "assistant", "content": typing_html, "meta": "Thinking…", "sources": []})
            st.session_state.pending_query = ui_q
            st.session_state.run_backend = True
            st.rerun()

    # 3) A flow IS active → schedule the flow step with a loader
    else:
        typing_html = "<span class='dot-flashing' aria-label='typing'></span> Thinking…"
        st.session_state.messages.append({"role": "assistant", "content": typing_html, "meta": "Thinking…", "sources": []})
        st.session_state.pending_flow = st.session_state.flow
        st.session_state.flow_pending_input = ui_q
        st.session_state.run_backend = True
        st.rerun()



if submitted and user_input.strip():
    ui_q = user_input.strip()
    # 1) Append user message
    st.session_state.messages.append({"role": "user", "content": ui_q, "meta": "", "sources": []})

    # 2) If no active flow, detect & start one; else route to active flow
    if not st.session_state.flow:
        if is_remittance_intent(ui_q):
            # start remittance flow
            first_msg = reset_remittance_state(st.session_state)
            st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Remittance", "sources": []})
            st.rerun()

        elif is_financial_intent(ui_q):
            # start financial flow
            first_msg = reset_financial_state(st.session_state)
            st.session_state.messages.append({"role": "assistant", "content": first_msg, "meta": "Flow: Financial Planning", "sources": []})
            st.rerun()

        else:
            # no flow detected => fallback to your normal backend Q&A
            from rag_backend import answer_query
            res = answer_query(ui_q)
            ans = (res.get("answer") or "").strip()
            used_rag = bool(res.get("used_rag"))
            sources = res.get("sources") or []
            meta = "Answer grounded in uploaded context ✅" if used_rag else "General knowledge fallback ⚠️"
            st.session_state.messages.append({"role": "assistant", "content": ans, "meta": meta, "sources": sources})
            st.rerun()

    else:
        # Active flow
        if st.session_state.flow == "remittance":
            out = handle_remittance_turn(ui_q, st.session_state)
        elif st.session_state.flow == "financial":
            out = handle_financial_turn(ui_q, st.session_state)
        else:
            out = {"text": "Okay, ending current flow. Ask me anything else.", "used_backend": False, "used_rag": False, "sources": [], "done": True}
            st.session_state.flow = None

        meta = "Flow: Guidance" if not out.get("used_backend") else (
            "Answer grounded in uploaded context ✅" if out.get("used_rag") else "General knowledge fallback ⚠️"
        )
        st.session_state.messages.append({"role": "assistant", "content": out.get("text",""), "meta": meta, "sources": out.get("sources", [])})

        # If flow done, clear it
        if out.get("done"):
            st.session_state.flow = None
            st.session_state.flow_state = None

        st.rerun()


