# app.py
import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from rag_backend import answer_query  # backend function

load_dotenv()
st.set_page_config(page_title="SG RAG Chat", page_icon="🇸🇬", layout="wide")

# push content down so the toolbar doesn't overlap
st.markdown("""
<style>
  .block-container {max-width: 1400px; padding-top: 4.5rem; padding-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# No sidebar; session messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta, sources}]

# Header + clear
c1, c2 = st.columns([6, 1])
with c1:
    st.markdown("### Singapore Integration Assistant (RAG-first)")
with c2:
    if st.button("Clear", help="Clear this conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

# ---------- conversation renderer ----------
def render_messages_html(messages):
    html_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        meta = m.get("meta", "")
        sources = m.get("sources", [])
        klass = "user" if role == "user" else "bot"
        bubble = f'<div class="bubble {klass}">{content}</div>'
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
components.html(final_html, height=720, scrolling=True)

# ---------- INPUT AT THE BOTTOM (Form = Enter submits) ----------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your question (press Enter to send)",
        key="chatbox",
        placeholder="e.g., What documents are required for an S Pass application?"
    )
    submitted = st.form_submit_button("Send", type="primary")

if submitted and user_input.strip():
    ui_q = user_input.strip()

    # 1) append user
    st.session_state.messages.append({"role": "user", "content": ui_q, "meta": "", "sources": []})

    # 2) backend answer
    res = answer_query(ui_q)
    ans = (res.get("answer") or "").strip()
    used_rag = bool(res.get("used_rag"))
    sources = res.get("sources") or []
    meta = "Answer grounded in uploaded context ✅" if used_rag else "General knowledge fallback ⚠️"

    # 3) append assistant
    st.session_state.messages.append({"role": "assistant", "content": ans, "meta": meta, "sources": sources})

    # 4) refresh so the top conversation updates immediately
    st.rerun()

