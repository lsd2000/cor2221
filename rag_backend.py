# rag_backend.py
import os, json, requests
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# llama-cloud-services
from llama_cloud_services import LlamaCloudIndex

# Make langdetect deterministic
DetectorFactory.seed = 0
load_dotenv()

# === Config ===
LC_SDK_API_KEY   = os.getenv("LLAMACLOUD_API_KEY") or os.getenv("LLAMA_CLOUD_SERVICES_API_KEY")
LC_INDEX_NAME    = os.getenv("LLAMACLOUD_INDEX_NAME", "COR2221")
LC_PROJECT_NAME  = os.getenv("LLAMACLOUD_PROJECT_NAME", "Default")
LC_ORG_ID        = os.getenv("LLAMACLOUD_ORG_ID")
TOP_K            = int(os.getenv("TOP_K", "4"))

SEA_LION_API_KEY = os.getenv("SEA_LION_API_KEY")
SEA_LION_BASE    = os.getenv("SEA_LION_BASE", "https://api.sea-lion.ai")
SEA_LION_MODEL   = os.getenv("SEA_LION_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")
DEFAULT_TIMEOUT  = 30

if not LC_SDK_API_KEY or not LC_ORG_ID:
    raise RuntimeError("Missing LlamaCloud credentials (.env).")
if not SEA_LION_API_KEY:
    raise RuntimeError("Missing SEA_LION_API_KEY (.env).")

SUPPORTED_LANGS = {
    "en": "English", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "zh": "Chinese", "hi": "Hindi", "ta": "Tamil", "ms": "Malay"
}

NOT_FOUND_TOKEN = "<<NOT_FOUND>>"
FORCE_EN_QUERY = False

# Cache the index
_LC_INDEX = None

def detect_lang(text: str) -> str:
    try:
        code = detect(text)
    except Exception:
        return "en"
    return "zh" if code.startswith("zh") else code

def _sealion_chat(messages, temperature=0.2, max_tokens=1024) -> str:
    url = f"{SEA_LION_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {SEA_LION_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": SEA_LION_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "max_completion_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

def translate_with_sealion(text: str, target_lang_code: str) -> str:
    lang_name = SUPPORTED_LANGS.get(target_lang_code, target_lang_code)
    sys_prompt = (
        f"You are a precise translator into {lang_name}. "
        "Output ONLY the translation. No preface, no quotes, no notes."
    )
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
    return _sealion_chat(messages, temperature=0.0, max_tokens=1024)

def _get_index() -> LlamaCloudIndex:
    global _LC_INDEX
    if _LC_INDEX is None:
        _LC_INDEX = LlamaCloudIndex(
            name=LC_INDEX_NAME,
            project_name=LC_PROJECT_NAME,
            organization_id=LC_ORG_ID,
            api_key=LC_SDK_API_KEY,
        )
    return _LC_INDEX

def retrieve_context(query: str, top_k: int = TOP_K):
    try:
        nodes = _get_index().as_retriever().retrieve(query)
    except Exception as e:
        print(f"[retrieve warn] {e}")
        return [], []
    nodes = list(nodes)[:int(top_k)]
    ctx, srcs = [], []
    for n in nodes:
        text = getattr(n, "text", None) or (getattr(n, "node", {}) or {})
        if isinstance(text, dict):
            text = text.get("text", "")
        text = (text or "")
        md = getattr(n, "metadata", None)
        if md is None and hasattr(n, "node") and isinstance(n.node, dict):
            md = n.node.get("metadata")
        src = "unknown"
        if isinstance(md, dict):
            src = md.get("file_name") or md.get("source") or md.get("document_id") or "unknown"
        if text.strip():
            ctx.append(text.strip()); srcs.append(str(src))
    return ctx, srcs

def clamp_context(chunks, max_chars=48000):
    out, total = [], 0
    for c in chunks:
        c = (c or "").strip()
        if not c: continue
        if total + len(c) > max_chars:
            rest = max_chars - total
            if rest > 0: out.append(c[:rest] + " ...[truncated]...")
            break
        out.append(c); total += len(c)
    return "\n\n---\n\n".join(out)

def filter_context(chunks, include_any=()):
    if not include_any: return chunks
    keys = [k.lower() for k in include_any]
    sel = []
    for c in chunks:
        low = (c or "").lower()
        if any(k in low for k in keys): sel.append(c)
    return sel or chunks

def make_rag_prompt_strict(context_text: str, answer_lang_code: str):
    lang_name = SUPPORTED_LANGS.get(answer_lang_code, "the user's language")
    return (
        "You are helping immigrants in Singapore. Follow these STRICT rules:\n"
        "1) Use ONLY the provided CONTEXT below for all facts. If a detail is not in CONTEXT, do not infer it.\n"
        "2) If the user's request is outside the scope of the CONTEXT, or the CONTEXT is insufficient to answer exactly, "
        f"respond EXACTLY with: {NOT_FOUND_TOKEN} (no extra words).\n"
        "3) For EP/S Pass topics, answer only what the document actually covers (e.g., supporting documents) unless steps are explicitly present.\n"
        '4) If you do answer, include at least one short quote from CONTEXT in double quotes to show grounding.\n'
        "5) Be concise and precise. Cite filenames inline like [filename]. No invented links or numbers.\n\n"
        f"Answer in {lang_name}.\n\n"
        f"CONTEXT:\n{context_text}\n"
    )


def make_general_prompt(answer_lang_code: str):
    lang = SUPPORTED_LANGS.get(answer_lang_code, "the user's language")
    return (
        "You are a helpful, concise assistant. "
        f"Answer in {lang}. "
        "For Singapore immigration/work passes, you may give general guidance and suggest checking ICA/MOM. "
        "Avoid hallucinating links or specific numbers if unsure."
    )

def answer_query(
    user_raw: str,
    require_keywords: tuple[str, ...] = (),
    force_general: bool = False
) -> dict:
    """
    Returns a dict:
    {
      "answer": str,
      "used_rag": bool,
      "sources": [str],
      "fallback_used": bool
    }
    """
    user_lang = detect_lang(user_raw)
    query_for_retrieval = user_raw
    if FORCE_EN_QUERY and user_lang != "en":
        try:
            query_for_retrieval = translate_with_sealion(user_raw, "en")
        except Exception as e:
            print(f"[translate warn] {e}; using original text.")

    # 1) Retrieve
    ctx_chunks, srcs = retrieve_context(query_for_retrieval, TOP_K)

    # 2) OPTIONAL GATE: caller can require certain keywords to appear in retrieved chunks
    #    - If require_keywords is provided and none of the chunks contain them -> disable RAG (force fallback)
    #    - If force_general is True -> disable RAG regardless of retrieval
    if force_general:
        ctx_chunks = []
    elif require_keywords:
        # filter_context keeps only chunks containing ANY of the keywords; returns original if none match,
        # so we must manually empty when there's no match to force fallback.
        filtered = filter_context(ctx_chunks, include_any=tuple(k.lower() for k in require_keywords))
        # Detect "no match" by checking if filtered == original but none of the keywords are in any chunk.
        if filtered is ctx_chunks:
            # verify no keyword present at all
            any_hit = any(
                any(k.lower() in (c or "").lower() for k in require_keywords)
                for c in ctx_chunks
            )
            if not any_hit:
                ctx_chunks = []  # nothing relevant -> skip RAG
        else:
            ctx_chunks = filtered

    # 3) EP/S Pass focusing (only for the pass domain; avoid biasing other domains like scams)
    if not force_general and not require_keywords:
        focus_terms, ur_low = [], user_raw.lower()
        if "s pass" in ur_low or "spass" in ur_low or " s-pass" in ur_low:
            focus_terms += ["s pass", "employment pass", "ep"]
        if "employment pass" in ur_low or (" ep " in f" {ur_low} "):
            focus_terms += ["employment pass", "ep", "s pass"]
        if focus_terms:
            ctx_chunks = filter_context(ctx_chunks, include_any=tuple(set(focus_terms)))

    context = clamp_context(ctx_chunks, max_chars=48000)

    # 4) STRICT RAG pass (only if we still have context after the gate)
    used_rag = False
    if ctx_chunks:
        sys_prompt = make_rag_prompt_strict(context, user_lang)
        rag_msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_raw}]
        rag_ans = _sealion_chat(rag_msgs, temperature=0.0, max_tokens=1024)
        if rag_ans:
            text = rag_ans.strip()

            # Treat common "refusal/explanation" patterns as NOT_FOUND, so we fall back.
            refusal_patterns = (
                "does not contain", "not contain information",
                "outside the scope", "not present in the context",
                "context focuses on", "cannot find", "insufficient",
                "<<>>", "<>",  # any placeholder noise
            )
            looks_like_refusal = any(pat in text.lower() for pat in refusal_patterns)

            if text != NOT_FOUND_TOKEN and not looks_like_refusal:
                used_rag = True
                return {"answer": text, "used_rag": True, "sources": srcs, "fallback_used": False}

    # 5) Fallback (general knowledge)
    general_sys = make_general_prompt(user_lang)
    gen_msgs = [{"role": "system", "content": general_sys}, {"role": "user", "content": user_raw}]
    general_reply = _sealion_chat(gen_msgs, temperature=0.2, max_tokens=1024)
    fallback_notice = ""
    if ctx_chunks:  # we had context but chose not to use it (or it wasn't sufficient)
        fallback_notice = (
            "\n⚠️ *Fallback Notice:*\n"
            "The uploaded context did not contain enough information to fully answer your question.\n"
            "Here’s a **general overview** based on public knowledge instead:\n\n"
        )
    return {
        "answer": fallback_notice + general_reply,
        "used_rag": False,
        "sources": srcs,
        "fallback_used": True
    }



    # Fallback
    general_sys = make_general_prompt(user_lang)
    gen_msgs = [{"role": "system", "content": general_sys}, {"role": "user", "content": user_raw}]
    general_reply = _sealion_chat(gen_msgs, temperature=0.2, max_tokens=1024)
    fallback_notice = ""
    if ctx_chunks:
        fallback_notice = (
            "\n⚠️ *Fallback Notice:*\n"
            "The uploaded context did not contain enough information to fully answer your question.\n"
            "Here’s a **general overview** based on public knowledge instead:\n\n"
        )
    return {
        "answer": fallback_notice + general_reply,
        "used_rag": False,
        "sources": srcs,
        "fallback_used": True
    }
