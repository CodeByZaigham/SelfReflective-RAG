"""
Streamlit front-end for the Self-Reflective RAG graph in ../Graph.

This file only ADDS a UI layer. It does not modify any file under Graph/ or
Graph/RAG_pipelines/ - it imports and runs your existing `workflow` exactly
as graph.py does, and turns the print() statements you already have into a
live, visual pipeline trace (a highlighted flow diagram + a step log).

Run it with:
    streamlit run UI/app.py

Run that command from the "temp AI" project root so relative paths (like the
./chromadb folder your embeddings.py creates) resolve the same way they do
today when you run `python Graph/graph.py`.
"""

import os
import sys
import shutil
import tempfile
import traceback

import streamlit as st

# --------------------------------------------------------------------------
# Make your existing Graph/ package importable exactly the way its own files
# expect (nodes.py does `from state import state`, `from LLM import get_llm`,
# etc. - flat imports, not package-relative ones), by putting Graph/ itself
# on sys.path. Nothing inside Graph/ is touched.
# --------------------------------------------------------------------------
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(UI_DIR)
GRAPH_DIR = os.path.join(PROJECT_ROOT, "Graph")

if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)

st.set_page_config(
    page_title="Self-Reflective RAG",
    page_icon="🧠",
    layout="wide",
)

with st.spinner("Loading your pipeline (first run also downloads the embedding model)…"):
    try:
        from edges import workflow  # noqa: E402  (your compiled LangGraph workflow)
    except Exception:
        st.error(
            "Couldn't import your Graph pipeline (`from edges import workflow`). "
            "Check that all dependencies for your Graph/ code are installed in "
            "this environment, and that GRAPH_DIR below actually points at your "
            "Graph folder."
        )
        st.code(f"GRAPH_DIR = {GRAPH_DIR}")
        with st.expander("Full error"):
            st.code(traceback.format_exc())
        st.stop()

if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY isn't set in this environment - LLM.py's calls will fail. "
        "Add it to your .env file (loaded via python-dotenv in LLM.py).",
        icon="⚠️",
    )

# --------------------------------------------------------------------------
# Static description of your graph (mirrors Graph/edges.py) - used only to
# draw the pipeline diagram and to label each node nicely. Purely cosmetic;
# it does not drive execution, `workflow` still does that.
# --------------------------------------------------------------------------
NODE_META = {
    "decide_retrieval": ("🧭", "Decide if retrieval is needed"),
    "load_document": ("📄", "Load document"),
    "vector_database": ("🧱", "Build vector database"),
    "retriever": ("🔍", "Retrieve candidate chunks"),
    "docs_relevence": ("🧪", "Filter for relevance"),
    "response_generator_from_context": ("✍️", "Generate answer from context"),
    "generate_direct": ("💬", "Generate direct answer"),
    "is_answer_supported": ("🛡️", "Check factual grounding"),
    "revise_answer": ("🔁", "Revise answer"),
    "is_answer_usable": ("✅", "Check if answer is usable"),
    "rewrite_query": ("✏️", "Rewrite query"),
}

DOT_LABELS = {
    "decide_retrieval": "Decide\\nRetrieval",
    "load_document": "Load\\nDocument",
    "vector_database": "Build\\nVector DB",
    "retriever": "Retrieve\\nChunks",
    "docs_relevence": "Filter\\nRelevance",
    "response_generator_from_context": "Generate\\n(from context)",
    "generate_direct": "Generate\\n(direct)",
    "is_answer_supported": "Check\\nGrounding",
    "revise_answer": "Revise\\nAnswer",
    "is_answer_usable": "Check\\nUsability",
    "rewrite_query": "Rewrite\\nQuery",
}

DOT_EDGES = [
    ("START", "decide_retrieval", None),
    ("decide_retrieval", "retriever", "needed, indexed"),
    ("decide_retrieval", "load_document", "needed, new doc"),
    ("decide_retrieval", "generate_direct", "not needed"),
    ("load_document", "vector_database", None),
    ("vector_database", "retriever", None),
    ("retriever", "docs_relevence", None),
    ("docs_relevence", "response_generator_from_context", "relevant"),
    ("docs_relevence", "END", "not relevant"),
    ("generate_direct", "END", None),
    ("response_generator_from_context", "is_answer_supported", None),
    ("is_answer_supported", "is_answer_usable", "supported"),
    ("is_answer_supported", "revise_answer", "not supported"),
    ("revise_answer", "is_answer_supported", None),
    ("is_answer_usable", "END", "usable"),
    ("is_answer_usable", "rewrite_query", "not usable"),
    ("rewrite_query", "retriever", None),
]

DONE_COLOR = "#2e7d32"
ACTIVE_COLOR = "#f5a623"
PENDING_COLOR = "#e6e6e6"
PENDING_FONT = "#666666"


def build_dot(current: str | None, visited: list[str]) -> str:
    """Render the (static) graph structure, highlighting progress so far."""
    visited_set = set(visited)

    def fill(node: str) -> str:
        if node == current:
            return ACTIVE_COLOR
        if node in visited_set:
            return DONE_COLOR
        return PENDING_COLOR

    def font(node: str) -> str:
        return "white" if (node == current or node in visited_set) else PENDING_FONT

    lines = [
        "digraph G {",
        "rankdir=LR;",
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.18,0.10"];',
        'edge [fontname="Helvetica", fontsize=9, color="#9e9e9e", fontcolor="#777777"];',
        'START [shape=circle, label="", width=0.18, style=filled, fillcolor="#333333"];',
        'END [shape=doublecircle, label="", width=0.18, style=filled, fillcolor="#333333"];',
    ]
    for node, label in DOT_LABELS.items():
        lines.append(f'{node} [label="{label}", fillcolor="{fill(node)}", fontcolor="{font(node)}"];')
    for a, b, label in DOT_EDGES:
        attr = f' [label="{label}"]' if label else ""
        lines.append(f"{a} -> {b}{attr};")
    lines.append("}")
    return "\n".join(lines)


def describe_output(node_name: str, output: dict) -> str:
    """Short, human-readable summary of what a node just produced."""
    if not isinstance(output, dict):
        return ""
    try:
        if node_name == "decide_retrieval":
            return f"→ `{str(output.get('retrieval_needed', '')).strip()}`"
        if node_name == "load_document":
            n = len(output.get("chunks") or [])
            return f"split document into {n} chunk(s)"
        if node_name == "vector_database":
            return "vector index built"
        if node_name == "retriever":
            n = len(output.get("retrieved_docs") or [])
            return f"retrieved {n} candidate chunk(s)"
        if node_name == "docs_relevence":
            if "relevent_docs" in output:
                return f"kept {len(output['relevent_docs'])} relevant chunk(s)"
            return "no relevant chunks found"
        if node_name in ("response_generator_from_context", "generate_direct", "revise_answer"):
            text = str(output.get("response", "")).strip().replace("\n", " ")
            return (text[:90] + "…") if len(text) > 90 else text
        if node_name == "is_answer_supported":
            return f"→ `{str(output.get('is_answer_supported', '')).strip()}`"
        if node_name == "is_answer_usable":
            return f"→ `{str(output.get('is_answer_usable', '')).strip()}`"
        if node_name == "rewrite_query":
            return f"new query: “{str(output.get('query', '')).strip()}”"
    except Exception:
        return ""
    return ""


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_dir = os.path.join(tempfile.gettempdir(), "self_rag_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    dest = os.path.join(tmp_dir, uploaded_file.name if suffix else uploaded_file.name + ".txt")
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def run_pipeline(query: str, doc_path: str):
    """Generator yielding (node_name, node_output, state_so_far) per graph step."""
    initial_state = {
        "doc_path": doc_path or "",
        "query": query,
        "revise_attempts": 0,
        "retrieve_attempts": 0,
    }
    full_state = dict(initial_state)
    for update in workflow.stream(initial_state, stream_mode="updates"):
        for node_name, output in update.items():
            if isinstance(output, dict):
                full_state.update(output)
            yield node_name, output, dict(full_state)


# --------------------------------------------------------------------------
# Sidebar - knowledge base management
# --------------------------------------------------------------------------
st.session_state.setdefault("doc_path", "")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("📚 Knowledge base")

    uploaded = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "csv", "ppt", "pptx"]
    )
    if uploaded is not None:
        path = save_uploaded_file(uploaded)
        st.session_state.doc_path = path
        st.success(f"Ready: {uploaded.name}")

    with st.expander("Or set a document path manually"):
        manual_path = st.text_input("Document path", value=st.session_state.doc_path)
        if manual_path != st.session_state.doc_path:
            st.session_state.doc_path = manual_path

    db_exists = os.path.exists("chromadb")
    st.markdown("---")
    st.caption(
        ("🟢 Vector index found at `./chromadb`" if db_exists else "⚪ No vector index yet")
        + " (path is relative to wherever you launched `streamlit run` from)."
    )
    if db_exists:
        st.caption(
            "Your `decide_retrieval` routing reuses this index whenever one exists, "
            "even for a newly uploaded document. Clear it below to force re-indexing."
        )
    if st.button("🗑️ Clear vector database", disabled=not db_exists, use_container_width=True):
        shutil.rmtree("chromadb", ignore_errors=True)
        st.success("Cleared - the next query needing retrieval will rebuild it.")
        st.rerun()

    st.markdown("---")
    if st.button("🧹 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🧠 Self-Reflective RAG")
st.caption(
    "Ask a question. The agent decides whether it needs to retrieve from your "
    "documents, checks its own answer for grounding and usefulness, and revises "
    "itself when it isn't good enough - live below."
)

with st.expander("📊 How this pipeline works"):
    st.graphviz_chart(build_dot(None, []), use_container_width=True)
    st.caption(
        "Solid path: retrieve → filter → generate → check grounding → check usability. "
        "Dashed loops back to *Revise Answer* or *Rewrite Query* happen when a check fails, "
        "capped by your `revise_attempts` / `retrieve_attempts` counters."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander(f"🔍 Pipeline trace · {len(msg['trace'])} step(s)", expanded=False):
                st.graphviz_chart(build_dot(None, msg["trace"]), use_container_width=True)
                for line in msg.get("trace_log", []):
                    st.markdown(line)
            if msg.get("badges"):
                st.caption(" · ".join(msg["badges"]))
        st.markdown(msg["content"])

query = st.chat_input("Ask something about your document, or anything else…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status = st.status("Running pipeline…", expanded=True)
        diagram_slot = status.empty()
        log_slot = status.empty()
        diagram_slot.graphviz_chart(build_dot(None, []), use_container_width=True)

        visited: list[str] = []
        log_lines: list[str] = []
        full_state: dict = {}
        error_text = None

        try:
            for node_name, output, snapshot in run_pipeline(query, st.session_state.doc_path):
                visited.append(node_name)
                full_state = snapshot
                icon, label = NODE_META.get(node_name, ("⚙️", node_name))
                detail = describe_output(node_name, output)
                log_lines.append(f"{icon} **{label}**" + (f" — {detail}" if detail else ""))
                diagram_slot.graphviz_chart(build_dot(node_name, visited), use_container_width=True)
                log_slot.markdown("\n\n".join(log_lines))
        except Exception:
            error_text = traceback.format_exc()

        if error_text:
            status.update(label="Pipeline stopped early", state="error", expanded=True)
            st.error(
                "The pipeline raised an error partway through (see the trace above for "
                "how far it got)."
            )
            with st.expander("Error details"):
                st.code(error_text)
        else:
            status.update(label=f"Pipeline finished · {len(visited)} step(s)", state="complete", expanded=False)

        final_response = full_state.get("response") or (
            "The pipeline didn't produce a final answer for this query."
            if not error_text
            else "The pipeline stopped before producing an answer."
        )

        badges = []
        if "retrieval_needed" in full_state:
            badges.append(f"Retrieval: {str(full_state['retrieval_needed']).strip()}")
        if "is_answer_supported" in full_state:
            badges.append(f"Grounded: {str(full_state['is_answer_supported']).strip()}")
        if "is_answer_usable" in full_state:
            badges.append(f"Usable: {str(full_state['is_answer_usable']).strip()}")

        st.markdown(final_response)
        if badges:
            st.caption(" · ".join(badges))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_response,
                "trace": visited,
                "trace_log": log_lines,
                "badges": badges,
            }
        )