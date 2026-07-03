import streamlit as st
import requests
import uuid
import os
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🚀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f0f1a; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 15px;
        padding: 12px 20px; border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460; border-radius: 14px;
        padding: 20px; text-align: center;
    }
    .confidence-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 12px; font-weight: 600;
    }
    .badge-high { background: #1a4731; color: #4ade80; }
    .badge-medium { background: #3a3000; color: #facc15; }
    .badge-low { background: #4a1515; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.4rem; font-weight:700; background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🚀 Enterprise RAG System</h1>
    <p style='color:#94a3b8; font-size:1rem;'>Graph RAG · Hybrid Retrieval · Cross-Encoder Reranking · Ragas Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Upload ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDFs for Hybrid + Graph Indexing",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📥 Process & Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Upload files first.")
        else:
            with st.spinner("Running OCR, Chunking, Entity Extraction, Embedding…"):
                files_to_upload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                try:
                    response = requests.post(f"{API_URL}/upload", files=files_to_upload, timeout=300)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Processed **{data.get('chunks_processed', 0)}** chunks!")
                    else:
                        st.error(f"Backend error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")

    st.divider()
    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}…`")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_graph, tab_eval = st.tabs(["💬 Chat & QA", "🕸️ Graph Explorer", "📊 Evaluation Dashboard"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("confidence") is not None and msg["role"] == "assistant":
                    conf = msg["confidence"]
                    if conf >= 0.7:
                        badge_class, label = "badge-high", "High confidence"
                    elif conf >= 0.4:
                        badge_class, label = "badge-medium", "Medium confidence"
                    else:
                        badge_class, label = "badge-low", "Low confidence"
                    st.markdown(
                        f'<span class="confidence-badge {badge_class}">'
                        f'🎯 {label}: {conf:.0%}</span>',
                        unsafe_allow_html=True,
                    )
                    if msg.get("hallucination"):
                        st.warning("⚠️ Possible hallucination detected by self-reflection.")
                    if msg.get("sources"):
                        with st.expander("📎 Sources"):
                            for s in set(msg["sources"]):
                                st.markdown(f"- {s}")

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Hybrid retrieval → Reranking → Generating…"):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    confidence = data["confidence_score"]
                    is_hallucination = data["hallucination_flag"]
                    sources = data["sources"]

                    with st.chat_message("assistant"):
                        st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "confidence": confidence,
                        "hallucination": is_hallucination,
                        "sources": sources,
                    })
                    st.rerun()
                else:
                    st.error(f"Error from API: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Graph Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_graph:
    st.markdown("### 🕸️ Knowledge Graph Explorer")
    st.caption("Entities and relationships extracted from uploaded documents via Gemini LLM → Neo4j.")

    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh = st.button("🔄 Load Graph", use_container_width=True, type="primary")

    if refresh:
        with st.spinner("Fetching graph data from Neo4j…"):
            try:
                res = requests.get(f"{API_URL}/graph", timeout=30)
                if res.status_code == 200:
                    graph_data = res.json()
                    nodes = graph_data.get("nodes", [])
                    edges = graph_data.get("edges", [])
                    err = graph_data.get("error")

                    if err:
                        st.error(f"Neo4j error: {err}")
                    elif not nodes:
                        st.info("No graph data yet. Upload and index a document first.")
                    else:
                        # Build networkx graph for layout
                        G = nx.DiGraph()
                        for n in nodes:
                            G.add_node(n["id"], label=n["label"])
                        for e in edges:
                            G.add_edge(e["source"], e["target"], type=e["type"])

                        pos = nx.spring_layout(G, seed=42, k=2.0)

                        # Build Plotly figure
                        edge_x, edge_y = [], []
                        for u, v in G.edges():
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color="#475569"),
                            hoverinfo="none", mode="lines",
                        )

                        node_x = [pos[n][0] for n in G.nodes()]
                        node_y = [pos[n][1] for n in G.nodes()]
                        node_text = list(G.nodes())

                        # Color by label type
                        label_map = nx.get_node_attributes(G, "label")
                        unique_labels = list(set(label_map.values()))
                        palette = ["#6366f1", "#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#ef4444"]
                        color_map = {l: palette[i % len(palette)] for i, l in enumerate(unique_labels)}
                        node_colors = [color_map.get(label_map.get(n, ""), "#6366f1") for n in G.nodes()]

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode="markers+text",
                            hoverinfo="text",
                            text=node_text,
                            textposition="top center",
                            textfont=dict(size=10, color="#e2e8f0"),
                            marker=dict(size=14, color=node_colors, line=dict(width=1, color="#1e293b")),
                        )

                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode="closest",
                                paper_bgcolor="#0f0f1a",
                                plot_bgcolor="#0f0f1a",
                                margin=dict(b=20, l=5, r=5, t=20),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600,
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            f"**{len(nodes)}** entities · **{len(edges)}** relationships  |  "
                            "Also explore at [Neo4j Browser](http://localhost:7474)"
                        )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Evaluation Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 📊 Ragas Evaluation Dashboard")
    st.caption("Evaluates Faithfulness, Answer Relevancy, Context Precision, and Context Recall using Gemini LLM.")

    st.info("Ask several questions in the Chat tab first, then run evaluation to see metrics.", icon="ℹ️")

    if st.button("▶️ Run Ragas Evaluation", use_container_width=True, type="primary"):
        with st.spinner("Running Ragas evaluation with Gemini LLM…"):
            try:
                res = requests.get(f"{API_URL}/evaluate", timeout=300)
                if res.status_code == 200:
                    payload = res.json()
                    metrics = payload.get("metrics", {})

                    if "error" in metrics:
                        st.error(f"Evaluation error: {metrics['error']}")
                    elif "status" in metrics:
                        st.warning(metrics["status"])
                    else:
                        st.success("✅ Evaluation complete!")

                        # Metric cards row
                        metric_keys = {
                            "faithfulness": "🔒 Faithfulness",
                            "answer_relevancy": "🎯 Answer Relevancy",
                            "context_precision": "📐 Context Precision",
                            "context_recall": "🔁 Context Recall",
                        }
                        cols = st.columns(len(metric_keys))
                        for col, (key, label) in zip(cols, metric_keys.items()):
                            val = metrics.get(key, None)
                            if val is not None:
                                color = "#4ade80" if val >= 0.7 else "#facc15" if val >= 0.4 else "#f87171"
                                col.markdown(
                                    f"<div class='metric-card'>"
                                    f"<div style='font-size:13px;color:#94a3b8;margin-bottom:6px;'>{label}</div>"
                                    f"<div style='font-size:2rem;font-weight:700;color:{color};'>{val:.2%}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        # Bar chart
                        st.divider()
                        chart_data = {metric_keys.get(k, k): v for k, v in metrics.items() if isinstance(v, float)}
                        if chart_data:
                            df_chart = pd.DataFrame(
                                {"Metric": list(chart_data.keys()), "Score": list(chart_data.values())}
                            )
                            fig = go.Figure(
                                go.Bar(
                                    x=df_chart["Metric"],
                                    y=df_chart["Score"],
                                    marker_color=["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"],
                                    text=[f"{v:.1%}" for v in df_chart["Score"]],
                                    textposition="outside",
                                )
                            )
                            fig.update_layout(
                                paper_bgcolor="#0f0f1a",
                                plot_bgcolor="#0f0f1a",
                                font=dict(color="#e2e8f0"),
                                yaxis=dict(range=[0, 1.1], gridcolor="#1e293b"),
                                xaxis=dict(gridcolor="#1e293b"),
                                height=380,
                                margin=dict(t=30),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"API error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")

