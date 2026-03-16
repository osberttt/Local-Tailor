"""
localtailor/app.py
==================
Streamlit UI — run with: streamlit run localtailor/app.py

Three views:
  1. Dimension Board  — comments grouped by dimension value, with spans shown
  2. Intent Queue     — comments sorted by action priority
  3. Analytics        — charts, coverage, value distribution, accuracy metrics

Reads from:
  data/comments_clean_demo.json
  data/predictions_demo.json
  data/evaluation_demo.json  (optional — shows accuracy panel if present)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on sys.path (Streamlit runs from project root but
# imports need the localtailor package to be discoverable)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import streamlit as st
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

POST_ID = "demo"
COMMENTS_PATH = Path(f"data/comments_clean_{POST_ID}.json")
PREDICTIONS_PATH = Path(f"data/predictions_{POST_ID}.json")
EVAL_PATH = Path(f"data/evaluation_{POST_ID}.json")
DIMENSIONS_YAML = Path("config/dimensions.yaml")
EXAMPLES_JSON = Path("data/examples.json")

INTENT_PRIORITY = [
    "needs reply", "negative review", "Unclear",
    "comparison", "monitor", "positive review", "spam",
]

FLAG_COLORS = {
    "classified": "🟢",
    "unclear":    "🟡",
    "na":         "⚪",
}

st.set_page_config(
    page_title="Local Tailor",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.comment-card {
    background: #f8f9fa;
    border-left: 4px solid #2E75B6;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
}
.comment-card.unclear {
    border-left-color: #FFC107;
    background: #FFFDF0;
}
.comment-card.na {
    border-left-color: #cccccc;
    background: #f8f8f8;
}
.span-highlight {
    background: #D5E8F0;
    border-radius: 3px;
    padding: 1px 4px;
    font-weight: 600;
}
.metric-card {
    background: #1F3864;
    color: white;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}
.dim-header {
    font-size: 16px;
    font-weight: 700;
    color: #1F3864;
    border-bottom: 2px solid #2E75B6;
    padding-bottom: 4px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not COMMENTS_PATH.exists():
        return None, None, None

    with open(COMMENTS_PATH) as f:
        comments_data = json.load(f)
    comments = {c["id"]: c for c in comments_data["comments"]}

    predictions = {}
    if PREDICTIONS_PATH.exists():
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)

    evaluation = None
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            evaluation = json.load(f)

    return comments, predictions, evaluation


def get_dimensions(predictions):
    if not predictions:
        return []
    sample = next(iter(predictions.values()))
    return list(sample.keys())


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(comments, predictions, evaluation):
    with st.sidebar:
        st.markdown("## 🧵 Local Tailor")
        st.markdown("*Comment intelligence for your business*")
        st.divider()

        view = st.radio(
            "View",
            [
                "📋 Dimension Board",
                "📬 Intent Queue",
                "📊 Analytics",
                "📤 Export",
                "⚙️ Config",
                # TODO: "🔑 Login" — Facebook login / access token UI
                # TODO: "📡 Facebook" — connect to post, fetch comments
            ],
            label_visibility="collapsed",
        )

        st.divider()

        if comments and predictions:
            dims = get_dimensions(predictions)
            st.markdown("**Dataset**")
            st.markdown(f"- {len(comments)} comments")
            st.markdown(f"- {len(dims)} dimensions")

            # Classified / unclear / na counts
            st.markdown("**Coverage**")
            for dim in dims:
                classified = sum(
                    1 for p in predictions.values()
                    if p.get(dim, {}).get("flag") == "classified"
                )
                pct = int(classified / len(comments) * 100)
                st.progress(pct / 100, text=f"{dim}: {pct}%")

        if evaluation:
            st.divider()
            st.markdown("**Accuracy**")
            st.metric("Overall", evaluation.get("overall_accuracy_pct", "—"))

        st.divider()
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return view


# ── View 1: Dimension Board ───────────────────────────────────────────────────

def render_dimension_board(comments, predictions):
    st.title("📋 Dimension Board")
    dims = get_dimensions(predictions)

    selected_dim = st.selectbox(
        "Select dimension to explore",
        dims,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.divider()

    # Group comments by value
    groups = defaultdict(list)
    for cid, pred in predictions.items():
        entry = pred.get(selected_dim, {})
        value = entry.get("value", "N/A")
        groups[value].append((cid, entry))

    # Sort groups: Unclear first (needs attention), then values, then N/A last
    def sort_key(v):
        if v == "Unclear":
            return (0, v)
        if v == "N/A":
            return (2, v)
        return (1, v)

    sorted_values = sorted(groups.keys(), key=sort_key)

    for value in sorted_values:
        entries = groups[value]
        flag = entries[0][1].get("flag", "na") if entries else "na"
        icon = FLAG_COLORS.get(flag, "⚪")

        with st.expander(
            f"{icon} **{value}** — {len(entries)} comment{'s' if len(entries)!=1 else ''}",
            expanded=(value == "Unclear" or len(entries) > 0 and value != "N/A"),
        ):
            # Sort by confidence desc (most confident first)
            sorted_entries = sorted(entries, key=lambda x: x[1].get("score", 0), reverse=True)

            for cid, entry in sorted_entries:
                comment = comments.get(cid, {})
                message = comment.get("message", "—")
                span = entry.get("span")
                score = entry.get("score", 0)

                # Highlight span in message
                if span and span in message:
                    highlighted = message.replace(
                        span, f'<mark style="background:#FFD54F; padding:1px 3px; border-radius:3px;">{span}</mark>', 1
                    )
                else:
                    highlighted = message

                st.markdown(f"""
                <div style="margin-bottom:12px; font-size:14px; line-height:1.6;">
                  {highlighted}
                  <div style="margin-top:4px; font-size:12px; color:#888;">
                    confidence: {score:.2f}
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()


# ── View 2: Intent Queue ──────────────────────────────────────────────────────

def render_intent_queue(comments, predictions):
    st.title("📬 Intent Queue")
    st.caption("Comments sorted by action priority. Unanswered questions first.")

    if "intent" not in get_dimensions(predictions):
        st.warning("No 'intent' dimension in predictions.")
        return

    # Build queue
    queue = []
    for cid, pred in predictions.items():
        intent_entry = pred.get("intent", {})
        tone_entry = pred.get("tone", {})
        intent_val = intent_entry.get("value", "N/A")
        tone_val = tone_entry.get("value", "N/A")
        score = intent_entry.get("score", 0)
        flag = intent_entry.get("flag", "na")
        comment = comments.get(cid, {})

        if intent_val == "N/A":
            continue

        priority = INTENT_PRIORITY.index(intent_val) if intent_val in INTENT_PRIORITY else 99
        queue.append({
            "cid": cid,
            "message": comment.get("message", "—"),
            "intent": intent_val,
            "tone": tone_val,
            "score": score,
            "flag": flag,
            "priority": priority,
            "like_count": comment.get("like_count", 0),
        })

    # Sort: priority first, then like count desc
    queue.sort(key=lambda x: (x["priority"], -x["like_count"]))

    # Summary stats
    needs_reply = sum(1 for q in queue if q["intent"] == "needs reply")
    negative = sum(1 for q in queue if q["intent"] == "negative review")
    unclear = sum(1 for q in queue if q["flag"] == "unclear")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total in queue", len(queue))
    c2.metric("❓ Need reply", needs_reply)
    c3.metric("⚠️ Negative", negative)
    c4.metric("🟡 Unclear", unclear)

    st.divider()

    # Filter
    filter_intent = st.multiselect(
        "Filter by intent",
        options=list(Counter(q["intent"] for q in queue).keys()),
        default=[],
        placeholder="Show all",
    )
    if filter_intent:
        queue = [q for q in queue if q["intent"] in filter_intent]

    # Render cards
    for q in queue:
        flag_icon = FLAG_COLORS.get(q["flag"], "⚪")
        intent_color = {
            "needs reply": "#FF6B35",
            "negative review": "#E74C3C",
            "Unclear": "#FFC107",
            "comparison": "#9B59B6",
            "monitor": "#3498DB",
            "positive review": "#27AE60",
            "spam": "#95A5A6",
        }.get(q["intent"], "#666")

        st.markdown(f"""
        <div class="comment-card" style="border-left-color: {intent_color};">
          {q['message']}
          <div style="margin-top:6px; font-size:12px; color:#666;">
            {flag_icon}
            <strong style="color:{intent_color};">{q['intent']}</strong>
            &nbsp;·&nbsp; tone: {q['tone']}
            &nbsp;·&nbsp; confidence: {q['score']:.2f}
            {f"&nbsp;·&nbsp; 👍 {q['like_count']}" if q['like_count'] > 0 else ""}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── View 3: Analytics ─────────────────────────────────────────────────────────

def render_analytics(comments, predictions, evaluation):
    st.title("📊 Analytics")

    dims = get_dimensions(predictions)
    total = len(comments)

    # ── Coverage chart ────────────────────────────────────────────────────────
    st.subheader("Dimension Coverage")
    st.caption("What percentage of comments mention each dimension (not N/A)")

    coverage_data = {}
    for dim in dims:
        classified = sum(
            1 for p in predictions.values()
            if p.get(dim, {}).get("flag") in ("classified", "unclear")
        )
        coverage_data[dim] = round(classified / total * 100, 1)

    coverage_df = pd.DataFrame(
        {"Dimension": list(coverage_data.keys()),
         "Coverage (%)": list(coverage_data.values())}
    ).sort_values("Coverage (%)", ascending=True)

    st.bar_chart(coverage_df.set_index("Dimension"), horizontal=True)

    st.divider()

    # ── Per-dimension value distribution ─────────────────────────────────────────
    st.subheader("Value Distribution per Dimension")

    selected = st.selectbox(
        "Dimension",
        [d for d in dims if d not in ("intent", "tone")],
        format_func=lambda x: x.replace("_", " ").title(),
        key="analytics_dim",
    )

    value_counts = Counter()
    for pred in predictions.values():
        entry = pred.get(selected, {})
        if entry.get("flag") == "classified":
            value_counts[entry.get("value", "N/A")] += 1

    if value_counts:
        df = pd.DataFrame(
            {"Value": list(value_counts.keys()),
             "Count": list(value_counts.values())}
        ).sort_values("Count", ascending=False).set_index("Value")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df)
    else:
        st.info("No classified predictions for this dimension yet.")

    st.divider()

    # ── Intent summary ────────────────────────────────────────────────────────
    st.subheader("Intent Summary")
    intent_counts = Counter()
    for pred in predictions.values():
        entry = pred.get("intent", {})
        if entry.get("flag") == "classified":
            intent_counts[entry.get("value", "unknown")] += 1

    if intent_counts:
        intent_df = pd.DataFrame(
            {"Intent": list(intent_counts.keys()),
             "Count": list(intent_counts.values())}
        ).sort_values("Count", ascending=False)
        st.bar_chart(intent_df.set_index("Intent"))
    else:
        st.info("No intent predictions yet.")

    st.divider()

    # ── Accuracy panel (if evaluation data available) ─────────────────────────
    if evaluation:
        st.subheader("📐 Accuracy vs Ground Truth")
        st.caption("Measured against the synthetic dataset ground truth labels")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Overall Accuracy", evaluation.get("overall_accuracy_pct", "—"))

        per_dim = evaluation.get("per_dimension", {})
        if per_dim:
            acc_rows = []
            for dim, stats in per_dim.items():
                acc_rows.append({
                    "Dimension": dim,
                    "Accuracy": stats.get("accuracy_pct", "—"),
                    "Correct": stats.get("correct", 0),
                    "Total": stats.get("classified_total", 0),
                    "Unclear": stats.get("unclear_count", 0),
                    "N/A Precision": f"{stats.get('na_precision',0)*100:.0f}%",
                    "N/A Recall": f"{stats.get('na_recall',0)*100:.0f}%",
                })
            st.dataframe(pd.DataFrame(acc_rows).set_index("Dimension"), use_container_width=True)

    st.divider()

    # ── Cross-dimension pivot ─────────────────────────────────────────────────
    st.subheader("Cross-Dimension Co-occurrence")
    st.caption("Select two dimensions to see how their values co-occur")

    col_a, col_b = st.columns(2)
    dim_a = col_a.selectbox("Dimension A", dims, key="pivot_a")
    dim_b = col_b.selectbox("Dimension B", [d for d in dims if d != dim_a], key="pivot_b")

    pivot = defaultdict(Counter)
    for pred in predictions.values():
        a_entry = pred.get(dim_a, {})
        b_entry = pred.get(dim_b, {})
        a_val = a_entry.get("value", "N/A") if a_entry.get("flag") == "classified" else "N/A"
        b_val = b_entry.get("value", "N/A") if b_entry.get("flag") == "classified" else "N/A"
        pivot[a_val][b_val] += 1

    if pivot:
        pivot_df = pd.DataFrame(pivot).fillna(0).astype(int)
        st.dataframe(pivot_df, use_container_width=True)


# ── View 4: Export ────────────────────────────────────────────────────────────

def render_export(comments, predictions, evaluation):
    st.title("📤 Export Report")
    st.caption("Generate a shareable HTML or PDF report from current predictions.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Report Contents")
        include_accuracy = st.checkbox(
            "Include accuracy metrics",
            value=evaluation is not None,
            disabled=evaluation is None,
            help="Only available if evaluation data exists (run pipeline with ground truth).",
        )
        include_intent = st.checkbox("Include intent queue", value=True)
        include_top_comments = st.checkbox("Include top representative comments", value=True)

    with col2:
        st.subheader("Format")
        export_html = st.checkbox("HTML", value=True)
        export_pdf = st.checkbox("PDF", value=True)

    st.divider()

    if st.button("🚀 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                from localtailor.reporter import generate_report

                eval_path = EVAL_PATH if (include_accuracy and evaluation) else None

                result = generate_report(
                    comments_path=COMMENTS_PATH,
                    predictions_path=PREDICTIONS_PATH,
                    post_id=POST_ID,
                    evaluation_path=eval_path,
                    html=export_html,
                    pdf=export_pdf,
                )

                st.success("Report generated!")

                # HTML download
                html_path = result.get("html")
                if html_path and Path(html_path).exists():
                    with open(html_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download HTML Report",
                            data=f.read(),
                            file_name=Path(html_path).name,
                            mime="text/html",
                            use_container_width=True,
                        )
                    st.caption(f"Saved to: `{html_path}`")

                # PDF download
                pdf_path = result.get("pdf")
                if pdf_path and Path(pdf_path).exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=f.read(),
                            file_name=Path(pdf_path).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    st.caption(f"Saved to: `{pdf_path}`")

            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.exception(e)

    st.divider()
    st.subheader("Previously Generated Reports")

    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = sorted(reports_dir.glob("*.html"), reverse=True)[:10]
        if report_files:
            for rfile in report_files:
                size_kb = rfile.stat().st_size // 1024
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"`{rfile.name}` — {size_kb} KB")
                with open(rfile, "rb") as f:
                    col_b.download_button(
                        "⬇️ HTML",
                        data=f.read(),
                        file_name=rfile.name,
                        mime="text/html",
                        key=str(rfile),
                    )
                # Check for matching PDF
                pdf_match = rfile.with_suffix(".pdf")
                if pdf_match.exists():
                    with open(pdf_match, "rb") as f:
                        col_b.download_button(
                            "⬇️ PDF",
                            data=f.read(),
                            file_name=pdf_match.name,
                            mime="application/pdf",
                            key=str(pdf_match),
                        )
        else:
            st.info("No reports generated yet. Click 'Generate Report' above.")
    else:
        st.info("No reports folder found yet.")


# ── View 5: Config Editor ─────────────────────────────────────────────────

def _load_config_raw():
    """Load raw YAML and examples JSON for editing."""
    dims_raw = []
    if DIMENSIONS_YAML.exists():
        with open(DIMENSIONS_YAML, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        dims_raw = raw.get("dimensions", [])

    examples = {}
    if EXAMPLES_JSON.exists():
        with open(EXAMPLES_JSON, "r", encoding="utf-8") as f:
            examples = json.load(f)

    return dims_raw, examples


def _save_config(dims_raw, examples):
    """Write dimensions YAML and examples JSON back to disk."""
    DIMENSIONS_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(DIMENSIONS_YAML, "w", encoding="utf-8") as f:
        yaml.dump({"dimensions": dims_raw}, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Preserve metadata keys
    out = {}
    if EXAMPLES_JSON.exists():
        with open(EXAMPLES_JSON, "r", encoding="utf-8") as f:
            old = json.load(f)
        for k, v in old.items():
            if k.startswith("_"):
                out[k] = v
    out.update({k: v for k, v in examples.items() if not k.startswith("_")})

    EXAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EXAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")


def render_config_editor():
    st.title("⚙️ Config Editor")
    st.caption("Edit dimensions, values, and training examples. Changes are saved to config/dimensions.yaml and data/examples.json.")

    dims_raw, examples = _load_config_raw()

    # Initialize session state from files on first load
    if "cfg_dims" not in st.session_state:
        st.session_state.cfg_dims = dims_raw
        st.session_state.cfg_examples = {k: v for k, v in examples.items() if not k.startswith("_")}

    dims = st.session_state.cfg_dims
    exs = st.session_state.cfg_examples

    # ── Add new dimension ────────────────────────────────────────────────────
    st.subheader("Dimensions")

    with st.expander("Add new dimension", expanded=False):
        new_dim_name = st.text_input("Dimension name", key="new_dim_name", placeholder="e.g. material")
        new_dim_vals = st.text_input("Values (comma-separated)", key="new_dim_vals", placeholder="e.g. cotton, polyester, silk")
        if st.button("Add dimension", key="add_dim_btn"):
            name = new_dim_name.strip().lower().replace(" ", "_")
            if not name:
                st.error("Name cannot be empty.")
            elif any(d.get("name") == name for d in dims):
                st.error(f"Dimension '{name}' already exists.")
            elif not new_dim_vals.strip():
                st.error("Provide at least 2 comma-separated values.")
            else:
                labels = [v.strip() for v in new_dim_vals.split(",") if v.strip()]
                if len(labels) < 2:
                    st.error("Need at least 2 values.")
                else:
                    dims.append({
                        "name": name,
                        "enabled": True,
                        "values": [{"label": lbl, "description": ""} for lbl in labels],
                    })
                    exs[name] = {lbl: [] for lbl in labels}
                    st.rerun()

    st.divider()

    # ── Per-dimension editor ─────────────────────────────────────────────────
    if not dims:
        st.info("No dimensions configured. Add one above.")
        return

    dim_names = [d.get("name", f"dim_{i}") for i, d in enumerate(dims)]
    selected_idx = st.selectbox(
        "Select dimension to edit",
        range(len(dims)),
        format_func=lambda i: dim_names[i].replace("_", " ").title(),
        key="cfg_dim_select",
    )
    dim = dims[selected_idx]
    dim_name = dim.get("name", "")

    # Delete dimension
    col_header, col_delete = st.columns([4, 1])
    col_header.subheader(dim_name.replace("_", " ").title())
    if col_delete.button("Delete dimension", key=f"del_dim_{dim_name}", type="secondary"):
        dims.pop(selected_idx)
        exs.pop(dim_name, None)
        st.rerun()

    # Enabled toggle
    dim["enabled"] = st.checkbox("Enabled", value=dim.get("enabled", True), key=f"enabled_{dim_name}")

    st.divider()

    # ── Values ───────────────────────────────────────────────────────────────
    values = dim.get("values", [])
    dim_examples = exs.get(dim_name, {})

    for vi, val in enumerate(values):
        label = val.get("label", "") if isinstance(val, dict) else val
        val_key = f"{dim_name}_{vi}_{label}"

        col_label, col_desc, col_del = st.columns([2, 3, 1])
        with col_label:
            st.markdown(f"**{label}**")
        with col_desc:
            new_desc = st.text_input(
                "Description",
                value=val.get("description", "") if isinstance(val, dict) else "",
                key=f"desc_{val_key}",
                label_visibility="collapsed",
                placeholder="Description...",
            )
            if isinstance(val, dict):
                val["description"] = new_desc
        with col_del:
            if st.button("Remove", key=f"del_val_{val_key}"):
                values.pop(vi)
                dim_examples.pop(label, None)
                st.rerun()

        # Examples for this value
        val_examples = dim_examples.get(label, [])
        for ei, ex_text in enumerate(val_examples):
            ex_col, del_col = st.columns([6, 1])
            with ex_col:
                updated = st.text_input(
                    f"Example {ei+1}",
                    value=ex_text,
                    key=f"ex_{val_key}_{ei}",
                    label_visibility="collapsed",
                )
                if updated != ex_text:
                    val_examples[ei] = updated
            with del_col:
                if st.button("x", key=f"del_ex_{val_key}_{ei}"):
                    val_examples.pop(ei)
                    dim_examples[label] = val_examples
                    st.rerun()

        # Add example
        new_ex = st.text_input(
            "Add example",
            key=f"new_ex_{val_key}",
            placeholder="Type a training example and press Enter...",
            label_visibility="collapsed",
        )
        if new_ex.strip():
            val_examples.append(new_ex.strip())
            dim_examples[label] = val_examples
            st.rerun()

        count = len(val_examples)
        color = "#27AE60" if count >= 8 else "#E67E22" if count >= 4 else "#E74C3C"
        st.markdown(f'<span style="font-size:12px; color:{color};">{count} example{"s" if count != 1 else ""} (8 recommended)</span>', unsafe_allow_html=True)
        st.divider()

    # ── Add new value ────────────────────────────────────────────────────────
    add_col1, add_col2 = st.columns([3, 1])
    new_val_label = add_col1.text_input("New value label", key=f"new_val_{dim_name}", placeholder="e.g. memory foam")
    if add_col2.button("Add value", key=f"add_val_btn_{dim_name}"):
        lbl = new_val_label.strip()
        if not lbl:
            st.error("Label cannot be empty.")
        elif any((v.get("label") if isinstance(v, dict) else v) == lbl for v in values):
            st.error(f"Value '{lbl}' already exists.")
        else:
            values.append({"label": lbl, "description": ""})
            dim_examples[lbl] = []
            exs[dim_name] = dim_examples
            st.rerun()

    st.divider()

    # ── Save button ──────────────────────────────────────────────────────────
    if st.button("Save changes", type="primary", use_container_width=True):
        _save_config(dims, exs)
        st.success("Saved to config/dimensions.yaml and data/examples.json")
        st.caption("Run `python run_pipeline.py retrain` to retrain with the updated config.")


# TODO: render_login() — Facebook access token input, store in st.session_state
#   - Text input for access token (or paste from Graph API Explorer)
#   - "Validate" button → call FB API /me to confirm token works
#   - Show token status (valid/expired) and connected page name
#   - Save token to local config (never committed to git)

# TODO: render_facebook() — Facebook post selector + comment fetcher
#   - Input: Facebook post URL or post ID
#   - "Fetch Comments" button → call Graph API /{post_id}/comments
#   - Show preview of fetched comments (count, sample)
#   - Save to data/comments_clean_{post_id}.json in expected schema
#   - Handle pagination (FB returns 25 per page by default)
#   - Handle rate limits and token expiration gracefully


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    comments, predictions, evaluation = load_data()

    # First-time user: no config exists yet → go straight to Config
    has_config = DIMENSIONS_YAML.exists() and DIMENSIONS_YAML.stat().st_size > 10

    view = render_sidebar(comments, predictions, evaluation)

    if "Config" in view:
        render_config_editor()
        return

    # TODO: if "Login" in view: render_login(); return
    # TODO: if "Facebook" in view: render_facebook(); return

    if not has_config:
        st.info("Welcome to Local Tailor! Start by setting up your dimensions.")
        st.markdown("Go to **Config** in the sidebar to define your dimensions and add training examples.")
        st.markdown("Once configured, run `python run_pipeline.py retrain` to train and predict.")
        return

    if comments is None:
        st.warning("No comments found. Fetch comments from Facebook or run the pipeline.")
        st.code("python run_pipeline.py predict")
        return

    if not predictions:
        st.warning("No predictions yet. Run the pipeline to classify your comments.")
        st.code("python run_pipeline.py predict")
        return

    if "Dimension Board" in view:
        render_dimension_board(comments, predictions)
    elif "Intent Queue" in view:
        render_intent_queue(comments, predictions)
    elif "Analytics" in view:
        render_analytics(comments, predictions, evaluation)
    elif "Export" in view:
        render_export(comments, predictions, evaluation)


if __name__ == "__main__":
    main()
