"""
localtailor/reporter.py
=======================
Generates HTML and PDF reports from predictions + evaluation data.

Uses Jinja2 to render templates/report.html, then fpdf2 for PDF.
All generation is fully offline — no system dependencies required.

Output:
  reports/report_{post_id}_{timestamp}.html
  reports/report_{post_id}_{timestamp}.pdf  (if pdf=True)
"""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


INTENT_PRIORITY = [
    "needs reply", "negative review", "Unclear",
    "comparison", "monitor", "positive review", "spam",
]

INTENT_CSS = {
    "needs reply":   "needs-reply",
    "negative review": "negative",
    "comparison":    "comparison",
}


def generate_report(
    comments_path: Path,
    predictions_path: Path,
    post_id: str,
    evaluation_path: Optional[Path] = None,
    html: bool = True,
    pdf: bool = True,
    template_path: str = "templates/report.html",
) -> Dict[str, Path]:
    """Generate HTML and/or PDF report.

    Args:
        comments_path:    Path to comments_clean_{post_id}.json
        predictions_path: Path to predictions_{post_id}.json
        post_id:          Session identifier
        evaluation_path:  Optional path to evaluation_{post_id}.json
        html:             Whether to generate an HTML report
        pdf:              Whether to generate a PDF report
        template_path:    Path to Jinja2 template

    Returns:
        {"html": Path, "pdf": Path}  (keys only present for generated formats)
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        raise ImportError("Run: pip install jinja2")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(comments_path, "r", encoding="utf-8") as f:
        comments_data = json.load(f)
    comments = {c["id"]: c for c in comments_data["comments"]}

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    evaluation = None
    if evaluation_path and Path(evaluation_path).exists():
        with open(evaluation_path, "r", encoding="utf-8") as f:
            evaluation = json.load(f)

    # ── Build template context ────────────────────────────────────────────────
    dims = list(next(iter(predictions.values())).keys()) if predictions else []
    total = len(comments)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    context = {
        "metadata": {
            "post_id": post_id,
            "total_comments": total,
            "dimensions": dims,
            "generated_at": now,
        },
        "summary":      _build_summary(predictions, total),
        "coverage":     _build_coverage(predictions, dims, total),
        "dimensions":   _build_dimension_breakdown(predictions, comments, dims),
        "intent_queue": _build_intent_queue(predictions, comments),
        "accuracy":     evaluation,
    }

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {}

    # ── Save HTML ─────────────────────────────────────────────────────────────
    if html:
        template_dir = str(Path(template_path).parent)
        template_file = Path(template_path).name

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template(template_file)
        html_content = template.render(**context)

        html_path = output_dir / f"report_{post_id}_{timestamp}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"  HTML report -> {html_path}")
        result["html"] = html_path

    # ── Generate PDF ──────────────────────────────────────────────────────────
    if pdf:
        pdf_path = output_dir / f"report_{post_id}_{timestamp}.pdf"
        try:
            result["pdf"] = _generate_pdf(context, pdf_path)
            print(f"  PDF report  -> {pdf_path}")
        except ImportError:
            print("  WARNING: fpdf2 not installed. PDF skipped.")
            print("     Install: pip install fpdf2")
        except Exception as e:
            print(f"  WARNING: PDF generation failed: {e}")

    if not result:
        print("  No format selected, nothing generated.")

    return result


# ── Context builders ──────────────────────────────────────────────────────────

def _build_summary(predictions: Dict, total: int) -> Dict:
    needs_reply = 0
    negative_reviews = 0
    unclear_total = 0

    for pred in predictions.values():
        intent = pred.get("intent", {})
        if intent.get("value") == "needs reply" and intent.get("flag") == "classified":
            needs_reply += 1
        if intent.get("value") == "negative review" and intent.get("flag") == "classified":
            negative_reviews += 1
        for dim_pred in pred.values():
            if dim_pred.get("flag") == "unclear":
                unclear_total += 1
                break  # count per comment, not per dimension

    return {
        "needs_reply":     needs_reply,
        "negative_reviews": negative_reviews,
        "unclear_total":   unclear_total,
    }


def _build_coverage(predictions: Dict, dims: List[str], total: int) -> List[Dict]:
    coverage = []
    for dim in dims:
        count = sum(
            1 for p in predictions.values()
            if p.get(dim, {}).get("flag") in ("classified", "unclear")
        )
        pct = round(count / total * 100) if total > 0 else 0
        coverage.append({"name": dim, "count": count, "pct": pct})
    coverage.sort(key=lambda x: -x["pct"])
    return coverage


def _build_dimension_breakdown(
    predictions: Dict,
    comments: Dict,
    dims: List[str],
) -> List[Dict]:
    result = []
    for dim in dims:
        # Value → count
        val_counts: Counter = Counter()
        for cid, pred in predictions.items():
            entry = pred.get(dim, {})
            if entry.get("flag") == "classified":
                val = entry.get("value", "?")
                val_counts[val] += 1

        if not val_counts:
            continue

        total_classified = sum(val_counts.values())

        rows = []
        for val, count in sorted(val_counts.items(), key=lambda x: -x[1]):
            pct = round(count / total_classified * 100) if total_classified > 0 else 0
            rows.append({
                "value":    val,
                "count":    count,
                "pct":      pct,
            })

        # Top 3 most confident comments per dimension
        top_entries = []
        for cid, pred in predictions.items():
            entry = pred.get(dim, {})
            if entry.get("flag") == "classified":
                top_entries.append({
                    "cid":     cid,
                    "message": comments.get(cid, {}).get("message", ""),
                    "value":   entry.get("value"),
                    "span":    entry.get("span"),
                    "score":   entry.get("score", 0),
                })
        top_entries.sort(key=lambda x: -x["score"])
        top_comments = top_entries[:3]

        result.append({
            "name":         dim,
            "rows":         rows,
            "top_comments": top_comments,
        })

    return result


def _build_intent_queue(predictions: Dict, comments: Dict) -> List[Dict]:
    queue = []
    for cid, pred in predictions.items():
        intent_entry = pred.get("intent", {})
        tone_entry = pred.get("tone", {})
        intent_val = intent_entry.get("value", "N/A")
        flag = intent_entry.get("flag", "na")

        if flag not in ("classified", "unclear") or intent_val == "N/A":
            continue

        comment = comments.get(cid, {})
        priority = INTENT_PRIORITY.index(intent_val) if intent_val in INTENT_PRIORITY else 99

        queue.append({
            "message":   comment.get("message", ""),
            "intent":    intent_val,
            "tone":      tone_entry.get("value", "—"),
            "like_count": comment.get("like_count", 0),
            "priority":  priority,
            "css_class": INTENT_CSS.get(intent_val, ""),
        })

    queue.sort(key=lambda x: (x["priority"], -x["like_count"]))
    return queue[:30]  # cap at 30 items for the report


# ── PDF generation (fpdf2 — pure Python, no system deps) ─────────────────

def _generate_pdf(context: Dict, pdf_path: Path) -> Path:
    """Build a PDF report using fpdf2. No system dependencies required."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Local Tailor Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    meta = context["metadata"]
    pdf.cell(0, 6, f"Post: {meta['post_id']}  |  {meta['total_comments']} comments  |  {meta['generated_at']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = context["summary"]
    _pdf_heading(pdf, "Summary")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Needs reply: {summary['needs_reply']}    Negative reviews: {summary['negative_reviews']}    Unclear: {summary['unclear_total']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Coverage ─────────────────────────────────────────────────────────
    _pdf_heading(pdf, "Dimension Coverage")
    pdf.set_font("Helvetica", "", 10)
    for cov in context["coverage"]:
        bar_w = cov["pct"] * 1.2  # max ~120mm
        pdf.set_fill_color(46, 117, 182)
        pdf.cell(40, 7, f"{cov['name']}", border=0)
        pdf.cell(bar_w, 7, "", fill=True)
        pdf.cell(0, 7, f"  {cov['pct']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Dimension breakdowns ─────────────────────────────────────────────
    for dim in context["dimensions"]:
        _pdf_heading(pdf, dim["name"].replace("_", " ").title())

        # Value table
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 6, "Value", border=1)
        pdf.cell(30, 6, "Count", border=1, align="C")
        pdf.cell(30, 6, "%", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        for row in dim["rows"]:
            pdf.cell(60, 6, _truncate_to_width(pdf, str(row["value"]), 58), border=1)
            pdf.cell(30, 6, str(row["count"]), border=1, align="C")
            pdf.cell(30, 6, f"{row['pct']}%", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

        # Top comments
        if dim.get("top_comments"):
            pdf.ln(2)
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 5, "Top comments:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for tc in dim["top_comments"]:
                msg = tc["message"][:120] + ("..." if len(tc["message"]) > 120 else "")
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, f"[{tc['value']}] {msg}  (conf: {tc['score']:.2f})")
        pdf.ln(4)

    # ── Intent queue ─────────────────────────────────────────────────────
    intent_queue = context.get("intent_queue", [])
    if intent_queue:
        _pdf_heading(pdf, "Intent Queue (top 30)")
        comment_w = pdf.w - pdf.l_margin - pdf.r_margin - 50  # 30 + 20 for intent/tone
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(30, 6, "Intent", border=1)
        pdf.cell(20, 6, "Tone", border=1)
        pdf.cell(comment_w, 6, "Comment", border=1, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8)
        for item in intent_queue:
            msg = _truncate_to_width(pdf, item["message"], comment_w - 2)
            pdf.cell(30, 6, item["intent"], border=1)
            pdf.cell(20, 6, str(item["tone"]), border=1)
            pdf.cell(comment_w, 6, msg, border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── Accuracy ─────────────────────────────────────────────────────────
    accuracy = context.get("accuracy")
    if accuracy:
        _pdf_heading(pdf, "Accuracy vs Ground Truth")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Overall: {accuracy.get('overall_accuracy_pct', 'N/A')}", new_x="LMARGIN", new_y="NEXT")

        per_dim = accuracy.get("per_dimension", {})
        if per_dim:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(40, 6, "Dimension", border=1)
            pdf.cell(25, 6, "Accuracy", border=1, align="C")
            pdf.cell(25, 6, "Correct", border=1, align="C")
            pdf.cell(25, 6, "Total", border=1, align="C")
            pdf.cell(25, 6, "Unclear", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 9)
            for dim_name, stats in per_dim.items():
                pdf.cell(40, 6, dim_name, border=1)
                pdf.cell(25, 6, str(stats.get("accuracy_pct", "—")), border=1, align="C")
                pdf.cell(25, 6, str(stats.get("correct", 0)), border=1, align="C")
                pdf.cell(25, 6, str(stats.get("classified_total", 0)), border=1, align="C")
                pdf.cell(25, 6, str(stats.get("unclear_count", 0)), border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))
    return pdf_path


def _truncate_to_width(pdf, text: str, max_w: float) -> str:
    """Truncate text so it fits within max_w mm at the current font."""
    if pdf.get_string_width(text) <= max_w:
        return text
    while len(text) > 0 and pdf.get_string_width(text + "...") > max_w:
        text = text[:-1]
    return text + "..."


def _pdf_heading(pdf, text: str):
    """Render a section heading in the PDF."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(31, 56, 100)
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(46, 117, 182)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)
