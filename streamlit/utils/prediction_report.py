from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _h(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return '<p class="muted">No records available.</p>'
    head = "".join(f"<th>{_h(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_h(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def generate_prediction_report_html(payload: dict[str, Any]) -> str:
    predictions = payload.get("predictions", [])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []
    for index, item in enumerate(predictions, start=1):
        pair = item.get("pair", {})
        pred = item.get("prediction", {})
        model = item.get("model", {})
        exp = item.get("explainability", {})
        uncertainty = exp.get("uncertainty_metrics", {})
        evidence = item.get("evidence", {})
        validation = item.get("validation", {})
        global_metrics = validation.get("global_test_metrics", {}) or {}
        local = exp.get("local_component_contributions", []) or []
        local_rows = [
            {
                "component": row.get("display_name"),
                "score": _num(row.get("value"), 4),
                "background": _num(row.get("background_median"), 4),
                "contribution": _num(row.get("probability_change_when_replaced"), 4),
                "direction": row.get("direction"),
                "share": _pct(row.get("local_importance_share")),
            }
            for row in local
        ]
        tree_shap = exp.get("tree_shap", {}) or {}
        shap_rows = [
            {
                "component": row.get("display_name"),
                "score": _num(row.get("value"), 4),
                "shap": _num(row.get("shap_value_raw_ensemble_output"), 5),
                "direction": row.get("direction"),
            }
            for row in (tree_shap.get("values", []) or [])
        ]
        analogs = evidence.get("similar_compound_support", []) or []
        analog_rows = [
            {
                "cid": a.get("cid"),
                "name": a.get("name"),
                "similarity": _num(a.get("similarity"), 4),
            }
            for a in analogs
        ]
        sections.append(
            f"""
            <section class="prediction-section">
              <div class="section-heading">
                <span class="pair-number">{index}</span>
                <div><h2>{_h(pair.get('compound_name'))} × {_h(pair.get('target_name'))}</h2>
                <p>{_h(pair.get('compound_key'))} → {_h(pair.get('target_key'))}</p></div>
              </div>
              <div class="metric-grid">
                <div><span>Calibrated probability</span><strong>{_pct(pred.get('calibrated_probability'))}</strong></div>
                <div><span>Prediction</span><strong>{_h(pred.get('predicted_class'))}</strong></div>
                <div><span>Locked threshold</span><strong>{_num(pred.get('threshold'))}</strong></div>
                <div><span>Evidence tier</span><strong>{_h(evidence.get('tier'))}</strong></div>
                <div><span>Confidence band</span><strong>{_h(uncertainty.get('confidence_band'))}</strong></div>
                <div><span>Score source</span><strong>{_h(model.get('score_source'))}</strong></div>
              </div>

              <h3>How the prediction was produced</h3>
              <p>The deployable ensemble combines leakage-safe Stage 1 structural scoring with Stage 3 R-GCN and HGT predictions. The ensemble score is transformed by {_h(model.get('calibration'))} and compared with the validation-selected threshold.</p>

              <h3>Local component explanation</h3>
              <p>Each contribution is the change in final calibrated probability when that component score is replaced with its training-background median. Positive values support the active-interaction decision; negative values support the inactive decision.</p>
              {_table(local_rows, [('component','Component'),('score','Pair score'),('background','Background median'),('contribution','Probability change'),('direction','Direction'),('share','Local importance')])}

              <h3>TreeSHAP explanation</h3>
              <p>TreeSHAP explains the uncalibrated Extra Trees ensemble output; the replacement analysis above explains changes after probability calibration.</p>
              {_table(shap_rows, [('component','Component'),('score','Pair score'),('shap','SHAP value'),('direction','Direction')]) if shap_rows else '<p class="muted">TreeSHAP was unavailable for this run.</p>'}

              <h3>Uncertainty and explainability metrics</h3>
              <div class="metric-grid compact">
                <div><span>Decision margin</span><strong>{_num(uncertainty.get('decision_margin'),4)}</strong></div>
                <div><span>Model disagreement (SD)</span><strong>{_num(uncertainty.get('component_disagreement_std'),4)}</strong></div>
                <div><span>Predictive entropy</span><strong>{_num(uncertainty.get('predictive_entropy_bits'),4)} bits</strong></div>
                <div><span>Test MCC</span><strong>{_num(global_metrics.get('mcc'),4)}</strong></div>
                <div><span>Test balanced accuracy</span><strong>{_num(global_metrics.get('balanced_accuracy'),4)}</strong></div>
                <div><span>Test ECE</span><strong>{_num(global_metrics.get('expected_calibration_error'),4)}</strong></div>
              </div>

              <h3>Knowledge-graph evidence</h3>
              <p><strong>{_h(evidence.get('tier'))}:</strong> {_h(evidence.get('tier_reason'))}</p>
              <p>Direct interaction present: <strong>{_h(evidence.get('known_direct_interaction', False))}</strong>. Endpoint paths: <strong>{_h(evidence.get('endpoint_path_count', 0))}</strong>. Measure groups: <strong>{_h(evidence.get('measure_group_count', 0))}</strong>.</p>
              <h4>Similar-compound support</h4>
              {_table(analog_rows, [('cid','CID'),('name','Compound'),('similarity','Similarity')])}

              <div class="warning"><strong>Interpretation boundary:</strong> {_h(item.get('interpretation', {}).get('scope_warning'))}</div>
            </section>
            """
        )

    errors = payload.get("errors", []) or []
    error_html = _table(errors, [("compound", "Compound"), ("target", "Target"), ("error", "Reason")]) if errors else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRING interaction prediction report</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f7fa;color:#172033;margin:0;padding:32px}} .report{{max-width:1120px;margin:auto;background:#fff;padding:38px;border-radius:18px;box-shadow:0 14px 45px rgba(15,23,42,.10)}}
h1,h2,h3,h4{{color:#183b56}} .subtitle,.muted{{color:#64748b}} .prediction-section{{margin-top:28px;padding-top:26px;border-top:2px solid #e5e7eb}} .section-heading{{display:flex;gap:14px;align-items:center}} .section-heading p{{margin:4px 0;color:#64748b}} .pair-number{{display:grid;place-items:center;width:38px;height:38px;background:#ef4444;color:white;border-radius:50%;font-weight:700}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}} .metric-grid.compact{{grid-template-columns:repeat(3,minmax(0,1fr))}} .metric-grid div{{padding:14px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc}} .metric-grid span{{display:block;color:#64748b;font-size:12px;margin-bottom:7px}} .metric-grid strong{{font-size:16px}}
table{{border-collapse:collapse;width:100%;margin:12px 0 22px}} th,td{{border:1px solid #dbe3ea;padding:9px;text-align:left;font-size:13px}} th{{background:#183b56;color:#fff}} tr:nth-child(even){{background:#f8fafc}} .warning{{padding:14px;border-left:5px solid #d98952;background:#fff7ed;margin-top:18px}} .footer{{margin-top:32px;color:#64748b;font-size:12px;border-top:1px solid #e2e8f0;padding-top:14px}}
@media(max-width:800px){{.metric-grid,.metric-grid.compact{{grid-template-columns:1fr}} body{{padding:10px}} .report{{padding:18px}}}}
</style></head><body><main class="report"><h1>PRING compound-target prediction report</h1><p class="subtitle">Generated {generated}. Model-guided prioritization with calibrated probability, component-level explanation, uncertainty metrics and Neo4j evidence reconstruction.</p>
<p><strong>Requested pairs:</strong> {_h(payload.get('requested_pairs'))} &nbsp; <strong>Successful:</strong> {_h(payload.get('successful_pairs'))}</p>
{''.join(sections)}
{('<h2>Pairs not scored</h2>'+error_html) if errors else ''}
<div class="footer">This report documents a statistical prediction. Experimental or independent database validation is required before treating a missing interaction as confirmed.</div></main></body></html>"""


def prediction_summary_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("predictions", []):
        pair = item.get("pair", {})
        pred = item.get("prediction", {})
        evidence = item.get("evidence", {})
        unc = item.get("explainability", {}).get("uncertainty_metrics", {})
        rows.append({
            "compound": pair.get("compound_name"),
            "cid": pair.get("cid"),
            "target": pair.get("target_name"),
            "protein_id": pair.get("protein_id"),
            "calibrated_probability": pred.get("calibrated_probability"),
            "threshold": pred.get("threshold"),
            "predicted_class": pred.get("predicted_class"),
            "evidence_tier": evidence.get("tier"),
            "known_direct_interaction": evidence.get("known_direct_interaction"),
            "decision_margin": unc.get("decision_margin"),
            "component_disagreement_std": unc.get("component_disagreement_std"),
            "confidence_band": unc.get("confidence_band"),
            "score_source": item.get("model", {}).get("score_source"),
        })
    return pd.DataFrame(rows)


def prediction_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
