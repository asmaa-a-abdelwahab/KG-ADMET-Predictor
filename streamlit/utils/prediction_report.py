from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _h(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    text = str(value).strip()
    return html.escape(text if text and text.casefold() not in {"nan", "none", "null"} else "N/A")


def _pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "N/A"


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], empty_text: str = "No records available.") -> str:
    if not rows:
        return f'<p class="muted">{_h(empty_text)}</p>'
    head = "".join(f"<th>{_h(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_h(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _status_label(status: str) -> str:
    return {
        "known_interaction_rediscovered": "Known / rediscovered",
        "known_interaction_not_rediscovered": "Known evidence, model disagreement",
        "prediction_conflicts_with_known_inactive": "Conflicts with curated inactive evidence",
        "known_inactive_consistent": "Consistent with curated inactive evidence",
        "known_evidence_conflict": "Conflicting curated evidence",
        "novel_predicted_interaction": "Novel predicted interaction",
        "interaction_not_predicted": "Interaction not predicted",
    }.get(status, status.replace("_", " ").title())


def _status_class(status: str) -> str:
    if status == "known_interaction_rediscovered":
        return "known"
    if status in {
        "known_interaction_not_rediscovered",
        "prediction_conflicts_with_known_inactive",
        "known_evidence_conflict",
    }:
        return "warning-status"
    if status == "novel_predicted_interaction":
        return "predicted"
    return "not-predicted"


def _score_source_label(source: Any) -> str:
    text = str(source or "").strip().casefold()
    return {
        "validated_precomputed_record": "Validated precomputed record",
        "production_prediction_cache": "Production prediction cache",
        "live_component_inference": "New live component inference",
    }.get(text, "Unknown" if not text or text == "nan" else str(source))


def _probability_bar(probability: Any, threshold: Any) -> str:
    try:
        p = max(0.0, min(1.0, float(probability))) * 100.0
        t = max(0.0, min(1.0, float(threshold))) * 100.0
    except Exception:
        return ""
    return f"""
    <div class="probability-chart">
      <div class="probability-track">
        <div class="probability-fill" style="width:{p:.2f}%"></div>
        <div class="threshold-line" style="left:{t:.2f}%"><span>Threshold {t:.1f}%</span></div>
      </div>
      <div class="axis"><span>0%</span><span>Predicted {p:.2f}%</span><span>100%</span></div>
    </div>
    """


def _component_bars(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">Component scores unavailable.</p>'
    items = []
    for row in rows:
        try:
            score = max(0.0, min(1.0, float(row.get("score"))))
        except Exception:
            continue
        items.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{_h(row.get('display_name'))}</div>
              <div class="mini-track"><div class="mini-fill" style="width:{score*100:.2f}%"></div></div>
              <div class="bar-value">{score:.4f}</div>
            </div>
            """
        )
    return "<div class='bar-panel'>" + "".join(items) + "</div>"


def _contribution_bars(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">Local contribution analysis unavailable.</p>'
    max_abs = max(abs(float(row.get("probability_change_percentage_points", 0.0) or 0.0)) for row in rows) or 1.0
    items = []
    for row in rows:
        effect = float(row.get("probability_change_percentage_points", 0.0) or 0.0)
        width = min(100.0, abs(effect) / max_abs * 100.0)
        css = "positive" if effect > 0 else "negative" if effect < 0 else "neutral"
        items.append(
            f"""
            <div class="contribution-row">
              <div class="bar-label">{_h(row.get('display_name'))}</div>
              <div class="contribution-track"><div class="contribution-fill {css}" style="width:{width:.2f}%"></div></div>
              <div class="bar-value">{effect:+.2f} pp</div>
            </div>
            """
        )
    return "<div class='bar-panel'>" + "".join(items) + "</div>"


def _executive_summary_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        pair = item.get("pair", {})
        pred = item.get("prediction", {})
        evidence = item.get("evidence", {})
        uncertainty = item.get("explainability", {}).get("uncertainty_metrics", {})
        applicability = item.get("explainability", {}).get("applicability_domain", {})
        rows.append(
            {
                "compound": pair.get("compound_name"),
                "target": pair.get("target_name"),
                "probability": _pct(pred.get("calibrated_probability")),
                "result": _status_label(str(pred.get("result_status", ""))),
                "heuristic_confidence": str(
                    uncertainty.get("heuristic_confidence", uncertainty.get("model_certainty", "N/A"))
                ).title(),
                "evidence": f"{evidence.get('tier', 'N/A')} / {str(evidence.get('evidence_support', 'N/A')).title()}",
                "applicability": str(applicability.get("status", "unknown")).replace("_", " ").title(),
            }
        )
    return rows


def generate_prediction_report_html(payload: dict[str, Any]) -> str:
    predictions = payload.get("predictions", []) or []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    context = payload.get("report_context", {}) or {}
    provenance = context.get("model_provenance", {}) or {}
    validation = context.get("global_validation_metrics", {}) or {}
    parity = context.get("live_inference_parity", {}) or {}
    reference_audit = provenance.get("reference_provenance_audit", {}) or {}
    diagnostic_validation = reference_audit.get("scientific_status") == "diagnostic_only"

    summary_rows = _executive_summary_rows(predictions)
    status_counts = context.get("result_status_counts", {}) or {}
    source_counts = context.get("score_source_counts", {}) or {}

    sections: list[str] = []
    for index, item in enumerate(predictions, start=1):
        pair = item.get("pair", {})
        pred = item.get("prediction", {})
        model = item.get("model", {})
        exp = item.get("explainability", {})
        uncertainty = exp.get("uncertainty_metrics", {}) or {}
        applicability = exp.get("applicability_domain", {}) or {}
        local_container = exp.get("local_component_explanation", {}) or {}
        local = local_container.get("components", exp.get("local_component_contributions", [])) or []
        evidence = item.get("evidence", {}) or {}
        interpretation = item.get("interpretation", {}) or {}
        result_status = str(pred.get("result_status", ""))

        local_rows = [
            {
                "component": row.get("display_name"),
                "score": _num(row.get("value"), 4),
                "background": _num(row.get("background_median"), 4),
                "scope": str(row.get("background_scope", "")).replace("_", " "),
                "effect": f"{float(row.get('probability_change_percentage_points', 0.0) or 0.0):+.2f} pp",
                "direction": row.get("effect_relative_to_background"),
                "share": _pct(row.get("local_importance_share")) if row.get("local_importance_share") is not None else "Suppressed",
            }
            for row in local
        ]
        shap = exp.get("tree_shap", {}) or {}
        shap_rows = [
            {
                "component": row.get("display_name"),
                "score": _num(row.get("value"), 4),
                "shap": _num(row.get("shap_value_raw_ensemble_output"), 5),
                "effect": row.get("effect"),
            }
            for row in (shap.get("values", []) or [])
        ]
        domain_rows = [
            {
                "component": item_row.get("component_score"),
                "score": _num(item_row.get("value"), 4),
                "percentile": _pct(item_row.get("percentile")),
                "reference": f"{_num(item_row.get('q01'),4)}–{_num(item_row.get('q99'),4)}",
                "status": str(item_row.get("status", "")).replace("_", " "),
                "saturated": "Yes" if item_row.get("saturated") else "No",
            }
            for item_row in applicability.get("components", [])
        ]
        direct_rows = []
        for direct in evidence.get("direct_interactions", []) or []:
            interaction = direct.get("interaction", {}) or {}
            direct_rows.append(
                {
                    "interaction_id": direct.get("interaction_id"),
                    "label": interaction.get("label", interaction.get("interaction_label", interaction.get("type"))),
                    "endpoint_count": direct.get("endpoint_count"),
                    "measure_groups": direct.get("measure_group_count"),
                    "bioassays": direct.get("bioassay_count"),
                    "source": interaction.get("source", interaction.get("source_name", interaction.get("database"))),
                    "reference": interaction.get("reference", interaction.get("pmid", interaction.get("doi"))),
                }
            )
        analog_rows = []
        for analog in evidence.get("similar_compound_support", []) or []:
            interaction = analog.get("interaction", {}) or {}
            analog_rows.append(
                {
                    "cid": analog.get("cid"),
                    "compound": analog.get("name"),
                    "similarity": _num(analog.get("similarity"), 4),
                    "same_target": "Yes" if analog.get("same_target") else "No",
                    "endpoint_count": analog.get("endpoint_count"),
                    "interaction_label": interaction.get("label", interaction.get("interaction_label", interaction.get("type"))),
                    "source": interaction.get("source", interaction.get("source_name", interaction.get("database"))),
                    "reference": interaction.get("reference", interaction.get("pmid", interaction.get("doi"))),
                }
            )

        shap_html = ""
        if shap.get("status") == "computed" and shap_rows:
            shap_html = (
                "<h3>TreeSHAP ensemble explanation</h3>"
                "<p>TreeSHAP explains the uncalibrated ensemble output. Calibrated-probability effects are reported separately above.</p>"
                + _table(shap_rows, [("component", "Component"), ("score", "Score"), ("shap", "SHAP value"), ("effect", "Effect")])
            )
        elif shap.get("status") == "unavailable":
            shap_html = f'<p class="note"><strong>TreeSHAP diagnostic:</strong> {_h(shap.get("reason"))}</p>'

        relative_note = ""
        if local_container.get("relative_importance_reliable") is False:
            relative_note = f'<p class="note">{_h(local_container.get("relative_importance_note"))}</p>'

        sections.append(
            f"""
            <section class="prediction-section">
              <div class="section-heading">
                <span class="pair-number">{index}</span>
                <div>
                  <h2>{_h(pair.get('compound_name'))} × {_h(pair.get('target_name'))}</h2>
                  <p>{_h(pair.get('compound_key'))} → {_h(pair.get('target_key'))}</p>
                  <span class="status-badge {_status_class(result_status)}">{_h(_status_label(result_status))}</span>
                </div>
              </div>

              <div class="metric-grid">
                <div><span>Calibrated probability</span><strong>{_pct(pred.get('calibrated_probability'))}</strong></div>
                <div><span>Decision threshold</span><strong>{_pct(pred.get('threshold'))}</strong></div>
                <div><span>Heuristic confidence</span><strong>{_h(str(uncertainty.get('heuristic_confidence', uncertainty.get('model_certainty', 'N/A'))).title())}</strong></div>
                <div><span>Evidence support</span><strong>{_h(str(evidence.get('evidence_support', 'N/A')).title())}</strong></div>
                <div><span>Evidence tier</span><strong>{_h(evidence.get('tier'))}</strong></div>
                <div><span>Applicability domain</span><strong>{_h(str(applicability.get('status', 'unknown')).replace('_',' ').title())}</strong></div>
                <div><span>Score source</span><strong>{_h(_score_source_label(model.get('score_source')))}</strong></div>
                <div><span>Graph snapshot</span><strong>{_h(model.get('graph_version'))}</strong></div>
                <div><span>Recommended action</span><strong>{_h(interpretation.get('recommended_action'))}</strong></div>
              </div>

              <h3>Decision visualization</h3>
              {_probability_bar(pred.get('calibrated_probability'), pred.get('threshold'))}
              <p>{_h(interpretation.get('statement'))}</p>

              <h3>Component scores</h3>
              {_component_bars(item.get('component_score_details', []))}

              <h3>Local calibrated-probability explanation</h3>
              <p>Each component is replaced with a {_h(str(local_container.get('background_scope','reference')).replace('_',' '))} median. Effects are reported in probability percentage points.</p>
              {_contribution_bars(local)}
              {relative_note}
              {_table(local_rows, [('component','Component'),('score','Pair score'),('background','Background median'),('scope','Background scope'),('effect','Probability effect'),('direction','Effect relative to background'),('share','Relative share')])}
              {shap_html}

              <h3>Uncertainty and applicability</h3>
              <p class="muted">The confidence band is a heuristic diagnostic based on threshold margin, component dispersion, and entropy. It is not a validated uncertainty interval.</p>
              <div class="metric-grid compact">
                <div><span>Decision margin</span><strong>{_num(uncertainty.get('decision_margin'),4)}</strong></div>
                <div><span>Model disagreement (SD)</span><strong>{_num(uncertainty.get('component_disagreement_std'),4)}</strong></div>
                <div><span>Predictive entropy</span><strong>{_num(uncertainty.get('predictive_entropy_bits'),4)} bits</strong></div>
                <div><span>Applicability scope</span><strong>{_h(str(applicability.get('scope','unknown')).replace('_',' '))}</strong></div>
                <div><span>Reference sample</span><strong>{_h(applicability.get('sample_size'))}</strong></div>
                <div><span>Applicability interpretation</span><strong>{_h(applicability.get('reason'))}</strong></div>
              </div>
              {_table(domain_rows, [('component','Component'),('score','Score'),('percentile','Reference percentile'),('reference','1st–99th percentile'),('status','Status'),('saturated','Near 0/1')])}

              <h3>Knowledge-graph evidence</h3>
              <p><strong>{_h(evidence.get('tier'))}:</strong> {_h(evidence.get('tier_reason'))}</p>
              <p>Direct interaction: <strong>{_h(evidence.get('known_direct_interaction', False))}</strong>. Provenance completeness: <strong>{_h(evidence.get('provenance_completeness'))}</strong>. Endpoint paths: <strong>{_h(evidence.get('endpoint_path_count', 0))}</strong>. Measure groups: <strong>{_h(evidence.get('measure_group_count', 0))}</strong>. BioAssays: <strong>{_h(evidence.get('bioassay_count', 0))}</strong>.</p>
              <h4>Direct assertions</h4>
              {_table(direct_rows, [('interaction_id','Interaction ID'),('label','Label'),('endpoint_count','Endpoints'),('measure_groups','Measure groups'),('bioassays','BioAssays'),('source','Source'),('reference','Reference')])}
              <h4>Same-target analogue support</h4>
              {_table(analog_rows, [('cid','CID'),('compound','Compound'),('similarity','Similarity'),('same_target','Same CYP target'),('endpoint_count','Endpoints'),('interaction_label','Interaction label'),('source','Source'),('reference','Reference')])}

              <div class="warning"><strong>Interpretation boundary:</strong> {_h(interpretation.get('scope_warning'))}</div>
            </section>
            """
        )

    validation_rows = [
        {"metric": "MCC", "value": _num(validation.get("mcc"), 4)},
        {"metric": "Balanced accuracy", "value": _num(validation.get("balanced_accuracy"), 4)},
        {"metric": "ROC-AUC", "value": _num(validation.get("roc_auc"), 4)},
        {"metric": "Average precision", "value": _num(validation.get("average_precision"), 4)},
        {"metric": "Sensitivity / recall", "value": _num(validation.get("recall"), 4)},
        {"metric": "Specificity", "value": _num(validation.get("specificity"), 4)},
        {"metric": "Brier score", "value": _num(validation.get("brier_score"), 4)},
        {"metric": "Expected calibration error", "value": _num(validation.get("expected_calibration_error"), 4)},
    ]
    provenance_rows = [
        {"field": "Model", "value": provenance.get("model_name")},
        {"field": "Model version", "value": provenance.get("model_version")},
        {"field": "Graph snapshot", "value": provenance.get("graph_version")},
        {"field": "Calibration", "value": provenance.get("calibration")},
        {"field": "Threshold selection", "value": provenance.get("threshold_selection")},
        {"field": "Publication-valid", "value": False if diagnostic_validation else provenance.get("publishable")},
        {"field": "Reference scientific status", "value": reference_audit.get("scientific_status")},
        {"field": "Component scores", "value": ", ".join(provenance.get("score_columns", []) or [])},
        {"field": "Live parity status", "value": parity.get("status", "not run")},
        {"field": "Live parity sample size", "value": parity.get("sample_size")},
        {"field": "Live parity decision agreement", "value": _pct(parity.get("decision_agreement"))},
    ]
    errors = payload.get("errors", []) or []
    error_html = _table(errors, [("compound", "Compound"), ("target", "Target"), ("error", "Reason")]) if errors else ""
    validation_heading = "Diagnostic validation metrics" if diagnostic_validation else "Frozen-test validation"
    validation_qualification = (
        f'<div class="warning"><strong>Scientific-validity warning:</strong> '
        f'{_h(reference_audit.get("warning"))}</div>'
        if diagnostic_validation else
        "<p>These are model-level validation metrics and are reported once. "
        "They are not pair-specific correctness guarantees.</p>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRING interaction prediction report</title>
<style>
:root{{--navy:#183b56;--teal:#2a7f83;--red:#ef4444;--green:#287a4b;--amber:#b7791f;--ink:#172033;--muted:#64748b;--line:#dbe3ea;--panel:#f8fafc}}
*{{box-sizing:border-box}} body{{font-family:Arial,sans-serif;background:#f4f7fa;color:var(--ink);margin:0;padding:32px}} .report{{max-width:1180px;margin:auto;background:#fff;padding:38px;border-radius:18px;box-shadow:0 14px 45px rgba(15,23,42,.10)}}
h1,h2,h3,h4{{color:var(--navy)}} .subtitle,.muted{{color:var(--muted)}} .overview{{padding:20px;border:1px solid var(--line);border-radius:14px;background:var(--panel);margin:18px 0}} .prediction-section{{margin-top:34px;padding-top:28px;border-top:2px solid #e5e7eb}} .section-heading{{display:flex;gap:14px;align-items:flex-start}} .section-heading h2{{margin:0 0 6px}} .section-heading p{{margin:4px 0;color:var(--muted)}} .pair-number{{display:grid;place-items:center;min-width:38px;height:38px;background:var(--red);color:white;border-radius:50%;font-weight:700}}
.status-badge{{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:700}} .status-badge.known{{background:#dcfce7;color:#166534}} .status-badge.predicted{{background:#dbeafe;color:#1e40af}} .status-badge.not-predicted{{background:#f1f5f9;color:#475569}} .status-badge.warning-status{{background:#fef3c7;color:#92400e}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}} .metric-grid.compact{{grid-template-columns:repeat(3,minmax(0,1fr))}} .metric-grid div{{padding:14px;border:1px solid #e2e8f0;border-radius:12px;background:var(--panel)}} .metric-grid span{{display:block;color:var(--muted);font-size:12px;margin-bottom:7px}} .metric-grid strong{{font-size:15px;line-height:1.35}}
table{{border-collapse:collapse;width:100%;margin:12px 0 22px}} th,td{{border:1px solid var(--line);padding:9px;text-align:left;font-size:13px;vertical-align:top}} th{{background:var(--navy);color:#fff}} tr:nth-child(even){{background:var(--panel)}} .table-wrap{{overflow-x:auto}}
.probability-chart{{margin:16px 0 24px}} .probability-track{{height:24px;background:#e2e8f0;border-radius:999px;position:relative;overflow:visible}} .probability-fill{{height:100%;background:linear-gradient(90deg,var(--teal),var(--green));border-radius:999px}} .threshold-line{{position:absolute;top:-7px;height:38px;border-left:3px solid var(--red)}} .threshold-line span{{position:absolute;top:-24px;left:5px;font-size:11px;color:var(--red);white-space:nowrap}} .axis{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:7px}}
.bar-panel{{border:1px solid var(--line);border-radius:12px;padding:14px;background:var(--panel);margin:12px 0 20px}} .bar-row,.contribution-row{{display:grid;grid-template-columns:220px 1fr 90px;gap:12px;align-items:center;margin:10px 0}} .bar-label{{font-size:13px;font-weight:600}} .bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-size:13px}} .mini-track,.contribution-track{{height:14px;background:#e2e8f0;border-radius:999px;overflow:hidden}} .mini-fill{{height:100%;background:var(--teal)}} .contribution-fill{{height:100%}} .contribution-fill.positive{{background:var(--green)}} .contribution-fill.negative{{background:var(--red)}} .contribution-fill.neutral{{background:#94a3b8}}
.warning{{padding:14px;border-left:5px solid #d98952;background:#fff7ed;margin-top:18px}} .note{{padding:12px;background:#eff6ff;border-left:4px solid #3b82f6}} .footer{{margin-top:32px;color:var(--muted);font-size:12px;border-top:1px solid #e2e8f0;padding-top:14px}}
@media(max-width:800px){{.metric-grid,.metric-grid.compact{{grid-template-columns:1fr}} .bar-row,.contribution-row{{grid-template-columns:1fr}} body{{padding:10px}} .report{{padding:18px}}}}
</style></head><body><main class="report">
<h1>PRING compound-target prediction report</h1>
<p class="subtitle">Generated {generated}. Calibrated prediction, applicability-domain diagnostics, component explanation, model validation and evidence provenance.</p>
{validation_qualification if diagnostic_validation else ""}
<div class="overview"><strong>Requested pairs:</strong> {_h(payload.get('requested_pairs'))} &nbsp; <strong>Successful:</strong> {_h(payload.get('successful_pairs'))} &nbsp; <strong>Known/rediscovered:</strong> {_h(status_counts.get('known_interaction_rediscovered',0))} &nbsp; <strong>Novel predicted:</strong> {_h(status_counts.get('novel_predicted_interaction',0))} &nbsp; <strong>Not predicted:</strong> {_h(status_counts.get('interaction_not_predicted',0))}</div>
<h2>Executive summary</h2>
{_table(summary_rows, [('compound','Compound'),('target','Target'),('probability','Probability'),('result','Result'),('heuristic_confidence','Heuristic confidence'),('evidence','Evidence'),('applicability','Applicability')])}
<h2>Model and data provenance</h2>
{_table(provenance_rows, [('field','Field'),('value','Value')])}
<p><strong>Score sources used in this report:</strong> {_h(json.dumps(source_counts, sort_keys=True))}</p>
<h2>{validation_heading}</h2>
{validation_qualification}
{_table(validation_rows, [('metric','Metric'),('value','Value')])}
{''.join(sections)}
{('<h2>Pairs not scored</h2>'+error_html) if errors else ''}
<h2>Scientific limitations</h2>
<ul>
<li>The task is the aggregated PRING compound-CYP450 interaction label; it does not automatically distinguish substrate, inhibitor, inducer, binder, or metabolic mechanism.</li>
<li>A high probability is a prioritization signal, not proof of mechanism, causality, clinical effect, safety, or biological activity.</li>
<li>Tier 1 results are known interactions or rediscoveries and must not be counted as novel missing-interaction predictions.</li>
<li>Tier 3 results are model-only hypotheses and require the strongest independent validation.</li>
<li>Live predictions are allowed only after parity testing against validated precomputed component scores.</li>
</ul>
<div class="footer">Experimental or independent database validation is required before treating a predicted missing interaction as confirmed.</div>
</main></body></html>"""


def prediction_summary_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("predictions", []):
        pair = item.get("pair", {})
        pred = item.get("prediction", {})
        evidence = item.get("evidence", {})
        exp = item.get("explainability", {})
        uncertainty = exp.get("uncertainty_metrics", {})
        applicability = exp.get("applicability_domain", {})
        rows.append(
            {
                "compound": pair.get("compound_name"),
                "cid": pair.get("cid"),
                "target": pair.get("target_name"),
                "protein_id": pair.get("protein_id"),
                "calibrated_probability": pred.get("calibrated_probability"),
                "threshold": pred.get("threshold"),
                "result_status": pred.get("result_status"),
                "prediction": pred.get("predicted_class"),
                "heuristic_confidence": uncertainty.get(
                    "heuristic_confidence",
                    uncertainty.get("model_certainty"),
                ),
                "evidence_tier": evidence.get("tier"),
                "evidence_support": evidence.get("evidence_support"),
                "known_direct_interaction": evidence.get("known_direct_interaction"),
                "applicability_domain": applicability.get("status"),
                "decision_margin": uncertainty.get("decision_margin"),
                "component_disagreement_std": uncertainty.get("component_disagreement_std"),
                "score_source": item.get("model", {}).get("score_source"),
                "recommended_action": item.get("interpretation", {}).get("recommended_action"),
            }
        )
    return pd.DataFrame(rows)


def prediction_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
