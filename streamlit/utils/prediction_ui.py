from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .prediction_report import (
    generate_prediction_report_html,
    prediction_json_bytes,
    prediction_summary_dataframe,
)


def _fmt_probability(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def _fmt_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _prediction_label(item: dict[str, Any]) -> str:
    pair = item.get("pair", {})
    return f"{pair.get('compound_name', pair.get('cid', 'Compound'))} → {pair.get('target_name', pair.get('protein_id', 'Target'))}"


def _selected_prediction(
    payload: dict[str, Any],
    *,
    selector_key: str,
) -> dict[str, Any] | None:
    """Return the prediction selected within one result tab.

    Streamlit evaluates the content of every tab during the same script run. Each
    selectbox therefore needs a distinct widget key, even though only one tab is
    visible at a time.
    """
    predictions = payload.get("predictions", []) or []
    if not predictions:
        return None

    labels = [_prediction_label(item) for item in predictions]
    if len(labels) == 1:
        return predictions[0]

    selected = st.selectbox(
        "Prediction to inspect",
        labels,
        key=f"prediction_pair_selector_{selector_key}",
    )
    return predictions[labels.index(selected)]


def _render_summary(payload: dict[str, Any]) -> None:
    st.markdown("## Predicted compound-target interactions")
    summary = prediction_summary_dataframe(payload)
    if summary.empty:
        st.warning("No pair was successfully scored.")
    else:
        display = summary.copy()
        display["calibrated_probability"] = display["calibrated_probability"].map(_fmt_probability)
        display["threshold"] = display["threshold"].map(lambda x: _fmt_number(x, 3))
        display["decision_margin"] = display["decision_margin"].map(lambda x: _fmt_number(x, 3))
        display["component_disagreement_std"] = display["component_disagreement_std"].map(lambda x: _fmt_number(x, 3))
        st.dataframe(display, use_container_width=True, hide_index=True)

    errors = payload.get("errors", []) or []
    if errors:
        with st.expander(f"Pairs not scored ({len(errors)})", expanded=True):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    item = _selected_prediction(payload, selector_key="prediction")
    if not item:
        return
    pred = item.get("prediction", {})
    evidence = item.get("evidence", {})
    uncertainty = item.get("explainability", {}).get("uncertainty_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calibrated probability", _fmt_probability(pred.get("calibrated_probability")))
    c2.metric("Decision", pred.get("predicted_class", "N/A"))
    c3.metric("Evidence tier", evidence.get("tier", "N/A"))
    c4.metric("Confidence", str(uncertainty.get("confidence_band", "N/A")).title())
    st.info(item.get("interpretation", {}).get("statement", ""))
    score_source = str(item.get("model", {}).get("score_source", ""))
    if score_source == "live_component_inference":
        cache_status = item.get("prediction_cache", {}).get("status", "unknown")
        st.caption(f"This pair was scored by the live model components. Cache update status: {cache_status}.")
    elif score_source == "live_component_inference_cached":
        st.caption("This pair was previously scored by the live models and was returned from the persistent prediction cache.")
    elif score_source.startswith("precomputed"):
        st.caption("This pair was returned from the validated precomputed component-score frame.")
    if score_source.startswith("degraded"):
        component_status = item.get("component_details", {}).get("component_status", {})
        st.warning(
            "This prediction used fallback background values for one or more unavailable model components. "
            "It is suitable for troubleshooting only and should not be interpreted as a fully live ensemble result."
        )
        with st.expander("Component diagnostics", expanded=False):
            st.json(component_status)
    if evidence.get("known_direct_interaction"):
        st.warning("This relationship already has a direct interaction assertion in Neo4j. Interpret the output as rediscovery/validation rather than a missing-interaction discovery.")
    else:
        st.success("No direct interaction assertion was found in the connected graph; the result is treated as a missing-interaction candidate.")


def _render_explainability(payload: dict[str, Any]) -> None:
    st.markdown("## Explainability and model diagnostics")
    item = _selected_prediction(payload, selector_key="model_explanation")
    if not item:
        st.warning("No successful prediction is available to explain.")
        return
    local = pd.DataFrame(item.get("explainability", {}).get("local_component_contributions", []))
    if not local.empty:
        st.markdown("### Local component contribution")
        st.caption("Faithful leave-one-component-out explanation: each model score is replaced by its training-background median and the change in calibrated probability is measured.")
        chart = local.set_index("display_name")[["probability_change_when_replaced"]]
        st.bar_chart(chart)
        show_cols = [
            "display_name", "value", "background_median",
            "probability_change_when_replaced", "local_importance_share", "direction",
        ]
        st.dataframe(local[[c for c in show_cols if c in local.columns]], use_container_width=True, hide_index=True)

    tree_shap = item.get("explainability", {}).get("tree_shap", {}) or {}
    shap_values = pd.DataFrame(tree_shap.get("values", []) or [])
    if tree_shap.get("status") == "computed" and not shap_values.empty:
        st.markdown("### TreeSHAP explanation")
        st.caption("TreeSHAP explains the uncalibrated Extra Trees ensemble output. The calibrated-probability effect is shown separately in the local replacement analysis above.")
        st.bar_chart(shap_values.set_index("display_name")[["shap_value_raw_ensemble_output"]])
        st.dataframe(shap_values, use_container_width=True, hide_index=True)
    elif tree_shap:
        st.caption(f"TreeSHAP was not available for this run: {tree_shap.get('reason', 'unknown reason')}")

    global_importance = pd.DataFrame(item.get("explainability", {}).get("global_component_importance", []))
    if not global_importance.empty:
        st.markdown("### Global ensemble feature importance")
        global_importance = global_importance.copy()
        if "component_score" in global_importance:
            mapping = {
                "score__stage1_tabular_extra_trees": "Stage 1 Extra Trees",
                "score__stage3_rgcn_sampled": "Stage 3 R-GCN",
                "score__stage3_hgt_sampled": "Stage 3 HGT",
            }
            global_importance["component"] = global_importance["component_score"].map(mapping).fillna(global_importance["component_score"])
            st.bar_chart(global_importance.set_index("component")[["importance"]])
        st.dataframe(global_importance, use_container_width=True, hide_index=True)

    stage1_importance = pd.DataFrame(item.get("explainability", {}).get("stage1_structural_feature_importance", []))
    if not stage1_importance.empty:
        st.markdown("### Stage 1 structural feature importance")
        st.caption("Global Extra Trees importance for the leakage-safe FastRP pair features used by the structural baseline.")
        st.bar_chart(stage1_importance.set_index("feature")[["importance"]])
        st.dataframe(stage1_importance, use_container_width=True, hide_index=True)

    uncertainty = item.get("explainability", {}).get("uncertainty_metrics", {})
    calibration = item.get("explainability", {}).get("calibration_metrics_on_frozen_test", {})
    st.markdown("### Uncertainty and calibration")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Decision margin", _fmt_number(uncertainty.get("decision_margin"), 4))
    c2.metric("Model disagreement", _fmt_number(uncertainty.get("component_disagreement_std"), 4))
    c3.metric("Predictive entropy", f"{_fmt_number(uncertainty.get('predictive_entropy_bits'), 4)} bits")
    c4.metric("Brier score", _fmt_number(calibration.get("brier_score"), 4))
    c5.metric("Expected calibration error", _fmt_number(calibration.get("expected_calibration_error"), 4))

    with st.expander("Model validation metrics", expanded=False):
        global_metrics = item.get("validation", {}).get("global_test_metrics", {}) or {}
        st.dataframe(pd.DataFrame([global_metrics]), use_container_width=True, hide_index=True)
        target_metrics = item.get("validation", {}).get("target_specific_metrics")
        if target_metrics:
            st.markdown("**Target-specific test metrics**")
            st.dataframe(pd.DataFrame([target_metrics]), use_container_width=True, hide_index=True)


def _render_evidence(payload: dict[str, Any]) -> None:
    st.markdown("## Knowledge-graph evidence")
    item = _selected_prediction(payload, selector_key="evidence")
    if not item:
        st.warning("No successful prediction is available.")
        return
    evidence = item.get("evidence", {})
    st.markdown(f"### {evidence.get('tier', 'Evidence tier unavailable')}")
    st.write(evidence.get("tier_reason", "No evidence interpretation was returned."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Direct interaction", "Yes" if evidence.get("known_direct_interaction") else "No")
    c2.metric("Endpoint paths", int(evidence.get("endpoint_path_count", 0) or 0))
    c3.metric("Measure groups", int(evidence.get("measure_group_count", 0) or 0))

    direct = evidence.get("direct_interactions", []) or []
    if direct:
        st.markdown("### Direct interaction assertions")
        st.dataframe(pd.DataFrame(direct), use_container_width=True, hide_index=True)
    analogs = evidence.get("similar_compound_support", []) or []
    if analogs:
        st.markdown("### Similar-compound support")
        rows = []
        for item_row in analogs:
            rows.append({
                "cid": item_row.get("cid"),
                "name": item_row.get("name"),
                "similarity": item_row.get("similarity"),
                "interaction": str(item_row.get("interaction", {})),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if not direct and not analogs:
        st.info("No direct or similar-compound interaction support was found. This is a model-only hypothesis and should receive the strongest validation requirement.")


def _render_report(payload: dict[str, Any]) -> None:
    st.markdown("## Downloadable prediction report")
    report_html = generate_prediction_report_html(payload)
    summary = prediction_summary_dataframe(payload)
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download HTML report",
        data=report_html.encode("utf-8"),
        file_name="pring_interaction_prediction_report.html",
        mime="text/html",
        use_container_width=True,
    )
    c2.download_button(
        "Download prediction JSON",
        data=prediction_json_bytes(payload),
        file_name="pring_interaction_predictions.json",
        mime="application/json",
        use_container_width=True,
    )
    csv_bytes = summary.to_csv(index=False).encode("utf-8") if not summary.empty else b""
    c3.download_button(
        "Download summary CSV",
        data=csv_bytes,
        file_name="pring_interaction_prediction_summary.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=summary.empty,
    )
    st.caption("The preview below contains the same information as the downloadable report.")
    components.html(report_html, height=900, scrolling=True)


def display_prediction_workspace(payload: dict[str, Any]) -> None:
    """Render prediction outputs without replacing the application main-page design."""
    prediction_tab, explanation_tab, evidence_tab, report_tab = st.tabs(
        ["Prediction", "Model Explanation", "Evidence", "Download Report"]
    )
    with prediction_tab:
        _render_summary(payload)
    with explanation_tab:
        _render_explainability(payload)
    with evidence_tab:
        _render_evidence(payload)
    with report_tab:
        _render_report(payload)
