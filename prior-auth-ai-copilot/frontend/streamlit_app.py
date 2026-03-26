import streamlit as st
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_CASES = {
    "Lumbar Spine MRI — Approval case": {
        "case_id": "PA-DEMO-001",
        "member_id": "SYNTHETIC-MBR-1001",
        "patient_age": 52,
        "diagnosis": "Low back pain with radiculopathy",
        "requested_service": "MRI Lumbar Spine without contrast",
        "provider_specialty": "Orthopedic Surgery",
        "clinical_note_text": (
            "Patient is a 52-year-old male presenting with 8 weeks of "
            "low back pain radiating down the left leg to the knee. "
            "Patient reports numbness in the left foot. Physical exam "
            "shows decreased sensation L4-L5 distribution and positive "
            "straight leg raise at 45 degrees on the left. Patient "
            "completed 6 weeks of physical therapy (8 sessions) and has "
            "been on naproxen 500mg BID for 8 weeks with minimal "
            "improvement."
        ),
        "policy_name": "mri_lumbar_spine_policy",
    },
    "Sleep Study — Need more info case": {
        "case_id": "PA-DEMO-002",
        "member_id": "SYNTHETIC-MBR-1002",
        "patient_age": 44,
        "diagnosis": "Suspected obstructive sleep apnea",
        "requested_service": "Polysomnography in-lab sleep study",
        "provider_specialty": "Pulmonology",
        "clinical_note_text": (
            "Patient reports loud snoring and daytime sleepiness. "
            "No Epworth Sleepiness Scale score documented. "
            "No prior home sleep apnea test attempted."
        ),
        "policy_name": "sleep_study_policy",
    },
    "Biologic Therapy — Approval case": {
        "case_id": "PA-DEMO-003",
        "member_id": "SYNTHETIC-MBR-1003",
        "patient_age": 38,
        "diagnosis": "Rheumatoid arthritis, seropositive",
        "requested_service": "Adalimumab (Humira) 40mg injection",
        "provider_specialty": "Rheumatology",
        "clinical_note_text": (
            "Patient is a 38-year-old female with confirmed seropositive "
            "rheumatoid arthritis for 3 years. Failed methotrexate 20mg "
            "weekly for 6 months and hydroxychloroquine 400mg for 4 months. "
            "Labs: positive anti-CCP, CRP 18 mg/L. TB screening negative, "
            "hepatitis B negative."
        ),
        "policy_name": "biologic_therapy_policy",
    },
}

DECISION_COLORS = {
    "APPROVE": "green",
    "DENY": "red",
    "NEED_MORE_INFO": "orange",
}

DECISION_ICONS = {
    "APPROVE": "APPROVED",
    "DENY": "DENIED",
    "NEED_MORE_INFO": "MORE INFO NEEDED",
}


def run_review(payload: dict) -> dict | None:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.workflows.graph import pa_review_graph
        result = pa_review_graph.invoke(payload)
        return {
            "case_id": payload["case_id"],
            "decision": result.get("decision", "NEED_MORE_INFO"),
            "confidence": result.get("confidence", 0.0),
            "case_summary": result.get("case_summary", ""),
            "rationale": result.get("rationale", ""),
            "missing_information": result.get("missing_information", []),
            "criteria_results": result.get("criteria_results", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "reviewer_note": result.get("reviewer_note", ""),
            "citations": result.get("citations", []),
            "disclaimer": result.get(
                "disclaimer",
                "AI-generated draft — human review required"
            ),
            "prompt_tokens_total": result.get("prompt_tokens_total", 0),
            "completion_tokens_total": result.get(
                "completion_tokens_total", 0
            ),
        }
    except Exception as e:
        st.error(f"Review failed: {e}")
        return None


def render_decision_badge(decision: str) -> None:
    color = DECISION_COLORS.get(decision, "grey")
    label = DECISION_ICONS.get(decision, decision)
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {color};
            color: white;
            padding: 10px 28px;
            border-radius: 6px;
            font-size: 20px;
            font-weight: bold;
            letter-spacing: 1px;
            margin: 8px 0 16px 0;
        ">{label}</div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence: float) -> None:
    pct = int(confidence * 100)
    color = (
        "green" if confidence >= 0.8
        else "orange" if confidence >= 0.5
        else "red"
    )
    st.markdown(
        f"""
        <div style="margin-bottom:12px">
          <span style="font-size:13px;color:#666">
            Confidence: {pct}%
          </span>
          <div style="
            background:#eee;border-radius:4px;
            height:8px;margin-top:4px
          ">
            <div style="
              width:{pct}%;background:{color};
              height:8px;border-radius:4px
            "></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PA-Genie | Prior Auth AI Copilot",
    page_icon="",
    layout="wide",
)

st.title("PA-Genie: Prior Authorization AI Copilot")
st.caption(
    "GenAI-powered prior authorization review using RAG + LangGraph. "
    "Synthetic demo data only."
)

st.divider()

# ── Sidebar — sample cases ────────────────────────────────
with st.sidebar:
    st.header("Sample Cases")
    st.caption("Select a pre-built case to auto-fill the form.")
    selected_sample = st.selectbox(
        "Choose a sample case",
        options=["-- Manual entry --"] + list(SAMPLE_CASES.keys()),
    )
    st.divider()
    st.caption(
        "This is a synthetic demo. "
        "No real patient data is used. "
        "All AI outputs require human review."
    )

# ── Pre-fill from sample if selected ─────────────────────
prefill = {}
if selected_sample != "-- Manual entry --":
    prefill = SAMPLE_CASES[selected_sample]

# ── Input form ───────────────────────────────────────────
st.subheader("Prior Authorization Request")

col1, col2 = st.columns(2)

with col1:
    case_id = st.text_input(
        "Case ID",
        value=prefill.get("case_id", ""),
    )
    patient_age = st.number_input(
        "Patient Age",
        min_value=0, max_value=120,
        value=prefill.get("patient_age", 0),
    )
    diagnosis = st.text_input(
        "Diagnosis",
        value=prefill.get("diagnosis", ""),
    )

with col2:
    member_id = st.text_input(
        "Member ID (synthetic)",
        value=prefill.get("member_id", ""),
    )
    provider_specialty = st.text_input(
        "Provider Specialty",
        value=prefill.get("provider_specialty", ""),
    )
    requested_service = st.text_input(
        "Requested Service",
        value=prefill.get("requested_service", ""),
    )

clinical_note = st.text_area(
    "Clinical Note",
    value=prefill.get("clinical_note_text", ""),
    height=160,
)

run_btn = st.button(
    "Run Prior Auth Review",
    type="primary",
)

# ── Run review ───────────────────────────────────────────
if run_btn:
    if not all([case_id, diagnosis, requested_service, clinical_note]):
        st.warning("Please fill in Case ID, Diagnosis, Requested Service, and Clinical Note.")
    else:
        with st.spinner("Running AI review... this takes ~20 seconds"):
            payload = {
                "case_id": case_id,
                "member_id": member_id or None,
                "patient_age": int(patient_age),
                "diagnosis": diagnosis,
                "requested_service": requested_service,
                "provider_specialty": provider_specialty or None,
                "clinical_note_text": clinical_note,
                "policy_name": prefill.get("policy_name", None),
            }
            result = run_review(payload)

        if result:
            st.divider()
            st.subheader("Review Result")

            # Decision badge and confidence
            render_decision_badge(result["decision"])
            render_confidence_bar(result["confidence"])

            # Case summary
            st.markdown("**Case Summary**")
            st.info(result["case_summary"])

            # Rationale
            st.markdown("**Rationale**")
            st.write(result["rationale"])

            # Missing information
            if result["missing_information"]:
                st.markdown("**Missing Information Required**")
                for item in result["missing_information"]:
                    st.warning(item)

            # Criteria checklist
            if result["criteria_results"]:
                st.markdown("**Medical Necessity Criteria**")
                for c in result["criteria_results"]:
                    status = c["status"]
                    icon = (
                        "MET" if status == "MET"
                        else "NOT MET" if status == "NOT_MET"
                        else "UNKNOWN"
                    )
                    color = (
                        "green" if status == "MET"
                        else "red" if status == "NOT_MET"
                        else "orange"
                    )
                    st.markdown(
                        f":{color}[**{icon}**] {c['criterion']} — "
                        f"_{c['evidence']}_"
                    )

            # Reviewer note
            st.markdown("**Reviewer Note**")
            st.write(result["reviewer_note"])

            # Citations
            if result["citations"]:
                st.markdown("**Policy Citations**")
                for cit in result["citations"]:
                    st.code(cit)

            # Retrieved policy chunks
            with st.expander("View retrieved policy chunks"):
                for chunk in result["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['source_file']}** "
                        f"(score: {chunk['similarity_score']:.3f})"
                    )
                    st.text(chunk["content"][:400])
                    st.divider()

            # Token usage
            total_tokens = (
                result["prompt_tokens_total"]
                + result["completion_tokens_total"]
            )
            st.caption(f"Total tokens used: {total_tokens}")

            # Disclaimer
            st.divider()
            st.error(
                result.get(
                    "disclaimer",
                    "AI-generated draft — human review required"
                )
            )

# ── Footer ───────────────────────────────────────────────
st.divider()
st.caption(
    "PA-Genie v0.1.0 | Synthetic demo only | "
    "Not for real clinical use | "
    "All AI outputs require human review before any coverage decision"
)
