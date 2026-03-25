SYSTEM_SUMMARIZE_CASE = """
You are a clinical reviewer assistant for a health insurance prior
authorization team. Your job is to summarize prior authorization
requests clearly and objectively.

Rules:
- Be concise and factual
- Do not make clinical judgments
- Output valid JSON only — no markdown, no explanation outside JSON
- This output is decision support only, not a final coverage determination
"""

SYSTEM_EXTRACT_EVIDENCE = """
You are a clinical evidence extractor. Your job is to identify and
extract clinical facts from a patient note that are relevant to
prior authorization medical necessity criteria.

Rules:
- Extract only facts explicitly stated in the note
- Do not infer or assume information not present
- Output valid JSON only — no markdown, no explanation outside JSON
- This output is decision support only, not a final coverage determination
"""

SYSTEM_COMPARE_CRITERIA = """
You are a prior authorization criteria reviewer. Your job is to
compare extracted clinical evidence against policy criteria and
determine which criteria are met, not met, or unknown.

Rules:
- Base your assessment only on the provided evidence and policy text
- Mark criteria as unknown if the evidence does not address them
- Output valid JSON only — no markdown, no explanation outside JSON
- This output is decision support only, not a final coverage determination
"""

SYSTEM_RECOMMEND = """
You are a prior authorization recommendation engine. Based on
criteria comparison results, generate a structured recommendation.

Rules:
- Recommendation must be exactly one of: APPROVE, DENY, NEED_MORE_INFO
- If any required criterion is unknown or unmet, prefer NEED_MORE_INFO
  over DENY unless evidence clearly contradicts the criteria
- Cite retrieved policy chunks by their chunk_id
- Output valid JSON only — no markdown, no explanation outside JSON
- This output is decision support only, not a final coverage determination
- Always include: "AI-generated draft — human review required"
"""

SYSTEM_REVIEWER_NOTE = """
You are a clinical documentation assistant. Write a concise
structured reviewer note summarizing the prior auth decision.

Rules:
- Use professional clinical language
- Be objective and factual
- Output valid JSON only — no markdown, no explanation outside JSON
- Always end with: "AI-generated draft — human review required"
"""


def build_summarize_prompt(
    diagnosis: str,
    requested_service: str,
    patient_age: int,
    provider_specialty: str,
    clinical_note: str,
) -> str:
    """Build the user prompt for case summarization."""
    return f"""Summarize this prior authorization request.

Diagnosis: {diagnosis}
Requested service: {requested_service}
Patient age: {patient_age}
Provider specialty: {provider_specialty}
Clinical note: {clinical_note}

Return JSON with this exact structure:
{{
  "case_summary": "2-3 sentence summary of the case",
  "key_clinical_facts": ["fact 1", "fact 2", "fact 3"],
  "urgency_indicators": ["any red flags or urgent indicators, or empty list"]
}}"""


def build_extract_evidence_prompt(
    clinical_note: str,
    diagnosis: str,
    requested_service: str,
) -> str:
    """Build the user prompt for evidence extraction."""
    return f"""Extract clinical evidence relevant to prior authorization
from this note.

Diagnosis: {diagnosis}
Requested service: {requested_service}
Clinical note: {clinical_note}

Return JSON with this exact structure:
{{
  "symptom_duration": "documented duration or null",
  "conservative_treatments_tried": ["treatment 1", "treatment 2"],
  "clinical_findings": ["finding 1", "finding 2"],
  "lab_results": ["lab result 1", "lab result 2"],
  "specialist_involvement": "specialist noted or null",
  "contraindications_noted": ["contraindication 1", "or empty list"],
  "additional_relevant_facts": ["any other relevant facts"]
}}"""


def build_compare_criteria_prompt(
    extracted_evidence: str,
    retrieved_policy_chunks: str,
) -> str:
    """Build the user prompt for criteria comparison."""
    return f"""Compare the extracted clinical evidence against the
retrieved policy criteria.

Extracted evidence:
{extracted_evidence}

Retrieved policy criteria:
{retrieved_policy_chunks}

Return JSON with this exact structure:
{{
  "criteria_results": [
    {{
      "criterion": "criterion name",
      "status": "MET" or "NOT_MET" or "UNKNOWN",
      "evidence": "evidence supporting this assessment",
      "chunk_id": "the chunk_id this criterion came from"
    }}
  ],
  "overall_assessment": "brief overall assessment"
}}"""


def build_recommend_prompt(
    criteria_results: str,
    case_summary: str,
) -> str:
    """Build the user prompt for final recommendation."""
    return f"""Generate a prior authorization recommendation based on
the criteria comparison results.

Case summary: {case_summary}
Criteria results: {criteria_results}

Return JSON with this exact structure:
{{
  "decision": "APPROVE" or "DENY" or "NEED_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "rationale": "clear explanation of the decision",
  "missing_information": ["item needed 1", "item needed 2", or empty list],
  "citations": ["chunk_id_1", "chunk_id_2"],
  "disclaimer": "AI-generated draft — human review required"
}}"""


def build_reviewer_note_prompt(
    decision: str,
    rationale: str,
    criteria_results: str,
    missing_information: list[str],
) -> str:
    """Build the user prompt for reviewer note generation."""
    missing = (
        ", ".join(missing_information)
        if missing_information
        else "None"
    )
    return f"""Write a structured reviewer note for this prior
authorization decision.

Decision: {decision}
Rationale: {rationale}
Criteria results: {criteria_results}
Missing information: {missing}

Return JSON with this exact structure:
{{
  "reviewer_note": "professional clinical reviewer note 3-5 sentences",
  "disclaimer": "AI-generated draft — human review required"
}}"""
