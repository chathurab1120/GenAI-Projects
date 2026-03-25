from pydantic import BaseModel, Field


class PAReviewRequest(BaseModel):
    """
    Input schema for a prior authorization review request.
    All fields use synthetic data — no real PHI.
    """
    case_id: str = Field(
        description="Unique identifier for this case"
    )
    member_id: str | None = Field(
        default=None,
        description="Synthetic member identifier"
    )
    patient_age: int = Field(
        ge=0, le=120,
        description="Patient age in years"
    )
    diagnosis: str = Field(
        min_length=3,
        description="Primary diagnosis"
    )
    requested_service: str = Field(
        min_length=3,
        description="Service or procedure being requested"
    )
    provider_specialty: str | None = Field(
        default=None,
        description="Requesting provider specialty"
    )
    clinical_note_text: str = Field(
        min_length=10,
        description="Clinical note text supporting the request"
    )
    policy_name: str | None = Field(
        default=None,
        description="Optional specific policy to check against"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "case_id": "PA-2024-001",
                "member_id": "SYNTHETIC-MBR-1001",
                "patient_age": 52,
                "diagnosis": "Low back pain with radiculopathy",
                "requested_service": "MRI Lumbar Spine without contrast",
                "provider_specialty": "Orthopedic Surgery",
                "clinical_note_text": (
                    "Patient has 8 weeks of low back pain radiating "
                    "to the left leg. Completed 8 PT sessions and "
                    "naproxen 500mg BID. Positive straight leg raise."
                ),
                "policy_name": "mri_lumbar_spine_policy"
            }
        }
    }
