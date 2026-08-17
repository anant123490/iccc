"""Groq service for AI-assisted dental report generation."""

from groq import Groq

from .config import get_settings


# ---------------------------------------------------------
# Load application settings
# ---------------------------------------------------------

settings = get_settings()


# ---------------------------------------------------------
# Create Groq client
# ---------------------------------------------------------

client = Groq(
    api_key=settings.groq_api_key
)


# ---------------------------------------------------------
# Generate dental report
# ---------------------------------------------------------

def generate_report(
    icdas_grade: int,
    confidence: float,
) -> str:
    """
    Generate an AI-assisted dental clinical-support report.

    The ICDAS grade and confidence come from the existing
    machine-learning model. Groq only generates the readable
    clinical-support report.
    """

    prompt = f"""
You are an AI assistant that generates dental
clinical-support reports.

A machine learning model has already analyzed
a dental image.

The model produced:

ICDAS Grade: {icdas_grade}
Confidence: {confidence:.2f}

Generate a concise and professional dental
clinical-support report.

Include:

1. ICDAS Grade
2. Severity interpretation
3. Clinical finding
4. Recommended next step
5. Clinical disclaimer

Important rules:

- Do NOT change the ICDAS grade provided by
  the machine learning model.
- Do NOT change the confidence value.
- Do NOT invent image findings that were not
  provided by the system.
- Do NOT claim that you personally examined
  the patient.
- Do NOT provide a definitive diagnosis.
- Clearly state that the report is AI-assisted.
- Clearly state that the report does not replace
  examination by a qualified dental professional.
- Keep the report concise and professional.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI dental report generation "
                    "assistant. Generate clear, concise, "
                    "professional clinical-support reports."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Test Groq independently
# ---------------------------------------------------------

if __name__ == "__main__":

    report = generate_report(
        icdas_grade=6,
        confidence=0.94,
    )

    print("\n==============================")
    print("AI DENTAL REPORT")
    print("==============================\n")

    print(report)