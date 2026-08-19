import os
import json

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured"
    )


# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# DEFAULT FINDINGS
# ---------------------------------------------------------

def get_default_finding(icdas_grade: int) -> str:

    findings = {
        0: "No visible evidence of dental caries",

        1: "Initial visual change in enamel",

        2: "Distinct visual change in enamel",

        3: "Localized enamel breakdown without visible dentin",

        4: "Underlying dark shadow indicating possible dentin involvement",

        5: "Distinct cavity with visible dentin",

        6: "Extensive distinct cavity with visible dentin",
    }

    return findings.get(
        icdas_grade,
        "Dental finding could not be determined",
    )


# ---------------------------------------------------------
# DEFAULT RECOMMENDATIONS
# ---------------------------------------------------------

def get_default_recommendation(
    icdas_grade: int,
) -> str:

    recommendations = {

        0:
        "Continue routine oral hygiene and preventive dental care",

        1:
        "Preventive dental care and monitoring are recommended",

        2:
        "Dental evaluation and preventive management are recommended",

        3:
        "Dental evaluation is recommended for appropriate management",

        4:
        "Dental evaluation is recommended to assess dentin involvement",

        5:
        "Prompt restorative treatment by a dentist",

        6:
        "Urgent dental evaluation and restorative treatment",
    }

    return recommendations.get(
        icdas_grade,
        "Consult a dentist for further evaluation",
    )


# ---------------------------------------------------------
# DEFAULT URGENCY
# ---------------------------------------------------------

def get_default_urgency(
    icdas_grade: int,
) -> str:

    if icdas_grade <= 1:
        return "LOW"

    elif icdas_grade == 2:
        return "MODERATE"

    elif icdas_grade in [3, 4]:
        return "HIGH"

    else:
        return "CRITICAL"


# ---------------------------------------------------------
# GENERATE REPORT
# ---------------------------------------------------------

def generate_report(
    icdas_grade: int,
    confidence: float,
) -> dict:

    prompt = f"""
You are an AI dental assistant.

An ICDAS machine-learning model produced this prediction:

ICDAS grade: {icdas_grade}
Confidence: {confidence:.1f}%

Provide a concise clinical summary.

Return ONLY valid JSON.

Do NOT return:
- HTML
- CSS
- Markdown
- code blocks
- <div>
- <p>
- <style>
- explanations outside JSON

Return exactly:

{{
    "finding": "...",
    "recommendation": "...",
    "urgency": "..."
}}

Urgency must be exactly one of:

LOW
MODERATE
HIGH
CRITICAL

ICDAS reference:

Grade 0:
No visible evidence of caries.

Grade 1:
Initial change in enamel.

Grade 2:
Distinct visual change in enamel.

Grade 3:
Localized enamel breakdown without visible dentin.

Grade 4:
Underlying dark shadow indicating dentin involvement.

Grade 5:
Distinct cavity with visible dentin.

Grade 6:
Extensive distinct cavity with visible dentin.

Keep the response concise.

Do not make a diagnosis beyond the provided ICDAS classification.
"""


    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dental AI assistant. "
                        "Return only valid JSON."
                    ),
                },

                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.2,

            max_tokens=300,
        )


        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


        # -------------------------------------------------
        # Remove accidental markdown fences
        # -------------------------------------------------

        if content.startswith("```"):

            content = content.replace(
                "```json",
                "",
            )

            content = content.replace(
                "```",
                "",
            )

            content = content.strip()


        # -------------------------------------------------
        # Convert JSON string → Python dictionary
        # -------------------------------------------------

        data = json.loads(content)


        # -------------------------------------------------
        # Validate fields
        # -------------------------------------------------

        finding = str(
            data.get(
                "finding",
                get_default_finding(icdas_grade),
            )
        )

        recommendation = str(
            data.get(
                "recommendation",
                get_default_recommendation(
                    icdas_grade
                ),
            )
        )

        urgency = str(
            data.get(
                "urgency",
                get_default_urgency(icdas_grade),
            )
        ).upper()


        # -------------------------------------------------
        # Validate urgency
        # -------------------------------------------------

        valid_urgencies = {
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        }

        if urgency not in valid_urgencies:

            urgency = get_default_urgency(
                icdas_grade
            )


        return {

            "finding": finding,

            "recommendation": recommendation,

            "urgency": urgency,
        }


    except Exception as e:

        print(
            f"Groq report generation failed: {e}"
        )


        # -------------------------------------------------
        # Safe fallback
        # -------------------------------------------------

        return {

            "finding": get_default_finding(
                icdas_grade
            ),

            "recommendation":
                get_default_recommendation(
                    icdas_grade
                ),

            "urgency":
                get_default_urgency(
                    icdas_grade
                ),
        }