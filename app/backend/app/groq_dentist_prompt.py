"""Patient-friendly ICDAS copy. Groq must not invent grades; this is the fallback lexicon."""

from __future__ import annotations

ICDAS_PLAIN = {
    "en": {
        0: {
            "stage": "Healthy tooth",
            "priority": "Low",
            "explanation": (
                "The AI did not find visible signs of tooth decay on this tooth. "
                "The outer surface looks healthy in the uploaded photograph."
            ),
            "meaning": (
                "This does not replace a dentist visit, but the screening model did not "
                "flag this tooth for decay-related changes."
            ),
            "next_step": "Continue your current oral hygiene habits and routine dental check-ups.",
            "table": "No visible signs of early tooth decay were identified by the AI model.",
        },
        1: {
            "stage": "Very early enamel change",
            "priority": "Low",
            "explanation": (
                "The tooth shows the earliest visible changes that may be associated with "
                "tooth decay. At this stage, the surface is still largely intact."
            ),
            "meaning": (
                "This is an early warning stage. The outer layer of the tooth may be starting "
                "to weaken, but that does not mean there is a hole."
            ),
            "next_step": (
                "Keep up careful brushing with fluoride toothpaste and ask a dentist to "
                "monitor this tooth."
            ),
            "table": "Very early enamel changes may be present.",
        },
        2: {
            "stage": "Early enamel decay",
            "priority": "Medium",
            "explanation": (
                "The AI detected clearer changes on the outer tooth layer than ICDAS 1. "
                "The surface appears to have early damage that deserves attention."
            ),
            "meaning": (
                "The tooth is not necessarily severely damaged, but it should not be ignored. "
                "This does not mean there is definitely a hole."
            ),
            "next_step": (
                "Maintain good oral hygiene and consider a dental examination so the tooth "
                "can be evaluated early."
            ),
            "table": "The AI identified changes consistent with an early enamel lesion.",
        },
        3: {
            "stage": "Moderate enamel damage",
            "priority": "High",
            "explanation": (
                "The outer protective layer of the tooth appears to have started breaking "
                "down in one area. This suggests the tooth may be progressing beyond the "
                "earliest stages."
            ),
            "meaning": (
                "A dentist should look at this tooth. Early professional evaluation may help "
                "prevent further problems."
            ),
            "next_step": "Schedule a dental examination and do not delay evaluation of this tooth.",
            "table": (
                "Findings suggest localized breakdown of the outer tooth layer that may "
                "require professional evaluation."
            ),
        },
        4: {
            "stage": "Advanced visible decay",
            "priority": "High",
            "explanation": (
                "The AI detected a darker appearance that may indicate deeper damage under "
                "the outer tooth surface. This stage suggests timely professional evaluation."
            ),
            "meaning": (
                "The uploaded image suggests a more advanced change. Only a dentist can "
                "confirm what is happening inside the tooth."
            ),
            "next_step": (
                "Please arrange a dental consultation. Further clinical examination, and "
                "possibly imaging, may be appropriate."
            ),
            "table": (
                "The AI identified a darker underlying shadow suggestive of a deeper change "
                "and recommends timely dental examination."
            ),
        },
    }
}

# Hindi / Kannada use the same clinical facts; wording is native-language patient copy.
ICDAS_PLAIN["hi"] = {
    0: {
        "stage": "स्वस्थ दाँत",
        "priority": "Low",
        "explanation": "एआई को इस दाँत पर सड़न के दिखाई देने वाले संकेत नहीं मिले। अपलोड की गई तस्वीर में बाहरी सतह स्वस्थ दिखती है।",
        "meaning": "यह दंत चिकित्सक की जगह नहीं लेता, लेकिन स्क्रीनिंग मॉडल ने इस दाँत पर सड़न-संबंधी बदलाव नहीं दिखाए।",
        "next_step": "अपनी मौजूदा सफाई की आदतें और नियमित दंत जाँच जारी रखें।",
        "table": "एआई मॉडल को शुरुआती सड़न के दिखाई देने वाले संकेत नहीं मिले।",
    },
    1: {
        "stage": "बहुत शुरुआती एनामेल बदलाव",
        "priority": "Low",
        "explanation": "दाँत पर सड़न से जुड़े सबसे शुरुआती दिखने वाले बदलाव हो सकते हैं। इस चरण में सतह अधिकतर बनी रहती है।",
        "meaning": "यह एक शुरुआती चेतावनी है। बाहरी परत कमज़ोर हो सकती है, लेकिन इसका मतलब छेद होना नहीं है।",
        "next_step": "फ्लोराइड टूथपेस्ट से सावधानी से ब्रश करें और दंत चिकित्सक से निगरानी करवाएँ।",
        "table": "बहुत शुरुआती एनामेल बदलाव हो सकते हैं।",
    },
    2: {
        "stage": "शुरुआती एनामेल क्षति",
        "priority": "Medium",
        "explanation": "ICDAS 1 से अधिक स्पष्ट बाहरी-सतह बदलाव दिखे। दाँत की बाहरी परत पर शुरुआती क्षति का संकेत है।",
        "meaning": "दाँत ज़रूरी नहीं कि गंभीर रूप से क्षतिग्रस्त हो, लेकिन इसे नज़रअंदाज़ न करें।",
        "next_step": "अच्छी सफाई जारी रखें और जल्द जाँच के लिए दंत चिकित्सक से मिलें।",
        "table": "एआई ने शुरुआती एनामेल परिवर्तन के अनुरूप संकेत पहचाने।",
    },
    3: {
        "stage": "मध्यम एनामेल क्षति",
        "priority": "High",
        "explanation": "दाँत की बाहरी सुरक्षा परत एक क्षेत्र में टूटती दिख सकती है। समस्या शुरुआती चरण से आगे बढ़ सकती है।",
        "meaning": "इस दाँत की दंत चिकित्सक से जाँच करानी चाहिए।",
        "next_step": "दंत जाँच का समय तय करें और इस दाँत का मूल्यांकन न टालें।",
        "table": "बाहरी परत के स्थानीय टूटने का संकेत है; पेशेवर जाँच उचित है।",
    },
    4: {
        "stage": "उन्नत दिखाई देने वाली क्षति",
        "priority": "High",
        "explanation": "एआई ने गहरा रंग देखा जो बाहरी सतह के नीचे अधिक क्षति का संकेत हो सकता है। समय पर पेशेवर जाँच चाहिए।",
        "meaning": "तस्वीर अधिक उन्नत बदलाव सुझाती है। पुष्टि केवल दंत चिकित्सक कर सकते हैं।",
        "next_step": "कृपया दंत परामर्श लें। अतिरिक्त जाँच या इमेजिंग उचित हो सकती है।",
        "table": "गहरे बदलाव का संकेत; समय पर दंत जाँच की सलाह।",
    },
}

ICDAS_PLAIN["kn"] = {
    0: {
        "stage": "ಆರೋಗ್ಯಕರ ಹಲ್ಲು",
        "priority": "Low",
        "explanation": "ಎಐ ಈ ಹಲ್ಲಿನಲ್ಲಿ ಕೊಳೆತದ ಗೋಚರ ಚಿಹ್ನೆಗಳನ್ನು ಕಂಡುಹಿಡಿಯಲಿಲ್ಲ. ಅಪ್‌ಲೋಡ್ ಫೋಟೋದಲ್ಲಿ ಹೊರಮೈ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ.",
        "meaning": "ಇದು ದಂತವೈದ್ಯರ ಪರೀಕ್ಷೆಯನ್ನು ಬದಲಿಸುವುದಿಲ್ಲ, ಆದರೆ ಸ್ಕ್ರೀನಿಂಗ್ ಮಾದರಿ ಈ ಹಲ್ಲನ್ನು ಗುರುತಿಸಲಿಲ್ಲ.",
        "next_step": "ಪ್ರಸ್ತುತ ಶುಚಿತ್ವ ಮತ್ತು ನಿಯಮಿತ ದಂತ ತಪಾಸಣೆಯನ್ನು ಮುಂದುವರಿಸಿ.",
        "table": "ಎಐ ಮಾದರಿ ಆರಂಭಿಕ ಕೊಳೆತದ ಗೋಚರ ಚಿಹ್ನೆಗಳನ್ನು ಗುರುತಿಸಲಿಲ್ಲ.",
    },
    1: {
        "stage": "ಬಹಳ ಆರಂಭಿಕ ಎನಾಮೆಲ್ ಬದಲಾವಣೆ",
        "priority": "Low",
        "explanation": "ಕೊಳೆತಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಅತ್ಯಂತ ಆರಂಭಿಕ ಗೋಚರ ಬದಲಾವಣೆಗಳು ಇರಬಹುದು. ಈ ಹಂತದಲ್ಲಿ ಮೇಲ್ಮೈ ಹೆಚ್ಚಾಗಿ ಉಳಿದಿದೆ.",
        "meaning": "ಇದು ಆರಂಭಿಕ ಎಚ್ಚರಿಕೆ. ಹೊರ ಪದರ ದುರ್ಬಲವಾಗಬಹುದು, ಆದರೆ ರಂಧ್ರ ಇದೆ ಎಂದು ಅರ್ಥವಲ್ಲ.",
        "next_step": "ಫ್ಲೋರೈಡ್ ಟೂತ್‌ಪೇಸ್ಟ್‌ನಿಂದ ಜಾಗರೂಕವಾಗಿ ಬ್ರಷ್ ಮಾಡಿ ಮತ್ತು ದಂತವೈದ್ಯರಿಂದ ಗಮನಿಸಿ.",
        "table": "ಬಹಳ ಆರಂಭಿಕ ಎನಾಮೆಲ್ ಬದಲಾವಣೆಗಳು ಇರಬಹುದು.",
    },
    2: {
        "stage": "ಆರಂಭಿಕ ಎನಾಮೆಲ್ ಕೊಳೆತ",
        "priority": "Medium",
        "explanation": "ICDAS 1 ಕ್ಕಿಂತ ಸ್ಪಷ್ಟವಾದ ಹೊರಮೈ ಬದಲಾವಣೆಗಳು ಕಂಡುಬಂದಿವೆ. ಗಮನ ಅಗತ್ಯವಿರುವ ಆರಂಭಿಕ ಹಾನಿ ಇರಬಹುದು.",
        "meaning": "ಹಲ್ಲು ತೀವ್ರವಾಗಿ ಹಾನಿಯಾಗಿದೆ ಎಂದು ಖಚಿತವಲ್ಲ, ಆದರೆ ನಿರ್ಲಕ್ಷಿಸಬಾರದು.",
        "next_step": "ಒಳ್ಳೆಯ ಶುಚಿತ್ವ ಇರಿಸಿ ಮತ್ತು ಬೇಗನೆ ದಂತ ಪರೀಕ್ಷೆ ಪರಿಗಣಿಸಿ.",
        "table": "ಆರಂಭಿಕ ಎನಾಮೆಲ್ ಬದಲಾವಣೆಗೆ ಹೊಂದುವ ಸಂಕೇತಗಳನ್ನು ಎಐ ಗುರುತಿಸಿದೆ.",
    },
    3: {
        "stage": "ಮಧ್ಯಮ ಎನಾಮೆಲ್ ಹಾನಿ",
        "priority": "High",
        "explanation": "ಹಲ್ಲಿನ ಹೊರ ರಕ್ಷಣಾ ಪದರ ಒಂದು ಭಾಗದಲ್ಲಿ ಒಡೆಯಲು ಪ್ರಾರಂಭಿಸಿದಂತೆ ಕಾಣುತ್ತದೆ. ಸಮಸ್ಯೆ ಆರಂಭಿಕ ಹಂತವನ್ನು ದಾಟಿರಬಹುದು.",
        "meaning": "ಈ ಹಲ್ಲನ್ನು ದಂತವೈದ್ಯರು ನೋಡಬೇಕು.",
        "next_step": "ದಂತ ಪರೀಕ್ಷೆಯನ್ನು ನಿಗದಿಪಡಿಸಿ; ಈ ಹಲ್ಲಿನ ಮೌಲ್ಯಮಾಪನವನ್ನು ವಿಳಂಬ ಮಾಡಬೇಡಿ.",
        "table": "ಹೊರಪದರದ ಸ್ಥಳೀಯ ಒಡಕಿನ ಸಂಕೇತ; ವೃತ್ತಿಪರ ಪರೀಕ್ಷೆ ಸೂಕ್ತ.",
    },
    4: {
        "stage": "ಮುಂದುವರಿದ ಗೋಚರ ಕೊಳೆತ",
        "priority": "High",
        "explanation": "ಹೊರಮೈ ಕೆಳಗೆ ಆಳವಾದ ಹಾನಿ ಇರಬಹುದೆಂದು ಸೂಚಿಸುವ ಗಾಢ ನೋಟವನ್ನು ಎಐ ಗುರುತಿಸಿದೆ. ತಕ್ಷಣ ವೃತ್ತಿಪರ ಪರೀಕ್ಷೆ ಬೇಕು.",
        "meaning": "ಫೋಟೋ ಹೆಚ್ಚು ಮುಂದುವರಿದ ಬದಲಾವಣೆಯನ್ನು ಸೂಚಿಸುತ್ತದೆ. ದಂತವೈದ್ಯರು ಮಾತ್ರ ಖಚಿತಪಡಿಸಬಹುದು.",
        "next_step": "ದಂತ ಸಲಹೆ ಪಡೆಯಿರಿ. ಹೆಚ್ಚುವರಿ ಪರೀಕ್ಷೆ ಅಥವಾ ಇಮೇಜಿಂಗ್ ಸೂಕ್ತವಾಗಿರಬಹುದು.",
        "table": "ಆಳವಾದ ಬದಲಾವಣೆಯ ಸಂಕೇತ; ಸಮಯೋಚಿತ ದಂತ ಪರೀಕ್ಷೆ ಶಿಫಾರಸು.",
    },
}

DISCLAIMER = (
    "This report is generated by an AI-assisted dental screening system. "
    "It is intended for educational and preliminary screening purposes only "
    "and does not replace a clinical examination by a qualified dental professional."
)

DISCLAIMER_HI = (
    "यह रिपोर्ट एआई-सहायता प्राप्त दंत स्क्रीनिंग प्रणाली द्वारा बनाई गई है। "
    "यह केवल शैक्षिक और प्रारंभिक स्क्रीनिंग के लिए है और योग्य दंत चिकित्सक की "
    "नैदानिक जाँच का विकल्प नहीं है।"
)

DISCLAIMER_KN = (
    "ಈ ವರದಿಯನ್ನು ಎಐ-ಸಹಾಯಿತ ದಂತ ಸ್ಕ್ರೀನಿಂಗ್ ವ್ಯವಸ್ಥೆ ರಚಿಸಿದೆ. "
    "ಇದು ಶೈಕ್ಷಣಿಕ ಮತ್ತು ಪ್ರಾಥಮಿಕ ಸ್ಕ್ರೀನಿಂಗ್‌ಗಾಗಿ ಮಾತ್ರ ಮತ್ತು ಅರ್ಹ ದಂತವೈದ್ಯರ "
    "ವೈದ್ಯಕೀಯ ಪರೀಕ್ಷೆಯನ್ನು ಬದಲಿಸುವುದಿಲ್ಲ."
)


SCREENING_SYSTEM_PROMPT = """
You are an AI assistant that generates a patient-friendly dental screening report
for CCC AI Dentist Camera 2.0.

You are NOT the diagnostic model. A computer-vision pipeline already produced
ICDAS 0–4 grades. You only explain those results in clear language.

STRICT RULES:
1. Never predict ICDAS grades yourself.
2. Never modify any ICDAS prediction.
3. Never invent teeth that were not detected.
4. Never invent diseases, cavities, infections, fractures, or treatments.
5. Never contradict the structured model output.
6. Never claim the patient definitely has a disease.
7. Clearly state this is AI-assisted screening.
8. Recommendations must stay general oral-health guidance unless supported by the supplied ICDAS results.
9. Preserve every ICDAS grade exactly as provided.
10. Never mention ICDAS 5 or 6.
11. Never assign FDI tooth numbers. Use the supplied tooth_id values only (for example T01).
12. Never say: you definitely have a cavity; you need a filling; you need root canal treatment; you have an infection.
13. If a tooth confidence is below 0.55 (or below 55 if the value is a percentage), keep the tooth and grade, and say the confidence is relatively low so the finding should be interpreted cautiously and confirmed through professional examination.
14. Do not prescribe medications or irreversible treatments.
15. Write for a first-time dental patient. Avoid unexplained words such as enamel demineralization, cavitation, dentin shadow, or lesion progression unless you immediately explain them in plain language.
16. Vary wording naturally between reports. Do not use identical paragraphs for identical inputs. Do not change facts.
17. Generate the entire report in the requested language (English, Hindi, or Kannada). Keep ICDAS grades numeric and tooth IDs unchanged.

TONE: calm, professional, friendly, easy for non-medical users. Do not use alarming language. Do not exaggerate.

OUTPUT: Return ONLY valid JSON with this shape:
{
  "markdown": "<full markdown report>"
}

The markdown report MUST use this structure:

# 🦷 CCC AI Dentist Camera 2.0 — AI Dental Screening Report

## Patient Information
- Patient ID, Name, Age, Visit Date (from JSON)

## AI Screening Summary
Image quality, teeth detected, teeth analyzed, ICDAS 0–4 distribution.
Say whether most teeth appear healthy or whether several teeth need attention. Do not exaggerate.

## Tooth-by-Tooth Findings
A markdown table:
| Tooth | ICDAS | Confidence | Explanation |
Explanations must follow the supplied ICDAS grade only.

Then, for EVERY tooth in the JSON, a block:

### Tooth {tooth_id}
**ICDAS Grade:** {n}
**Current Tooth Stage (Easy to Understand):** ...
**What This Means:** ...
**Confidence:** ...
**Suggested Next Step:** ...

ICDAS 0 = Healthy tooth.
ICDAS 1 = Very early enamel change.
ICDAS 2 = Early enamel decay.
ICDAS 3 = Moderate enamel damage.
ICDAS 4 = Advanced visible decay.

## Overall Oral Health Summary
Healthy count, early enamel changes, teeth needing closer examination, teeth needing prompt professional assessment.

## Understanding Your Overall Dental Health
Patient-friendly sentences such as: most teeth appear healthy; a few show very early changes; some show early enamel damage; one or more should be examined by a dentist.

## Visual Severity Summary
| Tooth | ICDAS | Current Stage | Priority |
Priority is only a communication label from ICDAS: 0–1 Low, 2 Medium, 3–4 High.

## Personalized Recommendations
Depend on supplied grades:
- Mostly 0: brushing, flossing, routine check-ups
- 1–2: monitor, hygiene, consider fluoride after consulting a dentist
- 3: schedule examination, do not delay, careful cleaning
- 4: prompt consultation; further examination and imaging may be appropriate

## What Should I Do Next?
Simple action list for healthy teeth, ICDAS 1–2, and ICDAS 3–4 if those grades exist.

## Preventive Oral Care Tips
Always 3–5 practical tips. Randomize wording naturally.

## Important AI Disclaimer
Always end with:
> This report is generated by an AI-assisted dental screening system. It is intended for educational and preliminary screening purposes only and does not replace a clinical examination by a qualified dental professional.

If the language is Hindi or Kannada, translate headings and explanations naturally, but keep that English disclaimer as an additional final block as well as a native-language equivalent.
""".strip()
