# Example Request — Multi-Page Web Assistant

This example shows how Uni Assistant can use several official university pages together.

## Endpoint

```text
POST /web-multi-assistant
```

## Use case

A prospective student wants to understand what they should prepare before applying to a Master's programme.

Instead of relying on one page, the assistant receives several official sources:

- a programme page
- an international admission page
- a deadline page

The assistant should combine these sources and return a structured answer.

## Request

```json
{
  "urls": [
    "https://www.tugraz.at/studium/studienangebot/masterstudien/computational-social-systems",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassung-von-internationalen-studienwerberinnen-und-werbern/ueberblick-zulassung-von-internationalen-studienwerberinnen-und-werbern",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassungsfristen"
  ],
  "question": "What should I prepare before applying to this Master's programme?"
}
```

## Expected behavior

The assistant should:

- load all three official pages
- clean noisy webpage text
- classify source types
- retrieve relevant chunks across all sources
- answer using official source context
- separate facts, missing information, and next steps
- include source URLs

## Debug trace example

```text
DEBUG web source type: program_page
DEBUG web source type: admission_page
DEBUG web source type: deadline_page
DEBUG web multi detected_intent: admission
DEBUG web multi answer_mode: guidance_plan
DEBUG web multi route: contextual_plan
```

## Example response structure

```json
{
  "mode": "contextual_plan",
  "question": "What should I prepare before applying to this Master's programme?",
  "document_facts": [
    "Programme: Computational Social Systems.",
    "Degree: Master of Science (MSc).",
    "Duration: 4 semesters.",
    "ECTS: 120.",
    "Language of instruction: English.",
    "International applicants must submit an application for admission.",
    "The deadline depends on the applicant category and semester."
  ],
  "missing_info": [
    "The applicant's exact previous degree must be checked for eligibility.",
    "Country-specific document requirements may depend on the applicant's country.",
    "Exact accepted language certificates may require checking the official admission information."
  ],
  "answer": "1. Check whether your previous degree is relevant for the programme.\n2. Prepare the admission application form and required documents.\n3. Check which deadline category applies to you.\n4. Prepare documents in the required language and format.\n5. General guidance: verify country-specific requirements before submitting your application.",
  "sources": [
    "https://www.tugraz.at/studium/studienangebot/masterstudien/computational-social-systems",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassung-von-internationalen-studienwerberinnen-und-werbern/ueberblick-zulassung-von-internationalen-studienwerberinnen-und-werbern",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassungsfristen"
  ]
}
```

## Notes

This example reflects the intended structure of the multi-page assistant response.

The current prototype is still under development. The next technical improvement is source-balanced retrieval, so that programme facts, admission facts, and deadline facts are selected more consistently from their respective source types.
