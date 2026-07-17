"""Language-neutral cleaning for text extracted from web pages."""

import re


def clean_extracted_text(text: str) -> str:
    """Normalize extracted text and remove adjacent duplicate lines."""

    if not text:
        return ""

    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
    )

    cleaned_lines: list[str] = []
    previous_line_key: str | None = None

    for raw_line in text.split("\n"):
        line = re.sub(
            r"[ \t\f\v]+",
            " ",
            raw_line,
        ).strip()

        if not line:
            continue

        line_key = line.casefold()

        if line_key == previous_line_key:
            continue

        cleaned_lines.append(line)
        previous_line_key = line_key

    return "\n".join(cleaned_lines)