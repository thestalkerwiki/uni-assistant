"""Build section-aware input for evidence extraction."""

from typing import Any


HEADING_LEVELS = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
}


def build_contextual_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add the current HTML heading path to every content block.

    Heading blocks update the current section context but are not
    returned as evidence candidates themselves.
    """

    active_headings: dict[int, str] = {}
    contextual_blocks: list[dict[str, Any]] = []

    for source_index, block in enumerate(blocks):
        text = str(block.get("text", "")).strip()

        if not text:
            continue

        tag = block.get("tag")
        heading_level = HEADING_LEVELS.get(tag)

        if heading_level is not None:
            active_headings[heading_level] = text

            # A new h2 closes the previous h3.
            # A new h1 closes the previous h2 and h3.
            deeper_levels = [
                level
                for level in active_headings
                if level > heading_level
            ]

            for level in deeper_levels:
                del active_headings[level]

            continue

        contextual_block = dict(block)

        contextual_block["source_index"] = source_index
        contextual_block["section_path"] = [
            active_headings[level]
            for level in sorted(active_headings)
        ]

        contextual_blocks.append(contextual_block)

    return contextual_blocks
