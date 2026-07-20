"""Structure-aware extraction of content blocks from HTML."""

from typing import Any


TARGET_TAGS = ["h1", "h2", "h3", "p", "li"]


def _is_inside_navigation(block: Any) -> bool:
    """Return True when a block belongs to navigation."""

    return (
        block.find_parent("nav") is not None
        or block.find_parent(
            attrs={"role": "navigation"}
        ) is not None
    )


def _is_inside_toc_component(block: Any) -> bool:
    """
    Detect a local table-of-contents component.

    Expected general structure:
    container
    ├── direct heading containing a button
    └── descendant navigation element
    """

    current = block.parent

    while current is not None:
        if current.name in {"main", "body"}:
            return False

        direct_heading = current.find(
            ["h1", "h2", "h3"],
            recursive=False,
        )

        has_toggle_heading = (
            direct_heading is not None
            and direct_heading.find("button") is not None
        )

        has_navigation = (
            current.find("nav") is not None
            or current.find(
                attrs={"role": "navigation"}
            ) is not None
        )

        if has_toggle_heading and has_navigation:
            return True

        current = current.parent

    return False


def _is_back_to_top_control(block: Any) -> bool:
    """Detect a block whose only purpose is linking to page top."""

    link = block.find("a", href=True)

    if link is None:
        return False

    href = str(link.get("href", "")).strip().lower()

    block_text = block.get_text(" ", strip=True)
    link_text = link.get_text(" ", strip=True)

    return (
        href == "#top"
        and block_text == link_text
    )


def extract_structured_blocks(soup: Any) -> list[dict]:
    """
    Extract minimal content blocks from a BeautifulSoup document.

    The result preserves document order and excludes confidently
    identified navigation and structural duplication.
    """

    scope = (
        soup.find("main")
        or soup.find("article")
        or soup
    )

    candidates = scope.find_all(TARGET_TAGS)
    content_blocks: list[dict] = []

    for block in candidates:
        # A parent containing another target block would duplicate it.
        if block.find(TARGET_TAGS) is not None:
            continue

        if _is_inside_navigation(block):
            continue

        if _is_inside_toc_component(block):
            continue

        if _is_back_to_top_control(block):
            continue

        text = block.get_text(" ", strip=True)

        if not text:
            continue

        content_blocks.append(
            {
                "tag": block.name,
                "text": text,
            }
        )

    return content_blocks