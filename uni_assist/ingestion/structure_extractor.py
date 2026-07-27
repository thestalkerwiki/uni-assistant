"""Structure-aware extraction of content blocks from HTML."""

from typing import Any, Optional


TARGET_TAGS = ["h1", "h2", "h3", "p", "li"]


def _is_inside_navigation(block: Any) -> bool:
    """Return True when a block belongs to navigation."""

    return (
        block.find_parent("nav") is not None
        or block.find_parent(attrs={"role": "navigation"}) is not None
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
            direct_heading is not None and direct_heading.find("button") is not None
        )

        has_navigation = (
            current.find("nav") is not None
            or current.find(attrs={"role": "navigation"}) is not None
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

    return href == "#top" and block_text == link_text


def _extract_description_list_pairs(
    description_list: Any,
) -> list[dict]:
    """
    Extract semantic label-value pairs from an HTML description list.

    Supported HTML structure:
    <dl>
        <dt>Study duration</dt>
        <dd>6 semesters</dd>
    </dl>
    """

    items = description_list.find_all(
        ["dt", "dd"],
        recursive=False,
    )

    if not items:
        return []

    pairs: list[dict] = []
    current_labels: list[str] = []
    definition_seen = False

    for item in items:
        text = item.get_text(" ", strip=True)

        if not text:
            continue

        if item.name == "dt":
            # A new dt after one or more dd starts a new pair.
            if definition_seen:
                current_labels = []
                definition_seen = False

            current_labels.append(text)
            continue

        if item.name == "dd" and current_labels:
            label = " / ".join(current_labels)
            value = text

            pairs.append(
                {
                    "type": "label_value",
                    "tag": "dl",
                    "label": label,
                    "value": value,
                    "text": f"{label}: {value}",
                }
            )

            definition_seen = True

    return pairs


def _parse_inline_label_value(
    text: str,
) -> Optional[dict]:
    """
    Parse one short string shaped like:

    Duration of study: 6 semesters
    """

    label, separator, value = text.partition(":")

    if not separator:
        return None

    label = label.strip()
    value = value.strip()

    if not label or not value:
        return None

    # Stay conservative: labels should be short field names,
    # not complete prose sentences.
    if len(label) > 80 or len(label.split()) > 10:
        return None

    if len(value) > 300:
        return None

    return {
        "type": "label_value",
        "tag": "li",
        "label": label,
        "value": value,
        "text": f"{label}: {value}",
    }


def _extract_inline_label_value_items(
    list_tag: Any,
) -> dict[int, dict]:
    """
    Extract label-value pairs from a compact one-level list.

    Example:
    <ul>
        <li>Duration: 6 semesters</li>
        <li>ECTS: 180</li>
        <li>Language: German</li>
    </ul>
    """

    if (
        list_tag.find_parent("nav") is not None
        or list_tag.find_parent(attrs={"role": "navigation"}) is not None
    ):
        return {}

    items = list_tag.find_all(
        "li",
        recursive=False,
    )

    # Require a compact cluster, not one isolated sentence.
    if len(items) < 3 or len(items) > 12:
        return {}

    parsed_items: dict[int, dict] = {}

    for item in items:
        # Nested lists represent a more complex structure.
        if item.find(["ul", "ol"]) is not None:
            return {}

        text = item.get_text(" ", strip=True)
        parsed = _parse_inline_label_value(text)

        # Treat the list as semantic only when every direct item
        # follows the same label-value pattern.
        if parsed is None:
            return {}

        parsed_items[id(item)] = parsed

    return parsed_items


def extract_structured_blocks(soup: Any) -> list[dict]:
    """
    Extract minimal content blocks from a BeautifulSoup document.

    The result preserves document order and excludes confidently
    identified navigation and structural duplication.
    """

    scope = soup.find("main") or soup.find("article") or soup

    description_lists = {
        id(description_list): _extract_description_list_pairs(description_list)
        for description_list in scope.find_all("dl")
    }

    semantic_description_lists = {
        list_id: pairs for list_id, pairs in description_lists.items() if pairs
    }

    inline_label_value_items: dict[int, dict] = {}

    for list_tag in scope.find_all(["ul", "ol"]):
        inline_label_value_items.update(_extract_inline_label_value_items(list_tag))

    candidates = scope.find_all(TARGET_TAGS + ["dl"])

    content_blocks: list[dict] = []

    for block in candidates:
        # Preserve a semantic description list as label-value pairs.
        if block.name == "dl":
            pairs = semantic_description_lists.get(
                id(block),
                [],
            )

            if pairs:
                content_blocks.extend(pairs)

            continue

        inline_pair = inline_label_value_items.get(id(block))

        if inline_pair is not None:
            content_blocks.append(inline_pair)
            continue

        # Skip p/li descendants already represented by a label-value pair.
        parent_description_list = block.find_parent("dl")

        if (
            parent_description_list is not None
            and id(parent_description_list) in semantic_description_lists
        ):
            continue
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
