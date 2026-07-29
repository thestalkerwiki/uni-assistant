from bs4 import BeautifulSoup

from uni_assist.ingestion.structure_extractor import (
    extract_structured_blocks,
)


def test_extracts_description_list_pairs() -> None:
    html = """
    <html>
        <body>
            <main>
                <h1>Psychology</h1>

                <dl>
                    <dt>Study duration</dt>
                    <dd>
                        <p>6 semesters</p>
                    </dd>
                </dl>

                <dl>
                    <dt>Study scope</dt>
                    <dd>
                        <p>180 ECTS</p>
                    </dd>
                </dl>
            </main>
        </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")
    blocks = extract_structured_blocks(soup)

    label_value_pairs = {
        (block["label"], block["value"])
        for block in blocks
        if block.get("type") == "label_value"
    }

    ordinary_texts = {
        block.get("text")
        for block in blocks
        if block.get("type") != "label_value"
    }

    assert ("Study duration", "6 semesters") in label_value_pairs
    assert ("Study scope", "180 ECTS") in label_value_pairs

    assert "6 semesters" not in ordinary_texts
    assert "180 ECTS" not in ordinary_texts


def test_extracts_compact_inline_label_value_list() -> None:
    html = """
    <html>
        <body>
            <main>
                <h1>Programme information</h1>

                <ul>
                    <li>Duration of study: 6 semesters</li>
                    <li>ECTS credit points: 180</li>
                    <li>Academic degree: Bachelor of Science</li>
                    <li>Language of instruction: German</li>
                </ul>
            </main>
        </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")
    blocks = extract_structured_blocks(soup)

    label_value_pairs = {
        (block["label"], block["value"])
        for block in blocks
        if block.get("type") == "label_value"
    }

    ordinary_texts = {
        block.get("text")
        for block in blocks
        if block.get("type") != "label_value"
    }

    assert label_value_pairs == {
        ("Duration of study", "6 semesters"),
        ("ECTS credit points", "180"),
        ("Academic degree", "Bachelor of Science"),
        ("Language of instruction", "German"),
    }

    assert "Duration of study: 6 semesters" not in ordinary_texts
