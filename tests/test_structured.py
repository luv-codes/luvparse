"""Structure view: headings, inline formatting, furniture tags, spanning tables."""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf4llm")

from luvparse.structured import (  # noqa: E402
    StructuredConfig,
    apply_extras,
    clean_text,
    extract_table_html,
    printed_page_number,
    tag_page_number,
    structure_pdf,
    token_recall,
)


def _pdf(tmp_path, build):
    doc = pymupdf.open()
    page = doc.new_page()
    build(page)
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    return path


def _report(page):
    page.insert_text((72, 30), "ACME Annual Report", fontsize=8)
    page.insert_text((72, 80), "Quarterly Results", fontsize=20, fontname="hebo")
    page.insert_text((72, 120), "Revenue grew strongly in the period.", fontsize=11)
    page.insert_text((72, 140), "This is ", fontsize=11)
    page.insert_text((115, 140), "important", fontsize=11, fontname="hebo")
    page.insert_text((72, 800), "Page 1 of 3", fontsize=8)


def test_headings_and_furniture_tags(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _report))
    assert "<page_header>\nACME Annual Report\n</page_header>" in page.markdown
    assert "<page_footer>\nPage 1 of 3\n</page_footer>" in page.markdown
    assert page.markdown.lstrip().splitlines()[0].startswith("<page_header>")
    assert "# " in page.markdown and "Quarterly Results" in page.markdown
    assert page.header == "ACME Annual Report"
    assert page.printed_page_number == "1"


def test_tags_can_be_disabled(tmp_path):
    (page,) = structure_pdf(
        _pdf(tmp_path, _report), config=StructuredConfig(emit_page_tags=False)
    )
    assert "<page_header>" not in page.markdown
    assert page.header == "ACME Annual Report"


def test_layout_items_point_into_markdown(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _report))
    assert page.items
    for item in page.items:
        assert page.markdown[item["start_index"] : item["end_index"]] == item["md"]
        box = item["bbox"]
        assert 0 <= box["x"] <= 1 and 0 <= box["y"] <= 1 and box["w"] > 0 and box["h"] > 0


def _sup_sub(page):
    page.insert_text((72, 80), "Chemistry notes", fontsize=20, fontname="hebo")
    for i in range(6):
        page.insert_text((72, 120 + 16 * i), "The quick brown fox jumps over the lazy dog again and again.", fontsize=11)
        page.insert_text((72, 420 + 16 * i), "Further body text continues down the page in this paragraph.", fontsize=11)
    page.insert_text((72, 300), "Water is H", fontsize=11)
    width = pymupdf.get_text_length("Water is H", fontsize=11)
    page.insert_text((72 + width, 303), "2", fontsize=7)
    w2 = pymupdf.get_text_length("2", fontsize=7)
    page.insert_text((72 + width + w2, 300), "O and E = mc", fontsize=11)
    w3 = pymupdf.get_text_length("O and E = mc", fontsize=11)
    page.insert_text((72 + width + w2 + w3, 294), "2", fontsize=7)


def test_superscript_and_subscript(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _sup_sub))
    assert "H<sub>2</sub>O" in page.markdown
    assert "mc<sup>2</sup>" in page.markdown


def test_inline_extras_off(tmp_path):
    (page,) = structure_pdf(
        _pdf(tmp_path, _sup_sub), config=StructuredConfig(inline_extras=False)
    )
    assert "<sub>" not in page.markdown and "<sup>" not in page.markdown


def _merged_table(page):
    xs, ys = [72, 162, 252, 342, 432], [110, 134, 158, 182]
    page.insert_text((76, 125), "Region", fontsize=10, fontname="hebo")
    page.insert_text((166, 125), "Revenue", fontsize=10, fontname="hebo")
    page.insert_text((346, 125), "Total", fontsize=10, fontname="hebo")
    page.insert_text((166, 149), "Q1", fontsize=10, fontname="hebo")
    page.insert_text((256, 149), "Q2", fontsize=10, fontname="hebo")
    for col, text in enumerate(["North", "10", "12", "22"]):
        page.insert_text((76 + 90 * col, 173), text, fontsize=10)
    page.draw_rect(pymupdf.Rect(72, 110, 162, 158))
    page.draw_rect(pymupdf.Rect(162, 110, 342, 134))
    page.draw_rect(pymupdf.Rect(342, 110, 432, 158))
    page.draw_rect(pymupdf.Rect(162, 134, 252, 158))
    page.draw_rect(pymupdf.Rect(252, 134, 342, 158))
    for col in range(4):
        page.draw_rect(pymupdf.Rect(xs[col], 158, xs[col + 1], 182))


def test_merged_header_table_keeps_spans(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _merged_table))
    assert 'rowspan="2"' in page.markdown
    assert 'colspan="2"' in page.markdown
    assert "<th" in page.markdown and "<td>North</td>" in page.markdown
    assert token_recall("Region Revenue Total Q1 Q2 North 10 12 22", page.markdown) == 1.0


def test_own_tables_off_never_loses_text(tmp_path):
    (page,) = structure_pdf(
        _pdf(tmp_path, _merged_table), config=StructuredConfig(own_tables=False)
    )
    assert token_recall("Region Revenue Total Q1 Q2 North 10 12 22", page.markdown) == 1.0


def test_empty_page_does_not_crash(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, lambda p: None))
    assert page.markdown.strip() == ""


def test_clean_text():
    assert clean_text("eﬃcient ﬁnal") == "efficient final"
    assert clean_text("inter-\nnational") == "international"
    assert clean_text("well-\nKnown") == "well-\nKnown"
    assert clean_text("a­b c") == "ab c"


def test_apply_extras_is_conservative():
    out = apply_extras(
        "Water is H2O", [{"kind": "sub", "text": "2", "left": "H", "right": "O", "line_y": 0}]
    )
    assert out == "Water is H<sub>2</sub>O"
    assert apply_extras("nothing here", [{"kind": "sup", "text": "9", "left": "x", "right": "", "line_y": 0}]) == "nothing here"


def test_printed_page_number():
    assert printed_page_number("Report", "Page 4 of 9") == "4"
    assert printed_page_number("Report 2024", "Confidential") == ""


def test_extract_table_html_from_markdown_reply():
    html = extract_table_html("| Year | Sales |\n|---|---|\n| 2023 | 10 |\n| 2024 | 12 |")
    assert html.startswith("<table>") and "<th>Year</th>" in html and "<td>12</td>" in html
    assert extract_table_html("no table") == ""


def test_page_number_tag_matches_annotation_style():
    assert tag_page_number("ID 1997 \u2014 SV \u2014 1\nbody") == "ID 1997 \u2014 SV \u2014 <page_number>1</page_number>\nbody"
    assert tag_page_number("Ed. (6-11) Page 9 of 10") == "Ed. (6-11) Page 9 of 10"
    assert tag_page_number("7") == "<page_number>7</page_number>"
    assert tag_page_number("Total 2.5") == "Total 2.5"


def _bold_header(page):
    page.insert_text((72, 40), "RULES MANUAL", fontsize=9, fontname="hebo")
    page.insert_text((72, 80), "Rule 10.", fontsize=10, fontname="hebo")
    page.insert_text((170, 80), "Schedule Rating Plan", fontsize=10, fontname="hebo")
    page.draw_rect(pymupdf.Rect(72, 66, 160, 90))
    page.draw_rect(pymupdf.Rect(160, 66, 360, 90))
    for i in range(8):
        page.insert_text((72, 140 + 16 * i), "Body text for the paragraph that follows the rule heading.", fontsize=11)


def test_header_keeps_bold_and_bold_row_table_becomes_title_line(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _bold_header))
    assert "**RULES MANUAL**" in page.header
    assert "**RULES MANUAL**" in page.markdown.splitlines()
    assert "**Rule 10. Schedule Rating Plan**" in page.markdown.splitlines()
    (plain,) = structure_pdf(
        _pdf(tmp_path, _bold_header), config=StructuredConfig(bold_row_tables=False)
    )
    assert "**Rule 10. Schedule Rating Plan**" not in plain.markdown


def _furniture_page(page):
    page.insert_text((72, 24), "NATIONAL UNION FIRE", fontsize=8, fontname="hebo")
    page.insert_text((72, 34), "CALIFORNIA MANUAL RULES", fontsize=8, fontname="hebo")
    page.insert_text((72, 80), "Quarterly Results", fontsize=20, fontname="hebo")
    for i in range(10):
        page.insert_text((72, 140 + 16 * i), "Body text that continues across this page for the test.", fontsize=11)
    left = "Ed. ("
    x = 72
    page.insert_text((x, 800), left, fontsize=8)
    x += pymupdf.get_text_length(left, fontsize=8)
    width = pymupdf.get_text_length("6-11", fontsize=8)
    page.draw_rect(pymupdf.Rect(x, 793, x + width, 802), color=None, fill=(1, 1, 0))
    page.insert_text((x, 800), "6-11", fontsize=8)
    x += width
    page.insert_text((x, 800), ") Page 9 of 10", fontsize=8)


def test_header_lines_stay_separate_and_footer_highlight_is_mark(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _furniture_page))
    assert page.header.splitlines() == ["**NATIONAL UNION FIRE**", "**CALIFORNIA MANUAL RULES**"]
    assert page.footer == "Ed. (<mark>6-11</mark>) Page 9 of 10"
    assert "<page_footer>\nEd. (<mark>6-11</mark>) Page 9 of 10\n</page_footer>" in page.markdown


def _labelled_heading(page):
    page.insert_text((72, 80), "B", fontsize=16, fontname="hebo")
    page.insert_text((100, 80), "DECISION OF THE COMMISSION", fontsize=16, fontname="hebo")
    for i in range(10):
        page.insert_text((72, 140 + 16 * i), "Body text that continues across this page for the test.", fontsize=11)


def test_label_and_heading_on_one_line_become_one_title(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _labelled_heading))
    titles = [ln for ln in page.markdown.splitlines() if ln.startswith("#")]
    assert titles and "B DECISION OF THE COMMISSION" in titles[0]


def _ruled_table_with_sub(page):
    page.insert_text((76, 125), "PIN", fontsize=10, fontname="hebo")
    page.insert_text((176, 125), "FORMAT", fontsize=10, fontname="hebo")
    page.insert_text((76, 149), "V", fontsize=10)
    page.insert_text((76 + pymupdf.get_text_length("V", fontsize=10), 151), "DD", fontsize=6)
    page.insert_text((176, 149), "Binary", fontsize=10)
    page.insert_text((76, 173), "0", fontsize=10)
    page.insert_text((176, 173), "Offset", fontsize=10)
    for r in range(3):
        for c in range(2):
            page.draw_rect(pymupdf.Rect(72 + 100 * c, 110 + 24 * r, 172 + 100 * c, 134 + 24 * r))


def test_subscript_inside_table_cell(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _ruled_table_with_sub))
    assert "<td>V<sub>DD</sub></td>" in page.markdown


def _unruled_rows(page):
    # One bordered block, three printed lines per cell, no gridline between them.
    for c, head in enumerate(["Class", "Value"]):
        page.insert_text((76 + 100 * c, 125), head, fontsize=10, fontname="hebo")
    for k in range(3):
        page.insert_text((76, 150 + 14 * k), f"80{k}", fontsize=10)
        page.insert_text((176, 150 + 14 * k), f"1.{k}", fontsize=10)
    for c in range(2):
        page.draw_rect(pymupdf.Rect(72 + 100 * c, 110, 172 + 100 * c, 134))
        page.draw_rect(pymupdf.Rect(72 + 100 * c, 134, 172 + 100 * c, 200))


def test_rows_without_gridlines_are_split_per_line(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _unruled_rows))
    assert page.markdown.count("<tr>") == 4
    assert "<td>800</td><td>1.0</td>" in page.markdown
    assert "<td>802</td><td>1.2</td>" in page.markdown


def _borderless(page):
    page.insert_text((72, 80), "Supporting Policy Form", fontsize=14, fontname="hebo")
    rows = [("ICC18 ENT-06", "Single Premium Annuity Contract"), ("ICC18 DP", "Data Page"),
            ("ICC18 DP-A", "Data Page"), ("ICC18 DP-B", "Data Page"), ("ICC22 EE.48", "Rate Endorsement")]
    for i, (a, b) in enumerate(rows):
        page.insert_text((72, 120 + 22 * i), a, fontsize=10, fontname="hebo")
        page.insert_text((250, 120 + 22 * i), b, fontsize=10, fontname="hebo")


def test_borderless_table_is_found(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _borderless))
    assert "<table>" in page.markdown and "ICC18 DP-A" in page.markdown and "Data Page" in page.markdown
    (off,) = structure_pdf(_pdf(tmp_path, _borderless), config=StructuredConfig(text_tables=False))
    assert "<table>" not in off.markdown


def test_prose_is_not_mistaken_for_a_table(tmp_path):
    def prose(page):
        for i in range(25):
            page.insert_text((72, 100 + 14 * i), "Plain prose runs across the page as ordinary sentences do in a report.", fontsize=11)

    (page,) = structure_pdf(_pdf(tmp_path, prose))
    assert "<table>" not in page.markdown


class FakeVision:
    def __init__(self):
        self.calls = []

    def describe(self, image, *, prompt_type="figure", **_):
        self.calls.append(("describe", prompt_type))
        return "<table><tr><th>Year</th><th>Sales</th></tr><tr><td>2023</td><td>10</td></tr></table>"

    def ocr(self, image, *, prompt_type="default", **_):
        self.calls.append(("ocr", prompt_type))
        if prompt_type == "furniture":
            return ""
        return "<page_header>\nSECRET\n</page_header>\n\n# **Memo**\n\nSome <u>underlined</u> words and ~~struck~~ ones."


def _png_bytes():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def _chart_page(page):
    page.insert_text((72, 60), "Title of the report page", fontsize=18, fontname="hebo")
    for i in range(6):
        page.insert_text((72, 100 + 16 * i), "Body text that continues across this page for the test.", fontsize=11)
    page.insert_image(pymupdf.Rect(72, 220, 372, 420), stream=_png_bytes())
    for i in range(6):
        page.insert_text((72, 460 + 16 * i), "More body text after the figure on this page of the report.", fontsize=11)


def test_picture_box_comes_from_the_pdf_image(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _chart_page))
    pictures = [i for i in page.items if i["label"] in ("picture", "image")]
    assert pictures
    box = pictures[0]["bbox"]
    # the image keeps its aspect ratio inside the rectangle given to insert_image
    assert abs(box["x"] - 88.67 / 595) < 0.01 and abs(box["w"] - 266.7 / 595) < 0.02


def test_chart_picture_becomes_table_with_vision(tmp_path):
    fake = FakeVision()
    (page,) = structure_pdf(
        _pdf(tmp_path, _chart_page), config=StructuredConfig(chart_tables=True), vision=fake
    )
    assert ("describe", "chart_table") in fake.calls
    assert "<th>Year</th>" in page.markdown and page.charts == 1


def test_small_pictures_do_not_cost_a_vision_call(tmp_path):
    def icon(page):
        _chart_page(page)
        page.insert_image(pymupdf.Rect(450, 100, 470, 118), stream=_png_bytes())

    fake = FakeVision()
    structure_pdf(_pdf(tmp_path, icon), config=StructuredConfig(chart_tables=True), vision=fake)
    assert len([c for c in fake.calls if c[0] == "describe"]) == 1


def test_scanned_page_is_read_by_vision_with_formatting(tmp_path):
    def scan(page):
        page.insert_image(page.rect, stream=_png_bytes())

    fake = FakeVision()
    (page,) = structure_pdf(
        _pdf(tmp_path, scan), config=StructuredConfig(ocr_scanned=True), vision=fake
    )
    assert ("ocr", "markdown") in fake.calls
    assert "<u>underlined</u>" in page.markdown and "~~struck~~" in page.markdown
    assert page.header == "SECRET"


def test_sideways_margin_text_is_a_page_header(tmp_path):
    def stamp(page):
        _chart_page(page)
        page.insert_text((20, 400), "arXiv:2602.19961v1 [cs.CL] 23 Feb 2026", fontsize=9, rotate=90)

    (page,) = structure_pdf(_pdf(tmp_path, stamp))
    assert any(i["label"] == "page-header" and "arXiv" in plain_text_of(i) for i in page.items)


def plain_text_of(item):
    from luvparse.structured import plain_text

    return plain_text(item["md"])


def test_scan_with_hidden_text_layer_still_goes_to_vision(tmp_path):
    def scan(page):
        page.insert_image(page.rect, stream=_png_bytes(), keep_proportion=False)
        page.insert_text((72, 100), "hidden ocr words here", fontsize=11, render_mode=3)

    fake = FakeVision()
    (page,) = structure_pdf(_pdf(tmp_path, scan), config=StructuredConfig(ocr_scanned=True), vision=fake)
    assert ("ocr", "markdown") in fake.calls and "<u>underlined</u>" in page.markdown


def test_footer_is_plain_but_header_keeps_bold(tmp_path):
    def build(page):
        _bold_header(page)
        page.insert_text((72, 800), "Ed. (6-11) Page 9 of 10", fontsize=8, fontname="hebo")

    (page,) = structure_pdf(_pdf(tmp_path, build))
    assert page.header == "**RULES MANUAL**"
    assert page.footer == "Ed. (6-11) Page 9 of 10"


def _two_block_table(page):
    page.insert_text((72, 60), "Supporting Policy Form:", fontsize=12, fontname="hebo")
    rows = [("ICC18 ENT-06", "Single Premium Annuity Contract with Market"), ("ICC18 DP", "Data Page"),
            ("ICC18 DP-A", "Data Page"), ("ICC18 DP-B", "Data Page"), ("ICC22 EE.48", "Rate Endorsement")]
    y = 90
    for i, (a, b) in enumerate(rows):
        page.insert_text((72, y), a, fontsize=10, fontname="hebo")
        page.insert_text((250, y), b, fontsize=10, fontname="hebo")
        y += 16
        if i == 0:  # the first entry wraps onto a second line in its column
            page.insert_text((250, y), "Value Adjustment Feature", fontsize=10, fontname="hebo")
            y += 16


def test_columns_split_into_blocks_are_rejoined_as_one_table():
    from luvparse.structured import inject_text_tables

    doc = pymupdf.open()
    page = doc.new_page()
    _two_block_table(page)
    left = {"cls": "text", "text": "**ICC18 ENT-06 ...**", "bbox": [72, 80, 150, 175]}
    right = {"cls": "text", "text": "**Single Premium ...**", "bbox": [250, 80, 520, 175]}
    head = {"cls": "section-header", "text": "## **Supporting Policy Form:**", "bbox": [72, 48, 220, 64]}
    out = inject_text_tables(page, [head, left, right])
    tables = [sg for sg in out if sg["cls"] == "table"]
    assert len(tables) == 1 and len(out) == 1
    html = tables[0]["text"]
    assert 'colspan="2">Supporting Policy Form:' in html
    assert "Single Premium Annuity Contract with Market Value Adjustment Feature" in html
    assert html.count("<tr>") == 6


def test_two_column_prose_is_not_a_table():
    from luvparse.structured import inject_text_tables

    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(12):
        page.insert_text((72, 100 + 14 * i), "the quick brown fox jumps over the lazy dog today", fontsize=10)
        page.insert_text((320, 100 + 14 * i), "pack my box with five dozen liquor jugs right now", fontsize=10)
    left = {"cls": "text", "text": "a", "bbox": [72, 90, 290, 262]}
    right = {"cls": "text", "text": "b", "bbox": [320, 90, 540, 262]}
    out = inject_text_tables(page, [left, right])
    assert [sg["cls"] for sg in out] == ["text", "text"]


def test_header_pieces_merge_even_when_listed_out_of_order():
    from luvparse.structured import merge_furniture

    def piece(x0, x1, text):
        return {"cls": "page-header", "text": text, "bbox": [x0, 20, x1, 30]}

    out = merge_furniture([piece(105, 140, "B"), piece(300, 340, "far"), piece(50, 100, "A"), piece(145, 190, "C")], 600)
    boxes = sorted(tuple(sg["bbox"]) for sg in out)
    assert boxes == [(50, 20, 190, 30), (300, 20, 340, 30)]


def test_vector_icon_becomes_a_picture(tmp_path):
    def icon(page):
        _chart_page(page)
        shape = page.new_shape()
        for k in range(4):
            shape.draw_circle(pymupdf.Point(470 + 6 * k, 120), 8)
        shape.finish(fill=(0.1, 0.4, 0.8), color=(0, 0, 0))
        shape.commit()

    (page,) = structure_pdf(_pdf(tmp_path, icon))
    pictures = [i for i in page.items if i["label"] in ("picture", "image")]
    assert any(p["bbox"]["x"] <= 470 / 595 <= p["bbox"]["x"] + p["bbox"]["w"] and p["bbox"]["w"] < 0.2 for p in pictures)


def test_footer_stays_after_the_body(tmp_path):
    (page,) = structure_pdf(_pdf(tmp_path, _report))
    assert page.markdown.rstrip().endswith("</page_footer>")
    assert page.markdown.lstrip().startswith("<page_header>")


def test_bold_coloured_text_with_own_colour_rule_is_mark(tmp_path):
    def body(page):
        for i in range(6):
            page.insert_text((72, 300 + 14 * i), "Ordinary sentence of body text goes right here on the page.", fontsize=11)
        page.insert_text((72, 400), "Edition ", fontsize=11)
        orange = (0.8, 0.4, 0.0)
        page.insert_text((118, 400), "6-11", fontsize=11, fontname="hebo", color=orange)
        page.draw_rect((118, 402, 140, 403), color=orange, fill=orange)
        page.insert_text((72, 440), "Plain ", fontsize=11)
        page.insert_text((110, 440), "under", fontsize=11)
        page.draw_rect((110, 442, 140, 443), color=(0, 0, 0), fill=(0, 0, 0))

    (page,) = structure_pdf(_pdf(tmp_path, body))
    assert "<mark>6-11</mark>" in page.markdown
    assert "<u>under</u>" in page.markdown


def test_ocr_markings_become_header_and_footer_and_mark_is_dropped_on_strike():
    from luvparse.structured import _tidy_ocr

    out = _tidy_ocr("UNCLASSIFIED\n<mark>~~CONFIDENTIAL~~</mark>\n\nBody text here.\n\nUNCLASSIFIED\n~~CONFIDENTIAL~~")
    assert out.startswith("<page_header>\nUNCLASSIFIED\n~~CONFIDENTIAL~~\n</page_header>")
    assert out.rstrip().endswith("</page_footer>")
    assert "<mark>" not in out


def test_scanned_header_and_footer_come_from_zoomed_strips(tmp_path):
    class StripVision(FakeVision):
        def ocr(self, image, *, prompt_type="default", **_):
            if prompt_type == "furniture":
                self.calls.append(("ocr", prompt_type))
                return "UNCLASSIFIED\n<mark>~~CONFIDENTIAL~~</mark>"
            return super().ocr(image, prompt_type=prompt_type)

    def scan(page):
        page.insert_image(page.rect, stream=_png_bytes())

    fake = StripVision()
    (page,) = structure_pdf(_pdf(tmp_path, scan), config=StructuredConfig(ocr_scanned=True), vision=fake)
    assert page.header == "UNCLASSIFIED\n~~CONFIDENTIAL~~"
    assert page.footer == "UNCLASSIFIED\n~~CONFIDENTIAL~~"
    assert "SECRET" not in page.markdown


def test_dark_grey_text_with_matching_rule_is_underline_not_mark(tmp_path):
    def body(page):
        for i in range(6):
            page.insert_text((72, 300 + 14 * i), "Ordinary sentence of body text goes right here on the page.", fontsize=11)
        grey = (0.137, 0.122, 0.125)
        page.insert_text((72, 400), "B", fontsize=11, fontname="hebo", color=grey)
        page.draw_rect((72, 402, 79, 403), color=grey, fill=grey)

    (page,) = structure_pdf(_pdf(tmp_path, body))
    assert "<u>B</u>" in page.markdown and "<mark>" not in page.markdown


def test_garbled_security_markings_are_snapped_and_underline_keeps_full_stop():
    from luvparse.structured import _fix_markings, _tidy_ocr

    assert _fix_markings("UNCLASSIFIED ~~CONFI~~") == ("UNCLASSIFIED\n~~CONFIDENTIAL~~", True)
    assert _fix_markings("~~CUMENTAL~~")[0] == "~~CONFIDENTIAL~~"
    assert _fix_markings("Some header words")[1] is False
    assert "MODIFICATION.</u>" in _tidy_ocr("<u>MODIFICATION</u>. - text")


def test_standalone_title_line_is_promoted(tmp_path):
    def build(page):
        page.insert_text((72, 80), "Annual Rate Summary", fontsize=18)
        for i in range(6):
            page.insert_text((72, 130 + 14 * i), "Body text of the page continues here in a line.", fontsize=10)

    (page,) = structure_pdf(_pdf(tmp_path, build), config=StructuredConfig())
    assert "# Annual Rate Summary" in page.markdown


def test_vision_table_rejected_when_it_drops_the_words(tmp_path):
    from luvparse.structured import _table_suspect, vision_table_html

    def build(page):
        page.insert_text((72, 100), "Year Sales 2023 10 2024 12 2025 15", fontsize=11)

    path = _pdf(tmp_path, build)
    doc = pymupdf.open(path)
    page = doc[0]
    fake = FakeVision()  # replies with a table that lacks most words
    assert vision_table_html(page, tuple(page.rect), fake, StructuredConfig(), "") == ""
    assert _table_suspect(None) and not _table_suspect(
        {"n_cols": 5, "n_rows": 6, "cells": [{"text": "x"}] * 30}
    )
