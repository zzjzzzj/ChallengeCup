#!/usr/bin/env python3
"""Convert a practical Markdown document to DOCX with only the stdlib."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


INVALID_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
INLINE_RE = re.compile(r"(`[^`]+`|\*\*.+?\*\*|__.+?__)")
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


def xml_text(value: object) -> str:
    text = str(value)
    text = INVALID_XML_RE.sub("", text)
    return escape(text)


def unmark_text(text: str) -> str:
    text = html.unescape(text)
    replacements = {
        r"\_": "_",
        r"\*": "*",
        r"\`": "`",
        r"\[": "[",
        r"\]": "]",
        r"\(": "(",
        r"\)": ")",
        r"\#": "#",
        r"\+": "+",
        r"\-": "-",
        r"\.": ".",
        r"\!": "!",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def run_xml(text: str, *, bold: bool = False, code: bool = False) -> str:
    if text == "":
        return ""
    props: List[str] = []
    if bold:
        props.append("<w:b/>")
    if code:
        props.append(
            '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" '
            'w:eastAsia="Consolas"/><w:color w:val="B00020"/>'
        )
    rpr = "<w:rPr>%s</w:rPr>" % "".join(props) if props else ""
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return "<w:r>%s<w:t%s>%s</w:t></w:r>" % (rpr, preserve, xml_text(text))


def inline_runs(text: str, *, bold: bool = False) -> str:
    text = unmark_text(text)
    pieces: List[str] = []
    pos = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > pos:
            pieces.append(run_xml(text[pos : match.start()], bold=bold))
        token = match.group(0)
        if token.startswith("`"):
            pieces.append(run_xml(token[1:-1], code=True))
        elif token.startswith("**") or token.startswith("__"):
            pieces.append(run_xml(token[2:-2], bold=True))
        pos = match.end()
    if pos < len(text):
        pieces.append(run_xml(text[pos:], bold=bold))
    return "".join(pieces) or run_xml(" ")


def paragraph_xml(
    text: str = "",
    *,
    style: Optional[str] = None,
    bold: bool = False,
    left: Optional[int] = None,
    hanging: Optional[int] = None,
    before: int = 0,
    after: int = 120,
    keep_lines: bool = False,
    code: bool = False,
) -> str:
    props: List[str] = []
    if style:
        props.append('<w:pStyle w:val="%s"/>' % xml_text(style))
    if before or after:
        props.append('<w:spacing w:before="%d" w:after="%d"/>' % (before, after))
    if left is not None or hanging is not None:
        attrs = []
        if left is not None:
            attrs.append('w:left="%d"' % left)
        if hanging is not None:
            attrs.append('w:hanging="%d"' % hanging)
        props.append("<w:ind %s/>" % " ".join(attrs))
    if keep_lines:
        props.append("<w:keepLines/>")
    ppr = "<w:pPr>%s</w:pPr>" % "".join(props) if props else ""
    if code:
        body = run_xml(unmark_text(text), code=True)
    else:
        body = inline_runs(text, bold=bold)
    return "<w:p>%s%s</w:p>" % (ppr, body)


def horizontal_rule_xml() -> str:
    return (
        "<w:p><w:pPr><w:pBdr>"
        '<w:bottom w:val="single" w:sz="6" w:space="1" w:color="A6A6A6"/>'
        "</w:pBdr><w:spacing w:after=\"180\"/></w:pPr></w:p>"
    )


def split_table_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells: List[str] = []
    buf: List[str] = []
    escaped = False
    in_code = False
    for char in line:
        if escaped:
            buf.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            buf.append(char)
            continue
        if char == "`":
            in_code = not in_code
            buf.append(char)
            continue
        if char == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(char)
    cells.append("".join(buf).strip())
    return cells


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(TABLE_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in cells)


def table_xml(rows: Sequence[Sequence[str]]) -> str:
    max_cols = max((len(row) for row in rows), default=0)
    if max_cols == 0:
        return ""
    grid = "".join('<w:gridCol w:w="%d"/>' % max(1200, 9000 // max_cols) for _ in range(max_cols))
    row_xml: List[str] = []
    for row_index, row in enumerate(rows):
        cells: List[str] = []
        for col_index in range(max_cols):
            value = row[col_index] if col_index < len(row) else ""
            shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
            cells.append(
                "<w:tc><w:tcPr>"
                '<w:tcW w:w="%d" w:type="dxa"/>%s'
                "</w:tcPr>%s</w:tc>"
                % (max(1200, 9000 // max_cols), shade, paragraph_xml(value, bold=row_index == 0, after=60))
            )
        row_xml.append("<w:tr>%s</w:tr>" % "".join(cells))
    return (
        "<w:tbl><w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="808080"/>'
        '<w:left w:val="single" w:sz="4" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="808080"/>'
        '<w:right w:val="single" w:sz="4" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="808080"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="808080"/></w:tblBorders>'
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr><w:tblGrid>%s</w:tblGrid>%s</w:tbl>"
        % (grid, "".join(row_xml))
    )


def parse_blocks(lines: Sequence[str]) -> List[str]:
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            fence = stripped[:3]
            i += 1
            code_lines: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            if i < len(lines):
                i += 1
            for code_line in code_lines or [""]:
                blocks.append(paragraph_xml(code_line, style="CodeBlock", after=0, code=True))
            blocks.append(paragraph_xml("", after=80))
            continue

        if stripped == "":
            if blocks and not blocks[-1].endswith("<w:t> </w:t></w:r></w:p>"):
                blocks.append(paragraph_xml("", after=80))
            i += 1
            continue

        if stripped in {"***", "---", "___"}:
            blocks.append(horizontal_rule_xml())
            i += 1
            continue

        if i + 1 < len(lines) and "|" in line and is_table_separator(lines[i + 1]):
            rows = [split_table_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(split_table_row(lines[i]))
                i += 1
            blocks.append(table_xml(rows))
            blocks.append(paragraph_xml("", after=80))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            style = "Title" if level == 1 else "Heading%d" % min(level - 1, 5)
            blocks.append(paragraph_xml(text, style=style, keep_lines=True, after=160))
            i += 1
            continue

        unordered = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
        if unordered:
            level = len(unordered.group(1).replace("\t", "    ")) // 2
            text = "\u2022 " + unordered.group(2).strip()
            blocks.append(paragraph_xml(text, left=720 + level * 360, hanging=360, after=60))
            i += 1
            continue

        ordered = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
        if ordered:
            level = len(ordered.group(1).replace("\t", "    ")) // 2
            text = "%s. %s" % (ordered.group(2), ordered.group(3).strip())
            blocks.append(paragraph_xml(text, left=720 + level * 360, hanging=360, after=60))
            i += 1
            continue

        quote = re.match(r"^>\s*(.+)$", line)
        if quote:
            blocks.append(paragraph_xml(quote.group(1), left=540, after=80))
            i += 1
            continue

        blocks.append(paragraph_xml(stripped, after=120))
        i += 1
    return blocks


def document_xml(body: Iterable[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>"
        + "".join(body)
        + (
            "<w:sectPr>"
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1260" w:bottom="1440" w:left="1260" '
            'w:header="720" w:footer="720" w:gutter="0"/>'
            "</w:sectPr></w:body></w:document>"
        )
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="24"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:after="260"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimHei"/><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="280" w:after="160"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="220" w:after="120"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading5"><w:name w:val="heading 5"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="120" w:after="60"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock"><w:name w:val="CodeBlock"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0"/><w:shd w:fill="F2F2F2"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def package_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def app_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex Markdown Converter</Application>
</Properties>"""


def core_props_xml(title: str) -> str:
    timestamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>%s</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified>'
        "</cp:coreProperties>" % (xml_text(title), timestamp, timestamp)
    )


def write_docx(markdown_path: Path, output_path: Path, encoding: str) -> None:
    text = markdown_path.read_text(encoding=encoding)
    text = text.replace("\ufeff", "")
    lines = text.splitlines()
    title = markdown_path.stem
    for line in lines:
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            title = unmark_text(match.group(1).strip())
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml())
        docx.writestr("_rels/.rels", package_relationships_xml())
        docx.writestr("word/_rels/document.xml.rels", document_relationships_xml())
        docx.writestr("word/document.xml", document_xml(parse_blocks(lines)))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/app.xml", app_props_xml())
        docx.writestr("docProps/core.xml", core_props_xml(title))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Markdown input path.")
    parser.add_argument("--output", required=True, type=Path, help="DOCX output path.")
    parser.add_argument("--encoding", default="utf-8-sig", help="Input file encoding.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    write_docx(args.input, args.output, args.encoding)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
