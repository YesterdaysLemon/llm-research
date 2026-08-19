"""Render the Markdown preprint to a publication-style PDF with ReportLab."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "preprint.md"
OUTPUT = ROOT / "output" / "pdf" / "relational_activation_distillation_preprint.pdf"


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("C:/Windows/Fonts/calibrii.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("PaperRegular", regular))
            pdfmetrics.registerFont(TTFont("PaperBold", bold))
            pdfmetrics.registerFont(TTFont("PaperItalic", italic))
            return "PaperRegular", "PaperBold", "PaperItalic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


REGULAR, BOLD, ITALIC = register_fonts()


def inline_markup(value: str) -> str:
    """Convert the small inline Markdown subset used by the manuscript."""
    value = html.escape(value, quote=True)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#1d5c8c"><u>\1</u></link>',
        value,
    )
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", rf'<font name="{BOLD}">\1</font>', value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", rf'<font name="{ITALIC}">\1</font>', value)
    return value


def make_styles():
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "PaperBody",
        parent=sample["BodyText"],
        fontName=REGULAR,
        fontSize=9.2,
        leading=11.85,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#20252b"),
        spaceAfter=4.5,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "PaperTitle",
            parent=body,
            fontName=BOLD,
            fontSize=20,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#10283b"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "PaperSubtitle",
            parent=body,
            fontName=REGULAR,
            fontSize=12.5,
            leading=15.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#496575"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "PaperH2",
            parent=body,
            fontName=BOLD,
            fontSize=13.5,
            leading=16,
            textColor=colors.HexColor("#123d56"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "PaperH3",
            parent=body,
            fontName=BOLD,
            fontSize=10.8,
            leading=13.5,
            textColor=colors.HexColor("#31596c"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "center": ParagraphStyle(
            "PaperCenter", parent=body, alignment=TA_CENTER, spaceAfter=2
        ),
        "caption": ParagraphStyle(
            "PaperCaption",
            parent=body,
            fontSize=8.2,
            leading=10.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#39464f"),
            spaceBefore=3,
            spaceAfter=9,
            keepWithNext=False,
        ),
        "table": ParagraphStyle(
            "PaperTable",
            parent=body,
            fontSize=6.5,
            leading=7.7,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "table_label": ParagraphStyle(
            "PaperTableLabel",
            parent=body,
            fontName=BOLD,
            spaceBefore=2,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "PaperCode",
            parent=body,
            fontName="Courier",
            fontSize=8,
            leading=10.5,
            leftIndent=12,
            rightIndent=12,
            backColor=colors.HexColor("#f2f5f7"),
            borderPadding=7,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "numbered": ParagraphStyle(
            "PaperNumbered",
            parent=body,
            leftIndent=18,
            bulletIndent=0,
            spaceAfter=3,
        ),
    }


STYLES = make_styles()


def table_flowable(rows: list[list[str]]) -> LongTable:
    formatted = [
        [Paragraph(inline_markup(cell.strip()), STYLES["table"]) for cell in row]
        for row in rows
    ]
    ncols = max(len(row) for row in formatted)
    for row in formatted:
        row.extend([Paragraph("", STYLES["table"])] * (ncols - len(row)))
    width = 7.05 * inch
    first = 1.55 * inch if ncols >= 6 else 1.9 * inch
    remaining = (width - first) / max(1, ncols - 1)
    table = LongTable(
        formatted,
        colWidths=[first] + [remaining] * (ncols - 1),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
        spaceBefore=4,
        spaceAfter=9,
    )
    table.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9ef")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10283b")),
            ("FONTNAME", (0, 0), (-1, 0), BOLD),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aebdc5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )
    return table


def parse_markdown(text: str):
    lines = text.splitlines()
    story = []
    pending_table_label = None
    i = 0
    heading_count = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue

        if line.startswith("```"):
            language = line[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            story.append(Preformatted("\n".join(code_lines), STYLES["code"])); continue

        image_match = re.match(r"!\[([^\]]+)\]\(([^)]+)\)", line)
        if image_match:
            caption, relpath = image_match.groups()
            image_path = (SOURCE.parent / relpath).resolve()
            art = Image(str(image_path))
            max_width, max_height = 7.05 * inch, 4.65 * inch
            scale = min(max_width / art.imageWidth, max_height / art.imageHeight)
            art.drawWidth = art.imageWidth * scale
            art.drawHeight = art.imageHeight * scale
            art.hAlign = "CENTER"
            story.extend([Spacer(1, 4), art, Paragraph(inline_markup(caption), STYLES["caption"])])
            i += 1; continue

        if line.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [[cell.strip() for cell in item.strip("|").split("|")] for item in table_lines]
            if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
                rows.pop(1)
            flowables = [table_flowable(rows)]
            if pending_table_label is not None:
                flowables.insert(0, pending_table_label)
                pending_table_label = None
            story.append(KeepTogether(flowables)); continue

        if line.startswith("# "):
            heading_count += 1
            story.append(Paragraph(inline_markup(line[2:]), STYLES["title"])); i += 1; continue
        if line.startswith("## "):
            if heading_count == 1:
                story.append(Paragraph(inline_markup(line[3:]), STYLES["subtitle"]))
                heading_count += 1
            else:
                story.append(Paragraph(inline_markup(line[3:]), STYLES["h2"]))
            i += 1; continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), STYLES["h3"])); i += 1; continue

        if re.match(r"^- ", line):
            items = []
            while i < len(lines) and re.match(r"^- ", lines[i].strip()):
                item = re.sub(r"^- ", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(item), STYLES["body"]), leftIndent=13))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=22, bulletFontName=REGULAR)); continue

        if re.match(r"^\d+\. ", line):
            items: list[str] = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                item = re.sub(r"^\d+\. ", "", lines[i].strip())
                items.append(item)
                i += 1
            for number, item in enumerate(items, start=1):
                story.append(
                    Paragraph(
                        inline_markup(item),
                        STYLES["numbered"],
                        bulletText=f"{number}.",
                    )
                )
            continue

        if line == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9db1bc")))
            i += 1; continue

        paragraph_lines = [line]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate or candidate.startswith(("#", "|", "![", "```", "- ")) or re.match(r"^\d+\. ", candidate):
                break
            paragraph_lines.append(candidate)
            i += 1
        joined = " ".join(paragraph_lines)
        centered = len(story) < 5 and ("Independent researcher" in joined or joined.endswith("2026"))
        if joined.startswith("**Table "):
            pending_table_label = Paragraph(
                inline_markup(joined.rstrip("  ")), STYLES["table_label"]
            )
            continue
        elif centered:
            style = STYLES["center"]
        else:
            style = STYLES["body"]
        story.append(Paragraph(inline_markup(joined.rstrip("  ")), style))

    if pending_table_label is not None:
        story.append(pending_table_label)

    return story


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = LETTER
    canvas.setFont(REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#61727c"))
    canvas.drawCentredString(width / 2, 0.39 * inch, str(doc.page))
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.58 * inch,
        title="Relational Layer Geometry Improves Compositional Transfer Under a Capability Ceiling",
        author="Alireza Afshan",
        subject="Preprint on relational activation distillation and context-selected parameter reuse",
    )
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
