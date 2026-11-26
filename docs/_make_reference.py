from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

DOC = Document("docs/reference.docx")


def set_font(style, name="微软雅黑", size=None, bold=None, color=None):
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color:
        style.font.color.rgb = RGBColor.from_string(color)
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def shade(style, fill="F3F5F8"):
    ppr = style.element.get_or_add_pPr()
    shd = ppr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        ppr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)


def border_left(style):
    ppr = style.element.get_or_add_pPr()
    bdr = ppr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = OxmlElement("w:pBdr")
        ppr.append(bdr)
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "12")
    left.set(qn("w:space"), "4")
    left.set(qn("w:color"), "B9C7D4")
    bdr.append(left)


# Body
for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
    st = DOC.styles[name]
    set_font(st, size=10.5)
    pf = st.paragraph_format
    pf.line_spacing = 1.35
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)

# Headings
headings = {
    "Heading 1": (17, "1F3864", 18, 8),
    "Heading 2": (13.5, "2E5D8A", 14, 6),
    "Heading 3": (11.5, "333333", 10, 4),
    "Heading 4": (10.5, "444444", 8, 3),
}
for name, (size, color, before, after) in headings.items():
    st = DOC.styles[name]
    set_font(st, size=size, bold=True, color=color)
    pf = st.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = True

# Title block
set_font(DOC.styles["Title"], size=26, bold=True, color="1F3864")
DOC.styles["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
DOC.styles["Title"].paragraph_format.space_after = Pt(4)
set_font(DOC.styles["Subtitle"], size=13, bold=False, color="595959")
DOC.styles["Subtitle"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(DOC.styles["Author"], size=11, color="595959")
DOC.styles["Author"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(DOC.styles["Date"], size=11, color="595959")
DOC.styles["Date"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
DOC.styles["Date"].paragraph_format.space_after = Pt(18)

set_font(DOC.styles["TOC Heading"], size=16, bold=True, color="1F3864")
DOC.styles["TOC Heading"].paragraph_format.space_after = Pt(8)

# Code block style (pandoc will use this name for fenced blocks)
try:
    code = DOC.styles["Source Code"]
except KeyError:
    code = DOC.styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
set_font(code, name="Consolas", size=9)
code.paragraph_format.space_before = Pt(3)
code.paragraph_format.space_after = Pt(3)
code.paragraph_format.line_spacing = 1.0
code.paragraph_format.left_indent = Cm(0.2)
shade(code)
border_left(code)

# Inline code
try:
    verb = DOC.styles["Verbatim Char"]
except KeyError:
    verb = DOC.styles.add_style("Verbatim Char", WD_STYLE_TYPE.CHARACTER)
set_font(verb, name="Consolas", size=9)
shade(verb, "F3F5F8")

# Captions
for name in ("Table Caption", "Image Caption", "Caption"):
    if name in [s.name for s in DOC.styles]:
        set_font(DOC.styles[name], size=9, color="666666")

# Page margins
for section in DOC.sections:
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

DOC.save("docs/reference.docx")
print("reference styled")
