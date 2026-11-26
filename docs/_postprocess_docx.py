from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

PATH = "docs/操作手册.docx"
doc = Document(PATH)

# 1) Cover page: break after the Date paragraph; break after the TOC block.
for paragraph in doc.paragraphs:
    if paragraph.style.name == "Date":
        paragraph.add_run().add_break(WD_BREAK.PAGE)
        break

# 1b) Compact TOC styles and a clean break after the TOC block.
for style_name, size in (("toc 1", 9.5), ("toc 2", 9.0)):
    if style_name in [st.name for st in doc.styles]:
        st = doc.styles[style_name]
        st.font.size = Pt(size)
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(1)
        st.paragraph_format.line_spacing = 1.0

body = doc.element.body
toc_sdt = None
for sdt in body.findall(qn("w:sdt")):
    if sdt.findall(".//" + qn("w:docPartGallery")):
        toc_sdt = sdt
        break
if toc_sdt is not None:
    break_p = OxmlElement("w:p")
    run = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run.append(br)
    break_p.append(run)
    toc_sdt.addnext(break_p)

# 2) Tables: grid borders, header shading, 9pt font.
def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        valign = tc_pr.find(qn("w:vAlign"))
        if valign is not None:
            valign.addprevious(shd)
        else:
            tc_pr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)


def table_borders(table):
    tbl_pr = table._tbl.tblPr
    for existing in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "C9D3DD")
        borders.append(el)
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(borders)
    else:
        tbl_pr.append(borders)


def table_width(table):
    tbl_pr = table._tbl.tblPr
    for existing in tbl_pr.findall(qn("w:tblW")):
        tbl_pr.remove(existing)
    width = OxmlElement("w:tblW")
    width.set(qn("w:type"), "pct")
    width.set(qn("w:w"), "5000")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is not None:
        layout.addprevious(width)
    else:
        tbl_pr.append(width)


for table in doc.tables:
    table.autofit = True
    table_width(table)
    table_borders(table)
    for r_idx, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.insert(0, cant_split)
        if r_idx == 0 and tr_pr.find(qn("w:tblHeader")) is None:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for cell in row.cells:
            if r_idx == 0:
                shade_cell(cell, "D9E2F3")
            elif r_idx % 2 == 0:
                shade_cell(cell, "F7F9FC")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(1)
                paragraph.paragraph_format.space_after = Pt(1)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = "微软雅黑"
                    rpr = run._element.get_or_add_rPr()
                    rfonts = rpr.find(qn("w:rFonts"))
                    if rfonts is None:
                        rfonts = OxmlElement("w:rFonts")
                        rpr.append(rfonts)
                    rfonts.set(qn("w:eastAsia"), "微软雅黑")
                    if r_idx == 0:
                        run.font.bold = True

# 3) Header and footer (not on the cover page).
def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, separate, text, end):
        run._r.append(el)


for section in doc.sections:
    section.different_first_page_header_footer = True
    header = section.header
    hp = header.paragraphs[0]
    hp.text = ""
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("RAG 对比实验 · 操作手册")
    hr.font.size = Pt(8)
    hr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = fp.add_run("第 ")
    add_field(fp, " PAGE ")
    r2 = fp.add_run(" 页 / 共 ")
    add_field(fp, " NUMPAGES ")
    r3 = fp.add_run(" 页")
    for r in (r1, r2, r3):
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

# 4) Ask Word to refresh the TOC and page fields on open.
settings = doc.settings.element
update = settings.find(qn("w:updateFields"))
if update is None:
    update = OxmlElement("w:updateFields")
    settings.append(update)
update.set(qn("w:val"), "true")

# 5) Document properties.
doc.core_properties.title = "RAG 对比实验 · 操作手册"
doc.core_properties.subject = "入库 / 启动 / 实验运行 / 人工抽查打分 / 导出与统计"
doc.core_properties.author = "暖通工程知识库项目"
doc.core_properties.keywords = "RAG, HVAC, 操作手册, 人工评分, kappa"
doc.core_properties.comments = "由 rag_lab/docs/manual.md 生成"

doc.save(PATH)
print("postprocessed", PATH)
