#!/usr/bin/env python3
"""V10 print PDFs made from approved numbered and colored PNGs, without redrawing."""
import csv, json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'face_reference_v10/output'
PALETTE=ROOT/'review/palette.csv'
W,H=1122,1402
WIDTH_MM=400
PAGE=(WIDTH_MM*72/25.4, WIDTH_MM*72/25.4*H/W)
MONO=OUT/'lineart_numbered.png'
COLOR=OUT/'color_assignment_review.png'
PDF_MONO=OUT/'V10_lineart_numbered_19_colors.pdf'
PDF_COLOR=OUT/'V10_colored_and_palette_19_colors.pdf'

def check_source():
    for path in (MONO,COLOR):
        with Image.open(path) as im:
            if im.size!=(W,H):
                raise RuntimeError(f'Incorrect {path.name} dimensions: {im.size}')
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    if (audit['regions_total']!=87 or audit['regions_numbered']!=87
        or audit['regions_without_number']!=0 or audit['font_size_pt']!=5.3
        or audit['colors_total']!=19):
        raise RuntimeError('Numbering audit not consistent with approved V10')
    if not any(c.get('region')==66 and c.get('from')==19 and c.get('to')==17
               for c in audit.get('manual_color_corrections',[])):
        raise RuntimeError('Approved skin correction missing')
    with PALETTE.open(encoding='utf-8-sig',newline='') as f:
        palette=sorted(csv.DictReader(f),key=lambda x:int(x['Номер']))
        palette=list(palette)
    if len(palette)!=19 or [int(x['Номер']) for x in palette]!=list(range(1,20)):
        raise RuntimeError('Palette is not exactly 19 numbered source colors')
    return palette

def draw_exact_art(pdf,path):
    # Embed original generated pixels as lossless image; no resampling or overlays.
    pdf.setPageSize(PAGE)
    pdf.drawImage(ImageReader(str(path)),0,0,width=PAGE[0],height=PAGE[1],mask='auto')
    pdf.showPage()

def palette_page(pdf,palette):
    pdf.setPageSize(A4)
    width,height=A4
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if not Path(font).exists() or not Path(bold).exists():
        raise RuntimeError('DejaVu fonts missing')
    pdfmetrics.registerFont(TTFont('V10Palette',font))
    pdfmetrics.registerFont(TTFont('V10PaletteBold',bold))
    pdf.setFillColor(colors.HexColor('#18101B'))
    pdf.setFont('V10PaletteBold',23)
    pdf.drawString(42,height-70,'THE CASE  /  ПАЛИТРА V10')
    pdf.setFont('V10Palette',11)
    pdf.drawString(42,height-96,'19 цветов. Номера совпадают с чёрно-белой схемой.')
    pdf.drawString(42,height-114,'Белый фон не входит в палитру.')
    for i,row in enumerate(palette):
        col,rr=i//10,i%10
        x=42+col*264
        y=height-170-rr*62
        pdf.setFillColor(colors.HexColor(row['HEX']))
        pdf.setStrokeColor(colors.HexColor('#BDBDBD'))
        pdf.setLineWidth(.6)
        pdf.roundRect(x,y-22,59,38,5,stroke=1,fill=1)
        pdf.setFillColor(colors.HexColor('#18101B'))
        pdf.setFont('V10PaletteBold',13)
        pdf.drawString(x+70,y+1,f"№ {int(row['Номер']):02d}")
        pdf.setFont('V10Palette',10)
        pdf.drawString(x+70,y-15,row['HEX'].upper())
    pdf.showPage()

def verify_pdfs():
    p1,p2=PdfReader(str(PDF_MONO),strict=True),PdfReader(str(PDF_COLOR),strict=True)
    if len(p1.pages)!=1 or len(p2.pages)!=2:
        raise RuntimeError('Wrong PDF page count')
    for pg in (p1.pages[0],p2.pages[0]):
        if (abs(float(pg.mediabox.width)-PAGE[0])>.1
            or abs(float(pg.mediabox.height)-PAGE[1])>.1
            or not pg.images):
            raise RuntimeError('Wrong artwork page size or image missing')
    legend=p2.pages[1].extract_text()
    if not all(f'№ {i:02d}' in legend for i in range(1,20)):
        raise RuntimeError('Some palette numbers are missing')
    result={'status':'PASS','black_white_pdf':PDF_MONO.name,
            'colored_pdf':PDF_COLOR.name,'black_white_pages':1,'colored_pages':2,
            'artwork_size_px':[W,H],'artwork_width_mm':WIDTH_MM,
            'palette_colors':19,'skin_region_corrected':66}
    (OUT/'pdf_export_audit.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('V10 PDF_EXPORT_PASS',result)

def main():
    palette=check_source()
    bw=canvas.Canvas(str(PDF_MONO),pagesize=PAGE,pageCompression=1)
    bw.setTitle('V10 - Numbered lineart / 19 colors')
    draw_exact_art(bw,MONO)
    bw.save()
    color=canvas.Canvas(str(PDF_COLOR),pagesize=PAGE,pageCompression=1)
    color.setTitle('V10 - Colored proof and 19-color palette')
    draw_exact_art(color,COLOR)
    palette_page(color,palette)
    color.save()
    verify_pdfs()

if __name__=='__main__':
    main()
