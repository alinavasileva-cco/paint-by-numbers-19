#!/usr/bin/env python3
"""Produce *complete* V11 numbered paint-by-numbers from V9 scene / V10 face.
No image generation. Each final colored region is numbered at exactly 5.3pt.
Tiny unlabelable regions are merged to their largest touching neighbor instead
of silently omitting labels. A modified-pixel audit guards artistic changes.
"""
import sys,csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
import cairosvg
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.build import label_components, create_labels, trace_smooth_shared_interfaces
DIR=ROOT/'full_v11_v9_with_v10_face'
SOURCE=DIR/'full_color_v11.png'
W_MM=400.; FONT_PT=5.3; STROKE_MM=.2
def read_palette():
    with (ROOT/'review/palette.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=sorted(csv.DictReader(f),key=lambda x:int(x['Номер']))
    assert len(rows)==19 and [int(r['Номер']) for r in rows]==list(range(1,20))
    return rows,np.array([tuple(int(r[k]) for k in ('R','G','B')) for r in rows],dtype=np.uint8)

def region_contacts(ids):
    aa=[];cc=[]
    for a,b in ((ids[:,:-1],ids[:,1:]),(ids[:-1,:],ids[1:,:])):
        m=a!=b
        a,b=a[m],b[m]
        aa.append(np.stack((np.minimum(a,b),np.maximum(a,b)),axis=1))
    pairs,num=np.unique(np.concatenate(aa),return_counts=True,axis=0)
    touches=defaultdict(list)
    for (a,b),n in zip(pairs,num):
        if a==0 or b==0:continue
        touches[int(a)].append((int(b),int(n)))
        touches[int(b)].append((int(a),int(n)))
    return touches

def merge_unfit(indexed,px_to_mm):
    state=indexed.copy()
    notes=[]
    ncol=19
    pal=np.array([tuple(int(r[k]) for k in ('R','G','B')) for r in
      list(csv.DictReader((ROOT/'review/palette.csv').open(encoding='utf-8-sig'))) ],np.int32)
    for iteration in range(1,41):
        ids,colors,sizes=label_components(state,ncol)
        labels,fail,regions,bycolor=create_labels(state,list(range(1,20)),px_to_mm,0,0)
        notes.append({'iteration':iteration,'regions_total':regions,
                      'numbered':len(labels),'without_number':len(fail)})
        if not fail:
            return state,labels,notes,bycolor
        failing=set(int(ids[int(y),int(x)]) for f in fail for (x,y) in
                    [f['approximate_position']])
        failing.discard(0)
        touches=region_contacts(ids)
        desired=colors.copy()
        edits=0
        for reg in sorted(failing,key=lambda r:int(sizes[r])):
            neighbors=touches.get(reg,[])
            if not neighbors:continue
            def score(item):
                other,border=item
                delta=np.linalg.norm(pal[int(colors[reg])]-pal[int(colors[other])])
                return (border/(1+delta/80.),sizes[other],-delta)
            target=max(neighbors,key=score)[0]
            if int(desired[reg])!=int(colors[target]):
                desired[reg]=colors[target]
                edits+=1
        if not edits:break
        next_state=desired[ids]
        if np.array_equal(next_state,state):break
        state=next_state
    # Failure is explicit, never publish incomplete numbered pages.
    ids,c,s=label_components(state,19)
    labels,fail,regions,bycolor=create_labels(state,list(range(1,20)),px_to_mm,0,0)
    raise RuntimeError(f'Cannot print all regions with 5.3 pt: {len(fail)} remain, of {regions}')

def palette_page(pdf,rows):
    width,height=A4
    pdf.setPageSize(A4)
    regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    pdfmetrics.registerFont(TTFont('FullV11',regular))
    pdfmetrics.registerFont(TTFont('FullV11Bold',bold))
    pdf.setFillColor(colors.HexColor('#18101B'))
    pdf.setFont('FullV11Bold',19)
    pdf.drawString(36,height-62,'THE CASE — ПАЛИТРА 19 ЦВЕТОВ')
    pdf.setFont('FullV11',10)
    pdf.drawString(36,height-86,'Номера точно соответствуют цветной и чёрно-белой версиям V11.')
    pdf.drawString(36,height-104,'Белый цвет бумаги в палитру не входит.')
    for i,r in enumerate(rows):
        col,rr=i//10,i%10;x=36+col*270;y=height-150-rr*60
        pdf.setFillColor(colors.HexColor(r['HEX']))
        pdf.setStrokeColor(colors.HexColor('#BBBBBB'))
        pdf.roundRect(x,y-20,54,36,3,stroke=1,fill=1)
        pdf.setFillColor(colors.HexColor('#18101B'))
        pdf.setFont('FullV11Bold',12)
        pdf.drawString(x+64,y+2,'№ '+r['Номер'])
        pdf.setFont('FullV11',10)
        pdf.drawString(x+64,y-13,r['HEX'].upper())
    pdf.showPage()

def main():
    rows,pal=read_palette()
    image=np.asarray(Image.open(SOURCE).convert('RGB'))
    H,W=image.shape[:2]
    if (W,H)!=(2244,2804):raise RuntimeError('Wrong scene size')
    matches=np.all(image[:,:,None,:]==pal[None,None,:,:],axis=3)
    if not np.all(matches.any(axis=2)):raise RuntimeError('Non-palette pixels found')
    indexed=np.argmax(matches,axis=2).astype(np.uint8)
    px_to_mm=W_MM/W
    cleaned,labels,history,bycolor=merge_unfit(indexed,px_to_mm)
    final=pal[cleaned]
    changed=int(np.any(final!=image,axis=2).sum())
    fraction=changed/(W*H)
    Image.fromarray(final).save(DIR/'full_colored_matching_numbers.png',optimize=True)
    # 0.2mm SINGLE stroke across each shared color boundary; 5.3pt labels.
    paths,npath=trace_smooth_shared_interfaces(cleaned,px_to_mm,0,0)
    artwork_h_mm=H*px_to_mm
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_MM}mm" height="{artwork_h_mm:.4f}mm" viewBox="0 0 {W_MM} {artwork_h_mm:.4f}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<path d="{paths}" fill="none" stroke="#18101B" stroke-width="{STROKE_MM}" stroke-linecap="round" stroke-linejoin="round"/>']
    for number,x,y in labels:
        svg.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="DejaVu Sans" font-size="{FONT_PT*25.4/72.0:.4f}" text-anchor="middle" dominant-baseline="central" fill="#18101B">{number}</text>')
    svg.append('</svg>')
    svg_path=DIR/'full_lineart_numbered.svg'
    svg_path.write_text('\n'.join(svg),encoding='utf-8')
    cairosvg.svg2png(url=str(svg_path),write_to=str(DIR/'full_lineart_numbered.png'),output_width=W,output_height=H)
    cairosvg.svg2pdf(url=str(svg_path),write_to=str(DIR/'full_lineart_numbered.pdf'))
    colored_pdf=DIR/'full_colored_and_palette.pdf'
    pdf=canvas.Canvas(str(colored_pdf),pagesize=(W_MM*72/25.4,artwork_h_mm*72/25.4))
    pdf.setTitle('THE CASE — Full V11 colored painting with 19-color palette')
    pdf.drawImage(ImageReader(str(DIR/'full_colored_matching_numbers.png')),
                  0,0,width=W_MM*72/25.4,height=artwork_h_mm*72/25.4)
    pdf.showPage()
    palette_page(pdf,rows)
    pdf.save()
    a,b=PdfReader(str(DIR/'full_lineart_numbered.pdf')),PdfReader(str(colored_pdf))
    if len(a.pages)!=1 or len(b.pages)!=2:raise RuntimeError('PDF page audit failed')
    if len(labels)!=history[-1]['regions_total']:raise RuntimeError('Missing numbers')
    result={'status':'PASS' if fraction<.08 else 'REVIEW_REQUIRED',
      'source':'full_color_v11.png','regions_total':len(labels),
      'regions_numbered':len(labels),'regions_without_number':0,
      'font_pt':FONT_PT,'stroke_mm':STROKE_MM,'colors_total':19,
      'width_mm':W_MM,'height_mm':round(artwork_h_mm,2),
      'modified_pixels_by_numberability':changed,
      'modified_fraction_by_numberability':round(fraction,5),
      'regions_iterations':history,'segments':npath,
      'lineart_pdf_pages':len(a.pages),'colored_pdf_pages':len(b.pages)}
    (DIR/'audit_full_numbered.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('NUMBERED_ALL',json.dumps({k:v for k,v in result.items() if k!='regions_iterations'},ensure_ascii=False))
if __name__=='__main__':main()
