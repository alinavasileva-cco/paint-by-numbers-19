#!/usr/bin/env python3
"""Number the user's approved black-and-white face reference, without generating new art.

The 87 color assignments were obtained by matching this exact drawing to the
supplied colored version, then snapping to the original THE CASE palette.
The original mask is never modified. Seven unprintable micro-regions are
corrected in a DERIVED sheet only (48 stroke pixels).
No 'PASS' for subjective palette confidence: color artwork review still required.
"""
import base64,csv,hashlib,io,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent.parent
INPUT=ROOT/"face_reference_v10/approved_lineart_group4.tif"
ORIGINAL_GRAY=ROOT/"face_reference_v10/approved_lineart_original_grayscale.png"
OUT=ROOT/"face_reference_v10/output"
OUT.mkdir(parents=True,exist_ok=True)

NUMBERS=[5,7,2,16,5,16,16,5,7,11,16,16,16,17,2,17,16,17,4,17,9,9,16,9,12,1,9,16,4,4,1,4,17,18,3,19,7,14,11,7,18,17,16,7,5,9,17,4,14,4,10,4,13,17,4,19,19,8,6,16,17,4,9,19,3,19,16,17,9,15,4,3,9,16,16,4,5,9,4,5,18,16,18,12,10,17,10]
CONFIDENCE=[1.0,1.0,.817,.9415,.8305,1,1,.9168,1,1,1,.9918,1,.9752,.4698,.8899,.9995,.9702,.9983,.9445,.5974,.4742,1,.9537,.9658,1,.8944,1,1,1,1,1,1,1,1,1,1,1,.9362,.9792,.9045,.9998,.9902,1,.8035,.8038,.9669,1,.9267,.4259,.9654,.9885,.74,.9332,1,.7015,1,.8528,.5,.7827,.6128,1,.5796,.9953,.982,.9752,.8098,.9524,.7548,1,1,.983,.9576,.6138,.9388,.8986,.9461,.892,1,.9764,.9969,.8212,1,1,1,.6116,.7015]
FONT_PT=5.3;WIDTH_MM=400;THRESH=150
EXPECTED_INPUT_SHA256="a190de43574a70f35f52ddea1cb831e15a969b08a397238ddb7ad71943b90354"
EXPECTED_GRAY_SHA256="f52728559cc7505ea1649de0e218e4f077ee36827bded3d09ca3f34a7aaf5cad"

def components(ink):
    return cv2.connectedComponentsWithStats((~ink).astype("uint8"),connectivity=8)

def fix_unprintable(ink):
    derived=ink.copy()
    _,ids,stats,_=components(derived)
    corrections=[]
    for rid in range(1,len(stats)):
        x,y,w,h,area=map(int,stats[rid])
        if area<=20:
            local=ids[y:y+h,x:x+w]==rid
            derived[y:y+h,x:x+w][local]=True
            corrections.append({"region":rid,"kind":"ink_pinholes","changed_pixels":area})
        elif y==0 and area<=150 and h<=15:
            rr,cc=np.where(ids==rid)
            y0=max(0,y-20);y1=min(derived.shape[0],y+h+20)
            x0=max(0,x-20);x1=min(derived.shape[1],x+w+20)
            others=np.column_stack(np.where((ids[y0:y1,x0:x1]>0)&(ids[y0:y1,x0:x1]!=rid)))
            if len(others)==0:continue
            others+=np.array([y0,x0])
            best=(999999,None,None)
            for a in np.column_stack([rr,cc]):
                d=others-a;ds=(d*d).sum(1);i=int(np.argmin(ds))
                if ds[i]<best[0]:best=(int(ds[i]),a,others[i])
            if best[0]>120:continue
            _,a,b=best
            canvas=np.zeros(derived.shape,np.uint8)
            cv2.line(canvas,(int(a[1]),int(a[0])),(int(b[1]),int(b[0])),1,1)
            cut=(canvas>0)&derived
            derived[cut]=False
            corrections.append({"region":rid,"kind":"bridge_cropped_border","changed_pixels":int(cut.sum())})
    return derived,corrections

def run():
    data=INPUT.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    if sha!=EXPECTED_INPUT_SHA256:raise RuntimeError("Approved lineart mask SHA mismatch")
    gray_source=ORIGINAL_GRAY.read_bytes()
    gray_sha=hashlib.sha256(gray_source).hexdigest()
    if gray_sha!=EXPECTED_GRAY_SHA256:raise RuntimeError("Original antialiased user reference checksum mismatch")
    gray=np.asarray(Image.open(ORIGINAL_GRAY).convert("L"))
    segmented=np.asarray(Image.open(INPUT).convert("L"))
    H,W=gray.shape
    if segmented.shape!=gray.shape or not np.array_equal(gray<THRESH,segmented<THRESH):
        raise RuntimeError("Segmentation mask does not match approved antialiased artwork")
    if (W,H)!=(1122,1402):raise RuntimeError("Incorrect reference dimensions")
    original_ink=gray<THRESH
    ink,corrections=fix_unprintable(original_ink)
    n,region,stats,_=components(ink)
    if n-1!=len(NUMBERS):raise RuntimeError(f"Expected 87 paintable zones, got {n-1}")
    assert len(NUMBERS)==len(CONFIDENCE)
    # User-approved local fix: the narrow cheek segment immediately BELOW
    # the right-side eye belongs to the skin (17), not highlight (19).
    # Resolve it by a stable pixel in the approved 1122x1402 reference:
    # this changes exactly ONE connected region; the other 19s stay as-is.
    cheek_x,cheek_y=755,850
    cheek_region=int(region[cheek_y,cheek_x])
    numbers=NUMBERS.copy()
    if cheek_region<=0 or numbers[cheek_region-1]!=19:
        raise RuntimeError(f"V10 cheek selector is not a color-19 region: id={cheek_region}")
    numbers[cheek_region-1]=17
    with (ROOT/"review/palette.csv").open(encoding="utf-8-sig") as f:
        palette={int(d["Номер"]):tuple(int(d[k]) for k in ("R","G","B")) for d in csv.DictReader(f)}
    if len(palette)!=19:raise RuntimeError("Palette is not 19 original colors")
    work=np.asarray(Image.open(ORIGINAL_GRAY).convert("RGB")).copy()
    work[(ink)&(~original_ink)]=0
    work[(~ink)&(original_ink)]=255
    # White ink-proof and an indexed color check drawn with the exact same segmentation.
    colored=np.full((H,W,3),255,np.uint8)
    colored[ink]=(24,16,27)
    font_px=max(6,int(np.ceil(FONT_PT*25.4/72*W/WIDTH_MM)))
    font=ImageFont.truetype("DejaVuSans.ttf",font_px)
    digits=Image.new("RGBA",(W,H),(0,0,0,0))
    placed={}
    failures=[]
    occupied=np.zeros((H,W),np.uint8)
    for rid in range(1,n):
        number=numbers[rid-1]
        colored[region==rid]=palette[number]
        x,y,bw,bh,area=map(int,stats[rid])
        cell=(region[y:y+bh,x:x+bw]==rid).astype(np.uint8)
        label=str(number)
        bb=font.getbbox(label)
        tw,th=bb[2]-bb[0],bb[3]-bb[1]
        glyph=Image.new("L",(tw+4,th+4),0)
        ImageDraw.Draw(glyph).text((2-bb[0],2-bb[1]),label,font=font,fill=255)
        kernel=cv2.dilate((np.asarray(glyph)>20).astype(np.uint8),np.ones((3,3),np.uint8))
        fit=cv2.erode(cell,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        # The glyph must fit not just inside this connected cell, but also
        # fully inside the printed canvas. Cropped edge labels are forbidden.
        kh,kw=kernel.shape
        glob_x=x+np.arange(bw)[None,:]
        glob_y=y+np.arange(bh)[:,None]
        fit[(glob_x<kw//2)|(glob_x>=W-(kw-kw//2))|
            (glob_y<kh//2)|(glob_y>=H-(kh-kh//2))]=0
        if not np.any(fit):
            failures.append(rid);continue
        clearance=cv2.distanceTransform(cell,cv2.DIST_L2,5)
        clearance[fit==0]=0
        _,_,_,(cx,cy)=cv2.minMaxLoc(clearance)
        px,py=x+cx,y+cy
        gx,gy=px-glyph.width//2,py-glyph.height//2
        label_mask=np.asarray(glyph)>20
        actual_cell=region[gy:gy+glyph.height,gx:gx+glyph.width]
        if actual_cell.shape!=label_mask.shape or np.any(label_mask & (actual_cell!=rid)):
            raise RuntimeError(f"Label {rid} intersects a border")
        if np.any(occupied[gy:gy+glyph.height,gx:gx+glyph.width][label_mask]):
            raise RuntimeError(f"Labels overlap in region {rid}")
        occupied[gy:gy+glyph.height,gx:gx+glyph.width][label_mask]=1
        paint=Image.new("RGBA",glyph.size,(24,16,27,0))
        paint.putalpha(glyph)
        digits.alpha_composite(paint,(gx,gy))
        placed[rid]=(px,py)
    base=Image.fromarray(work).convert("RGBA")
    preview=Image.alpha_composite(base,digits).convert("RGB")
    preview.save(OUT/"lineart_numbered.png")
    # Enlarged proof images do not alter print glyph size; only zoom for QA.
    for name,bounds,zoom in [
        ("face_detail_numbered.png",(300,570,870,1220),3),
        ("eyes_detail_numbered.png",(320,630,800,940),4),
    ]:
        fragment=preview.crop(bounds)
        fragment.resize((fragment.width*zoom,fragment.height*zoom),Image.Resampling.LANCZOS).save(OUT/name)
    Image.fromarray(colored).save(OUT/"color_assignment_review.png")
    Image.fromarray(work).save(OUT/"lineart_corrected.png")
    with io.BytesIO() as png:
        Image.fromarray(work).save(png,format="PNG")
        embedded=base64.b64encode(png.getvalue()).decode("ascii")
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{WIDTH_MM*H/W:.4f}mm" viewBox="0 0 {W} {H}">',
         f'<image width="{W}" height="{H}" href="data:image/png;base64,{embedded}"/>']
    for rid,(px,py) in placed.items():
        svg.append(f'<text x="{px}" y="{py}" font-size="{FONT_PT}pt" font-family="DejaVu Sans" text-anchor="middle" dominant-baseline="central" fill="#18101B">{numbers[rid-1]}</text>')
    svg.append("</svg>")
    (OUT/"lineart_numbered.svg").write_text("\n".join(svg),encoding="utf-8")
    audit={"reference_sha256":sha,"original_antialiased_sha256":gray_sha,"original_input_unchanged":True,
           "derived_stroke_pixels_changed":int(np.count_nonzero(ink!=original_ink)),
           "corrections":corrections,"palette_used_from":"review/palette.csv",
           "regions_total":n-1,"regions_numbered":len(placed),
           "regions_without_number":len(failures),"unfit_region_ids":failures,
           "all_regions_numbered":not failures,"font_size_pt":FONT_PT,
           "colors_total":len(palette),"colors_used":sorted(set(numbers)),
           "manual_color_corrections":[{"region":cheek_region,"from":19,"to":17,
                                        "reference_point_xy":[cheek_x,cheek_y],
                                        "reason":"skin beneath right-side eye"}],
           "ambiguous_color_assignments_below_75pct":sum(v<.75 for v in CONFIDENCE),
           "color_assignments_review_required":True,
           "status":"REVIEW_REQUIRED" if any(v<.75 for v in CONFIDENCE) or failures else "PASS"}
    (OUT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    if failures:raise RuntimeError(f"Some cells were not labeled: {failures}")
    print("NUMBERS_PASS_COLOR_REVIEW_PENDING",audit["regions_total"],audit["regions_numbered"])
if __name__=="__main__":run()
