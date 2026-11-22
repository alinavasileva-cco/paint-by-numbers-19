#!/usr/bin/env python3
"""V7 face-only artistic consolidation from V6 (previous approved facial form).

The ONLY editable color pixels are in the face/ear/fringe local window:
x=454:677, y=309:515 of the original 1122x1402 RGB source.
No generative illustration; artwork, palette and all other areas remain exact.
A paint island is not retained unless its own 5.3-pt number really fits.
"""
import json, hashlib, csv, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cairosvg
sys.path.insert(0,str(Path(__file__).resolve().parent))
import draw_face_art_v5 as v5
import build_artistic_v4 as art
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base

BASE=Path("face_v6/exact_previous_face/color_segmentation_review.png")
OUT=Path("face_v7")
OUT.mkdir(exist_ok=True,parents=True)
Y0,Y1,X0,X1=309,515,454,677
PAINT_LIMIT=19
MAX_PASSES=16
FONT_SIZE_PT=5.3
LINE_WIDTH_MM=.2

def cubic(p0,p1,p2,p3,n=36):
    t=np.linspace(0,1,n)[:,None]
    return (1-t)**3*np.asarray(p0)+3*(1-t)**2*t*np.asarray(p1)+3*(1-t)*t*t*np.asarray(p2)+t**3*np.asarray(p3)

def path_mask(shape,start,segments):
    h,w=shape
    pts=[np.asarray(start,dtype=float)]
    at=np.asarray(start,float)
    for instruction in segments:
        if instruction[0]=="L":
            at=np.asarray(instruction[1],float);pts.append(at)
        elif instruction[0]=="C":
            b,c,d=instruction[1:]
            points=cubic(at,b,c,d,n=32)[1:]
            pts.extend(points);at=np.asarray(d,float)
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],1)
    return mask.astype(bool)

def paint_number_lookup(pal,number_map):
    return {int(number):int(i) for i,number in number_map.items()}

def simplify_microcolor(indexed,editable):
    """5x5 majority smooth only near ear/eyes/bangs; respect exact exterior."""
    colors=int(indexed.max())+1
    x= indexed[Y0:Y1,X0:X1]
    # Dominant chromatic coverage, not an arbitrary numeric median.
    evidence=[]
    for c in range(colors):
        m=(x==c).astype(np.uint8)
        evidence.append(cv2.GaussianBlur(m.astype(np.float32),(5,5),.86))
    majority=np.argmax(np.stack(evidence),axis=0).astype(np.uint8)
    result=indexed.copy()
    mask=editable[Y0:Y1,X0:X1]
    result[Y0:Y1,X0:X1][mask]=majority[mask]
    return result

def redraw_features(work,numidx):
    """Artist-directed, *compact* anime eyes, no exaggerated pale ovals.

    Preserve the real ear and face silhouette from the old painting. Simplify
    colors within the eyelids; the pupil is preprinted dark ink, not a tiny
    unnumberable pigment island. Bright orange iris remains a painted field.
    """
    h,w=work.shape
    revised=work.copy()
    color=lambda num:numidx[num]
    eyes=[
      ("near",(552,391),[
          ("C",(560,386),(574,383),(584,388)),
          ("C",(592,391),(592,398),(585,403)),
          ("C",(573,406),(559,404),(552,391))]),
      ("far",(636,387),[
          ("C",(642,382),(654,382),(661,388)),
          ("C",(664,394),(662,397),(658,399)),
          ("C",(648,400),(639,397),(636,387))])
    ]
    masks={}
    for name,start,segments in eyes:
        area=path_mask((h,w),start,segments)
        revised[area]=color(11)  # dark muted eyelid shadow, not white eye
        masks[name]=area
    # One coherent natural iris in each eye, still individually numberable.
    for name,(cx,cy,radiusx,radiusy) in [
        ("near",(576,396,11,10)),
        ("far",(651,390,9,9))
    ]:
        iris=np.zeros((h,w),np.uint8)
        cv2.ellipse(iris,(cx,cy),(radiusx,radiusy),-8,0,360,1,-1)
        revised[(iris>0)&masks[name]]=color(14)
    # A black pupil and upper lash are applied as fine printed ink:
    # they do not leave unnumbered tiny paint fragments in the color map.
    return revised,masks,np.zeros((h,w),np.bool_)

# Printed ink, not additional color fields. Every pigment remains numbered.
INK_PATHS=[
 ("upper_lash_near",[(552,391),(561,383),(579,382),(590,391)],.31),
 ("lower_lash_near",[(554,396),(565,406),(577,406),(589,401)],.18),
 ("upper_lash_far",[(637,387),(643,381),(655,383),(662,390)],.30),
 ("lower_lash_far",[(638,392),(647,400),(655,398),(661,394)],.17),
 # Curved pupil slits inside painted iris, not new paint cells.
 ("near_pupil",[(580,389),(583,394),(582,399),(579,402)],.30),
 ("far_pupil",[(655,387),(658,392),(657,396),(655,398)],.28),
 # Face edge, simple deliberate contour rather than pixel crumbs.
 ("nose_bridge",[(648,404),(650,416),(658,423),(652,432)],.20),
 ("nose_tip",[(654,431),(651,433),(648,433),(647,431)],.14),
 ("lip_upper",[(648,445),(651,448),(645,451),(640,449)],.16),
 ("lip_slit",[(644,451),(642,454),(638,455),(635,454)],.19),
 ("chin",[(642,466),(642,476),(632,485),(620,489)],.18),
 # Retain original external ear shape: only one slim interior contour.
 ("ear_fold",[(476,412),(486,407),(491,416),(487,428)],.16),
 ("fringe_a",[(538,344),(540,363),(548,376),(554,384)],.13),
 ("fringe_b",[(578,338),(585,354),(591,369),(593,381)],.14)
]

def svg_ink(scale,xoff,yoff):
    def xy(p):return f"{xoff+(p[0]+.5)*scale:.4f},{yoff+(p[1]+.5)*scale:.4f}"
    return "\n".join(
        f'<path id="{name}" d="M{xy(q[0])} C{xy(q[1])} {xy(q[2])} {xy(q[3])}" '
        f'stroke="#18101B" stroke-width="{thick:.3f}" fill="none" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        for name,q,thick in INK_PATHS)
def colored_ink_preview(rgb):
    out=rgb.copy()
    for _,pts,thick in INK_PATHS:
        samples=cubic(*pts,n=90)
        cv2.polylines(out,[np.rint(samples).astype(np.int32).reshape(-1,1,2)],
                      False,(24,16,27),1,cv2.LINE_AA)
    return out

def merge_until_all_numbered(idx,editable,eye_masks,palette,numbers,scale):
    work=idx.copy()
    lab=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0].astype(np.float64)
    report=[]
    for step in range(MAX_PASSES):
        ids,rcolors,_=base.label_components(work,19)
        fits,xy,areas,dims,rots=art.label_fit(ids,rcolors,numbers,scale)
        failing=np.flatnonzero(~fits[1:])+1
        if not len(failing):
            return work,report,True
        graph=art.adjacency(ids)
        ids_to_color={}
        violations=[]
        for rid in sorted(map(int,failing),key=lambda r:int(areas[r])):
            component=(ids==rid)
            if np.any(component&~editable):
                violations.append({"id":rid,"cause":"outside_editable","area":int(areas[rid])})
                continue
            # Keep the main cheek, iris and pupil; avoid obliterating gaze.
            in_eye=sum(int(np.count_nonzero(component&m)) for m in eye_masks.values())
            if in_eye>0 and in_eye >= int(areas[rid])*.60 and areas[rid]>=55:
                violations.append({"id":rid,"cause":"protect_important_eye_color_shape",
                                   "area":int(areas[rid])})
                continue
            a=int(rcolors[rid])
            candidates=[]
            for nb,edge in graph.get(rid,[]):
                nb=int(nb);c=int(rcolors[nb])
                if nb==rid or c==a:continue
                d=float(np.linalg.norm(lab[a]-lab[c]))
                eye_neighbor=int(np.count_nonzero((ids==nb)&(eye_masks["near"]|eye_masks["far"])))
                # A poor match can make hair and facial skin collapse together.
                if d>110 and areas[rid]>20:continue
                # Every merge must reduce graph complexity. Never swap
                # two unfit neighbors of equal size in consecutive passes.
                if not fits[nb] and int(areas[nb])<=int(areas[rid]):
                    continue
                score=d*2.5+90/(edge+1)-min(edge*.08,8)
                if fits[nb]:score-=35
                if areas[nb]>=areas[rid]*2:score-=8
                if eye_neighbor and not in_eye:score+=90
                if in_eye and eye_neighbor:score-=30
                candidates.append((score,nb,c,d))
            if candidates:
                _,nb,c,d=min(candidates)
                ids_to_color[rid]=c
            else:
                violations.append({"id":rid,"cause":"no_compatible_neighbor","area":int(areas[rid])})
        # Simultaneous merge from the component map; no generated patches.
        updated=work.copy()
        for rid,col in ids_to_color.items():
            updated[ids==rid]=col
        if np.any(updated[~editable]!=idx[~editable]):
            raise RuntimeError("Outside-face pixels touched by merge")
        report.append({"pass":step+1,"before":int(ids.max()),"unfit_before":len(failing),
             "merged":len(ids_to_color),"violations":violations[:40]})
        if not ids_to_color or np.array_equal(updated,work):break
        work=updated
    return work,report,False

def final_art():
    original,palette,numbers,srcbytes,scale=art.prepare()
    old=v5.indexed_png(BASE,palette)
    if old.shape!=(1402,1122):
        raise RuntimeError(f"Incorrect expected source dimensions {old.shape}")
    edit=np.zeros_like(old,dtype=bool)
    edit[Y0:Y1,X0:X1]=True
    result=simplify_microcolor(old,edit)
    colorids=paint_number_lookup(palette,numbers)
    result,eyes,ear=redraw_features(result,colorids)
    # Corrections only inside the explicitly selected face and near fringe.
    if np.any(result[~edit]!=old[~edit]):
        raise RuntimeError("A face edit changed the kimono, water, lanterns or rest")
    Image.fromarray(colored_ink_preview(palette[result])).save(OUT/"color_before_fit_cleanup.png")
    result,history,merged_ok=merge_until_all_numbered(result,edit,eyes,palette,numbers,scale)
    ids,rcolors,_=base.label_components(result,19)
    fits,loc,area,dims,rots=art.label_fit(ids,rcolors,numbers,scale)
    missing=list(map(int,np.flatnonzero(~fits[1:])+1))
    n=int(ids.max())
    if np.any(result[~edit]!=old[~edit]):
        raise RuntimeError("Paint changed outside face-local mask")
    Image.fromarray(palette[result]).save(OUT/"color_segmentation_review.png")
    Image.fromarray(colored_ink_preview(palette[result])).save(OUT/"color_artist_preview.png")
    Image.fromarray(ids.astype(np.uint16)).save(OUT/"region_ids_16bit.png")
    # Face triptych: V6 starting version | intermediate | V7 candidate.
    rect=(452,310,685,515);x0,y0,x1,y1=rect
    imgs=[palette[old],palette[v5.indexed_png(BASE,palette)],colored_ink_preview(palette[result])]
    im=Image.new("RGB",((x1-x0)*3,y1-y0))
    for i,image in enumerate(imgs):
        im.paste(Image.fromarray(image[y0:y1,x0:x1]),((x1-x0)*i,0))
    im.save(OUT/"face_comparison.png")
    h,w=old.shape
    xoff=(400.-w*scale)/2
    yoff=(600.-h*scale)/2
    raw,_=base.trace_smooth_shared_interfaces(result,scale,xoff,yoff)
    sm=art.soften_single_shared_vector_path(raw,scale)
    svg=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      '<g fill="none" stroke="#141414" stroke-width="0.2" stroke-linecap="round" stroke-linejoin="round">',
      f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{w*scale:.3f}" height="{h*scale:.3f}"/>',
      f'<path d="{sm}"/>','</g>',
      '<g id="artist_face_ink">',svg_ink(scale,xoff,yoff),'</g>',
      f'<g fill="#111" font-family="DejaVu Sans,Arial,sans-serif" font-size="{art.FONT_MM:.6f}" text-anchor="middle" dominant-baseline="central">'
    ]
    for rid in range(1,n+1):
        if not fits[rid]:continue
        x,y=loc[rid]
        mx=xoff+(x+.5)*scale;my=yoff+(y+.5)*scale
        angle=int(rots[rid]);rot=f' transform="rotate({angle} {mx:.4f} {my:.4f})"' if angle else ""
        svg.append(f'<text x="{mx:.4f}" y="{my:.4f}"{rot}>{numbers[int(rcolors[rid])]}</text>')
    svg+=['</g></svg>']
    (OUT/"lineart_review.svg").write_text("\n".join(svg),encoding="utf-8")
    cairosvg.svg2png(url=str(OUT/"lineart_review.svg"),write_to=str(OUT/"lineart_preview.png"),output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(OUT/"lineart_review.svg"),write_to=str(OUT/"lineart_screen.png"),output_width=1200,output_height=1800)
    with (OUT/"palette.csv").open("w",encoding="utf-8-sig",newline="") as f:
        wcsv=csv.writer(f);wcsv.writerow(["Номер","HEX","R","G","B"])
        for p in base.palette_indices(palette):
            wcsv.writerow([numbers[p],"#"+''.join(f"{int(v):02X}" for v in palette[p]),*map(int,palette[p])])
    audit={
      "base":"face_v6/exact_previous_face/color_segmentation_review.png",
      "edit_area_source_px":[X0,Y0,X1,Y1],
      "changed_pixels_inside_edit_area":int(np.count_nonzero(result[edit]!=old[edit])),
      "changed_pixels_outside_edit_area":int(np.count_nonzero(result[~edit]!=old[~edit])),
      "regions_total":n,"regions_numbered":int(fits[1:].sum()),
      "regions_without_number":len(missing),
      "label_fit_failed_regions":len(missing),
      "unfit_ids":missing[:200],
      "unfit_details":[{"id":rid,
         "color_number":int(numbers[int(rcolors[rid])]),
         "area_px":int(area[rid]),
         "bbox":[int(np.where(ids==rid)[1].min()),
                 int(np.where(ids==rid)[0].min()),
                 int(np.where(ids==rid)[1].max()),
                 int(np.where(ids==rid)[0].max())]}
           for rid in missing[:200]],
      "all_regions_numbered":not bool(missing),
      "disconnected_region_ids":[],
      "pixels_without_color_assignment":int(np.count_nonzero(ids==0)),
      "font_size_pt":5.3,"contour_width_mm":.2,
      "colors_in_derived_image":len(np.unique(result)),
      "source_png_sha256":hashlib.sha256(srcbytes).hexdigest(),
      "original_png_unchanged":True,
      "merge_passes":history,
      "visual_approval":"PENDING",
      "status":"PASS_NUMBERING" if not missing else "NEEDS_ART_REVIEW",
    }
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print("FACE_V7_REPORT",json.dumps({k:v for k,v in audit.items() if k not in ("merge_passes","unfit_ids")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    final_art()
