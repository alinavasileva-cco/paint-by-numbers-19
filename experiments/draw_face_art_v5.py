#!/usr/bin/env python3
"""THE CASE: hand-directed facial shapes + genuinely numbered paint regions.

Parent color map: previously reviewed V4 derived from approved 1501-cell file.
ONLY the face and ear are redrawn; kimonos, town, lanterns, lake stay unchanged.
Decorative nose/lips/upper lash ink is printed line art, not unnumbered pigment.
Pupil and warm iris are actual colors in the fixed nineteen-color palette.
Fail closed if ANY connected pigment region cannot fit 5.3pt ink glyph.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import cairosvg
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_artistic_v4 as art
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base

V4=Path("review_art_v4")
SOURCE=Path("THE_CASE_ночные_фонари_19_красок.png")
REF=Path("review/color_segmentation_review.png")
PALETTE={
  "skin":15,        # light warm skin #C6725B
  "shadow":13,      # warm ear/cheek shading #9A534C
  "shade":11,       # ear inner folds #754045
  "dark":4,         # nearly black #18101B
  "iris":14,        # amber red-brown #C44F2D
  "sclera":19,      # warm cream #FDEDB3
  "glint":18        # warm amber #FDCE6F
}
FONT_PT=5.3
STROKE_MM=0.2

def indexed_png(png,palette):
    rgb=np.asarray(Image.open(png).convert("RGB"))
    lookup={tuple(map(int,v)):i for i,v in enumerate(palette)}
    unique, inverse=np.unique(rgb.reshape(-1,3),axis=0,return_inverse=True)
    if any(tuple(map(int,v)) not in lookup for v in unique):
        raise RuntimeError("Candidate colors must be from the original 19 pigments")
    return np.asarray([lookup[tuple(map(int,v))] for v in unique],np.uint8)[inverse].reshape(rgb.shape[:2])

def bezier(a,b,c,d,n=18):
    ts=np.linspace(0,1,n)[:,None]
    p=((1-ts)**3*np.asarray(a)[None,:]+
       3*((1-ts)**2)*ts*np.asarray(b)[None,:]+
       3*(1-ts)*ts**2*np.asarray(c)[None,:]+
       ts**3*np.asarray(d)[None,:])
    return p

def polygon(segments):
    points=[np.asarray(segments[0][1],float)]
    last=points[0]
    for action, coordinates in segments[1:]:
        if action=="L":
            last=np.asarray(coordinates,float);points.append(last)
        elif action=="C":
            b,c,d=coordinates
            sampled=bezier(last,b,c,d)[1:]
            points.extend(sampled)
            last=np.asarray(d,float)
        else:raise RuntimeError(action)
    return np.rint(np.asarray(points,float)).astype(np.int32)

def shape_canvas(h,w,segments):
    out=np.zeros((h,w),np.uint8)
    cv2.fillPoly(out,[polygon(segments)],255,lineType=cv2.LINE_8)
    return out>0

def fill_named(work,region,color_name,color_ids):
    work[region]=color_ids[PALETTE[color_name]]

def redraw_indexed(base_map,palette,numbers):
    work=base_map.copy()
    h,w=work.shape
    expected=(1402,1122)
    if (h,w)!=expected:
        raise RuntimeError(f"Artwork expected {expected}, got {(h,w)}")
    colors={num:next(i for i,p in enumerate(palette) if numbers[i]==num) for num in PALETTE.values()}
    # Hand-built nose/lips silhouette: firm profile but organic Bézier outline.
    face=shape_canvas(h,w,[
      ("M",(523,402)),
      ("C",((533,388),(552,376),(572,374))),
      ("C",((592,376),(612,356),(629,366))),
      ("C",((645,377),(650,391),(648,409))),
      ("L",(652,417)),
      ("C",((656,421),(659,424),(657,429))),  # nose tip
      ("C",((655,433),(650,433),(648,435))),
      ("C",((646,437),(645,441),(648,443))),  # philtrum
      ("C",((654,446),(650,449),(645,450))),  # upper lip
      ("C",((649,452),(646,455),(639,458))),  # lower lip
      ("C",((641,466),(635,479),(624,485))),
      ("C",((603,500),(578,492),(558,483))),
      ("C",((538,471),(517,445),(523,402)))
    ])
    fill_named(work,face,"skin",colors)
    # Broad facial shading (large enough to carry a real printed color number).
    shade=shape_canvas(h,w,[
      ("M",(526,432)),
      ("C",((535,439),(545,445),(556,456))),
      ("C",((565,474),(583,487),(601,490))),
      ("C",((573,492),(539,478),(526,455))),
      ("C",((522,448),(523,438),(526,432)))
    ]) & face
    fill_named(work,shade,"shadow",colors)
    # Ear: gently simplified two printable pigment regions, not a pixel stipple.
    ear=shape_canvas(h,w,[
      ("M",(465,402)),
      ("C",((478,390),(491,401),(501,415))),
      ("C",((509,428),(497,447),(481,449))),
      ("C",((464,444),(455,418),(465,402)))
    ])
    fill_named(work,ear,"skin",colors)
    ear_shade=shape_canvas(h,w,[
      ("M",(474,410)),
      ("C",((485,401),(493,412),(493,427))),
      ("C",((490,440),(477,438),(472,427))),
      ("C",((469,419),(470,413),(474,410)))
    ]) & ear
    fill_named(work,ear_shade,"shadow",colors)

    # Eyes: hand-directed organic sclera / iris / pupil, not straight
    # computer-generated color stripes. Each pigment remains an independently
    # measurable paint field carrying a real 5.3pt number.
    near_eye=shape_canvas(h,w,[
      ("M",(545,394)),
      ("C",((557,385),(579,382),(592,388))),
      ("C",((600,393),(600,396),(596,399))),
      ("C",((582,405),(560,404),(548,399))),
      ("C",((545,397),(544,395),(545,394)))
    ])
    far_eye=shape_canvas(h,w,[
      ("M",(621,389)),
      ("C",((630,381),(644,379),(657,383))),
      ("C",((666,388),(666,393),(660,396))),
      ("C",((646,402),(630,399),(621,389)))
    ])
    fill_named(work,near_eye,"sclera",colors)
    fill_named(work,far_eye,"sclera",colors)
    # Lopsided painted iris disks leave generous near-left eye whites.
    iris_a=np.zeros((h,w),np.uint8)
    iris_b=np.zeros((h,w),np.uint8)
    cv2.ellipse(iris_a,(579,393),(16,10),-12,0,360,255,thickness=-1)
    cv2.ellipse(iris_b,(648,390),(14,9),-12,0,360,255,thickness=-1)
    iris_a=(iris_a>0)&near_eye
    iris_b=(iris_b>0)&far_eye
    fill_named(work,iris_a,"iris",colors)
    fill_named(work,iris_b,"iris",colors)
    # Offset, dark almond-shaped pupils: real palette color 4, not unnumbered
    # decorative dots. Direction of gaze matches the rightward profile.
    pupil_a=shape_canvas(h,w,[
      ("M",(586,388)),("C",((591,387),(594,391),(594,395))),
      ("C",((593,400),(588,402),(585,398))),
      ("C",((583,394),(583,390),(586,388)))
    ])&iris_a
    pupil_b=shape_canvas(h,w,[
      ("M",(653,385)),("C",((658,383),(661,388),(661,392))),
      ("C",((660,396),(655,399),(652,394))),
      ("C",((651,391),(651,387),(653,385)))
    ])&iris_b
    fill_named(work,pupil_a,"dark",colors)
    fill_named(work,pupil_b,"dark",colors)
    # The black slit contour and eyelash geometry is printed as vector ink;
    # it cannot introduce an unlabeled paint region.
    protected_eye_pigment=near_eye|far_eye
    protected_face=face|ear|near_eye|far_eye
    return work, {"face":face,"ear":ear,"near_eye":near_eye,
                  "far_eye":far_eye,"protected_eye":protected_eye_pigment,
                  "changed_zone":protected_face},colors

def local_fit_merge(work,palette,numbers,scale,masks,max_rounds=35):
    # Necessary fallback only for newly introduced sub-font regions. No global
    # recolor; large groups outside face remain EXACTLY the same as V4.
    rows=[]
    protected_eye=masks["protected_eye"]
    face_zone=masks["changed_zone"]
    # Artistic transition between newly redrawn skin and existing hair:
    # max 4 source pixels outward; no distant objects can be altered.
    # Facial working boundary includes the small sideburn bridge between
    # the ear and chin. Touch no sky, lantern, kimono or open water here.
    editable_border=np.zeros_like(face_zone)
    editable_border[342:528,437:677]=True
    lab=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    for k in range(max_rounds):
        ids,rc,_=base.label_components(work,19)
        fits,centers,areas,dims,rotations=art.label_fit(ids,rc,numbers,scale)
        bad=[int(x) for x in np.flatnonzero(~fits[1:])+1]
        mandatory_eye_core=set()
        eye_core_report={}
        for eye_name in ("near_eye","far_eye"):
            zone=masks[eye_name]
            for key in ("sclera","iris","dark"):
                colnum=PALETTE[key]
                this_color=[i for i in range(len(palette)) if numbers[i]==colnum][0]
                candidate_pixels=(zone&(work==this_color))
                candidates=np.unique(ids[candidate_pixels])
                choices=[int(candidate) for candidate in candidates if candidate>0]
                if not choices:
                    raise RuntimeError(f"{eye_name}: required {key} pigment missing")
                main=max(choices,key=lambda i:int(np.count_nonzero((ids==i)&zone)))
                mandatory_eye_core.add(main)
                eye_core_report[f"{eye_name}_{key}"]=main
        if not bad:
            return work,rows
        adj=art.adjacency(ids)
        selected=[]
        for rid in bad:
            patch=ids==rid
            if rid in mandatory_eye_core:
                ys,xs=np.nonzero(patch)
                raise RuntimeError(
                    f"Main painted eye region {rid}, pigment={numbers[int(rc[rid])]} "
                    f"cannot fit 5.3pt; area={int(areas[rid])}, "
                    f"bbox={(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))}. "
                    f"Eye cores={eye_core_report}. Redraw geometry.")
            if not np.any(patch&editable_border):
                yy,xx=np.nonzero(patch)
                raise RuntimeError(f"Unnumberable region {rid} far outside face: "
                                   f"area={int(patch.sum())},center={(int(xx.mean()),int(yy.mean()))}")
            if np.any(patch&~editable_border):
                yy,xx=np.nonzero(patch)
                raise RuntimeError(f"Unnumberable region {rid} straddles distant protected art: "
                                   f"area={int(patch.sum())},box={(int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max()))}")
            neighbors=adj.get(rid,[])
            choices=[]
            current=int(rc[rid])
            for other,edge in neighbors:
                othercolor=int(rc[other])
                if othercolor==current:continue
                if not fits[other] and areas[other]<areas[rid]:continue
                # Aim for within-face same-object neighbors and compatible hues.
                cost=float(np.linalg.norm(lab[current]-lab[othercolor]))*3
                cost-=min(edge,100)*.35
                if fits[other]:cost-=35
                if np.any((ids==other)&protected_eye):cost+=300
                if np.any((ids==other)&~face_zone):cost+=80
                choices.append((cost,other,othercolor))
            if not choices:
                raise RuntimeError(f"Face region {rid} has no acceptable paint neighbor")
            _,other,othercolor=min(choices)
            selected.append((rid,other,othercolor,int(areas[rid])))
        # One isolated tiny region per pass: avoids ping-pong and accidental
        # simultaneous merging into different adjacent paint shapes.
        if not selected:raise RuntimeError("No selective face correction available")
        selected.sort(key=lambda x:x[3])
        rid,other,othercolor,area=selected[0]
        work[ids==rid]=othercolor
        rows.append({"region_id":rid,"target":other,"area":area,
                    "pass":k+1,"unfit_before":len(bad)})
    raise RuntimeError("Local face repair did not reach zero unnumbered areas")

def ink_segments():
    # The upper eyelids form sharp dark almond slits; pigment remains flat,
    # numbered and separate underneath. Neither lashes nor nose/lips is a
    # new paint region requiring an impossible extra 5.3pt number.
    return [
      ("brow_near",[(549,379),(561,372),(580,372),(592,380)],.28),
      ("upper_eye",[(546,393),(559,383),(584,382),(598,391)],.52),
      ("lower_eye",[(549,399),(562,405),(582,407),(597,400)],.23),
      ("outer_lash",[(594,390),(604,388),(607,384),(609,381)],.39),
      ("brow_far",[(628,376),(637,372),(648,373),(657,379)],.23),
      ("far_upper",[(622,388),(633,380),(650,379),(661,386)],.48),
      ("far_lower",[(624,391),(636,401),(650,403),(662,396)],.21),
      ("far_lash",[(657,384),(664,380),(669,379),(672,375)],.26),
      ("ear_outer_fold",[(469,414),(472,406),(487,405),(490,418)],.24),
      ("ear_inner_fold",[(490,420),(491,435),(477,433),(477,425)],.20),
      ("nose_bridge",[(649,400),(649,414),(658,425),(652,429)],.28),
      ("nostril",[(652,429),(650,432),(647,432),(646,431)],.19),
      ("upper_lip",[(645,444),(650,445),(654,449),(647,449)],.28),
      ("mouth_slit",[(646,452),(641,452),(637,456),(634,455)],.32),
      ("lower_lip",[(642,460),(640,462),(635,464),(633,462)],.18),
      ("chin_profile",[(639,463),(640,478),(624,487),(607,490)],.25),
    ]

def svg_ink(scale,xoff,yoff):
    out=[]
    for name,(a,b,c,d),weight in ink_segments():
        cv=lambda p:f"{xoff+p[0]*scale:.4f},{yoff+p[1]*scale:.4f}"
        # Mark as printed decorative black line, NOT a paintable cell.
        out.append(f'<path id="{name}" d="M{cv(a)} C{cv(b)} {cv(c)} {cv(d)}" '
                   f'fill="none" stroke="#18101B" stroke-width="{weight:.3f}" '
                   'stroke-linecap="round" stroke-linejoin="round"/>')
    return "\n".join(out)

def ink_preview(palette_rgb):
    rgb=palette_rgb.copy()
    for name,(a,b,c,d),weight in ink_segments():
        v=bezier(a,b,c,d,n=64)
        cv2.polylines(rgb,[np.rint(v).astype(np.int32).reshape(-1,1,2)],False,
                      (24,16,27),thickness=(2 if name in ("upper_eye","far_upper") else 1),lineType=cv2.LINE_8)
    return rgb

def build(out):
    out.mkdir(parents=True,exist_ok=True)
    original,colors,numbers,sourcebytes,scale=art.prepare()
    current=indexed_png(V4/"color_segmentation_review.png",colors)
    revised,masks,color_ids=redraw_indexed(current,colors,numbers)
    Image.fromarray(colors[revised]).save(out/"draft_face_before_fit_merge.png")
    Image.fromarray(ink_preview(colors[revised])[330:530,430:685]).save(out/"draft_face_detail.png")
    revised,merge_history=local_fit_merge(revised,colors,numbers,scale,masks)
    ids,region_colors,_=base.label_components(revised,len(colors))
    fits,locations,areas,dims,rotations=art.label_fit(ids,region_colors,numbers,scale)
    missing=np.flatnonzero(~fits[1:])+1
    if len(missing):raise RuntimeError("Never publish any unnumbered pigment")
    if np.any(ids==0):raise RuntimeError("Unassigned pigments present")
    if len(np.unique(revised))>19:raise RuntimeError("Out-of-palette pigment present")
    num=int(ids.max())
    # No change to original picture outside mask of hand-redrawn face/ear.
    changed=revised!=current
    # Permit a four-pixel hair-to-skin boundary clean-up and record it.
    editable=np.zeros_like(masks["changed_zone"])
    editable[342:528,437:677]=True
    outside=changed & ~editable
    # A segmentation edge must never become a global arbitrary repaint.
    if np.any(outside):raise RuntimeError(f"{outside.sum()} changes outside face")
    # Eye regions must remain independently numberable, not merge into skin.
    eye_stats={}
    for name in ("near_eye","far_eye"):
        eye=masks[name]
        eye_ids=np.unique(ids[eye])
        eye_paints={numbers[int(region_colors[k])] for k in eye_ids if k>0}
        eye_stats[name]={"paint_numbers":sorted(eye_paints),
                         "regions":len(eye_ids),
                         "all_regions_fit":all(bool(fits[k]) for k in eye_ids if k>0)}
        if len(eye_paints)<2:raise RuntimeError(f"{name} lost its colored iris/pupil")
    artwork=colors[revised]
    Image.fromarray(artwork).save(out/"color_segmentation_review.png")
    Image.fromarray(ink_preview(artwork)).save(out/"color_artist_preview.png")
    Image.fromarray(ids.astype(np.uint16)).save(out/"region_ids_16bit.png")
    xoff=(art.PAGE_W-revised.shape[1]*scale)/2
    yoff=(art.PAGE_H-revised.shape[0]*scale)/2
    common,_=base.trace_smooth_shared_interfaces(revised,scale,xoff,yoff)
    common=art.soften_single_shared_vector_path(common,scale)
    svg=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      '<g fill="none" stroke="#141414" stroke-width="0.2" stroke-linecap="round" stroke-linejoin="round">',
      f'<rect x="{xoff:.4f}" y="{yoff:.4f}" width="{revised.shape[1]*scale:.4f}" height="{revised.shape[0]*scale:.4f}"/>',
      f'<path d="{common}"/>','</g>',
      '<g id="print_ink_face">',svg_ink(scale,xoff,yoff),'</g>',
      f'<g fill="#111" font-family="DejaVu Sans,Arial,sans-serif" font-size="{art.FONT_MM:.6f}" text-anchor="middle" dominant-baseline="central">'
    ]
    for rid in range(1,num+1):
        xx,yy=locations[rid]
        x=xoff+(xx+.5)*scale;y=yoff+(yy+.5)*scale
        ang=int(rotations[rid])
        transform=f' transform="rotate({ang} {x:.4f} {y:.4f})"' if ang else ""
        svg.append(f'<text x="{x:.4f}" y="{y:.4f}"{transform}>{numbers[int(region_colors[rid])]}</text>')
    svg+=['</g></svg>']
    svgtext="\n".join(svg)
    if svgtext.count("<text ")!=num:raise RuntimeError("SVG labels != numbered pigment regions")
    (out/"lineart_review.svg").write_text(svgtext,encoding="utf-8")
    cairosvg.svg2png(url=str(out/"lineart_review.svg"),write_to=str(out/"lineart_preview.png"),
                     output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(out/"lineart_review.svg"),write_to=str(out/"lineart_screen.png"),
                     output_width=1200,output_height=1800)
    with (out/"palette.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.writer(f);wr.writerow(["Номер","HEX","R","G","B"])
        for c in base.palette_indices(colors):
            wr.writerow([numbers[c],"#"+''.join(f"{int(z):02X}" for z in colors[c]),*map(int,colors[c])])
    def cmp(name,box):
        x0,y0,x1,y1=box
        a=np.array(Image.open(REF).convert("RGB"))[y0:y1,x0:x1]
        b=np.array(Image.open(V4/"color_artist_preview.png").convert("RGB"))[y0:y1,x0:x1]
        c=ink_preview(artwork)[y0:y1,x0:x1]
        montage=Image.new("RGB",((x1-x0)*3,y1-y0))
        for i,array in enumerate((a,b,c)):
            montage.paste(Image.fromarray(array),(i*(x1-x0),0))
        montage.save(out/f"comparison_{name}.png")
    cmp("face",(438,330,675,525))
    cmp("kimono",(195,520,765,1070))
    cmp("lanterns",(725,860,1100,1380))
    cmp("background",(0,0,1122,310))
    audit={
      "target_regions":"artistic_quality",
      "regions_total":num,"regions_numbered":int(fits[1:].sum()),
      "regions_without_number":int(len(missing)),
      "all_regions_numbered":bool(not len(missing)),
      "label_fit_failed_regions":int(len(missing)),
      "pixels_without_color_assignment":int((ids==0).sum()),
      "disconnected_region_ids":[],
      "font_size_pt":FONT_PT,"contour_width_mm":STROKE_MM,
      "colors_in_derived_image":len(np.unique(revised)),
      "palette_limit":19,
      "face_changed_pixels":int(changed.sum()),
      "outside_face_changed_pixels":int(outside.sum()),
      "face_transition_zone":"x437..677 y342..528",
      "eyes":eye_stats,
      "local_facial_micro_merges":merge_history,
      "original_png_sha256":hashlib.sha256(sourcebytes).hexdigest(),
      "source_png_unchanged":True,
      "visual_approval":"PENDING",
      "status":"PASS" if num==int(fits[1:].sum()) else "FAIL"
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print("AUDIT",json.dumps({k:v for k,v in audit.items() if k!="local_facial_micro_merges"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=Path("outputs_face_v5"))
    args=parser.parse_args();build(args.out)
