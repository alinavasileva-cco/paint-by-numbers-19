#!/usr/bin/env python3
"""V6: previous user-approved/restored face + unchanged V5 body/background.

Reference face: review_face_original/color_preview_face_untouched.png.
Other pixels: face_v5/color_segmentation_review.png.
Always produce literal composite; optionally test a *separate* printable
micro-merge candidate. Never falsely call literal face print-ready if it has
sub-5.3pt regions. Never modify approved /main/review.
"""
from pathlib import Path
import json, hashlib, csv, xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import cairosvg
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_artistic_v4 as art
import draw_face_art_v5 as v5
import restore_untouched_face as restore
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base

OUT=Path("face_v6")
PREV=Path("review_face_original/color_preview_face_untouched.png")
V5=Path("face_v5/color_segmentation_review.png")
V4=Path("review_art_v4/color_segmentation_review.png")
CANVAS=(400.,600.)
OUT.mkdir(parents=True,exist_ok=True)

def as_idx(path,palette):
    return v5.indexed_png(path,palette)

def blend_exact(palette):
    previous=as_idx(PREV,palette)
    latest=as_idx(V5,palette)
    before_redraw=as_idx(V4,palette)
    if previous.shape != latest.shape or latest.shape != before_redraw.shape:
        raise RuntimeError("All input maps must have matching dimensions")
    h,w=latest.shape
    protected=restore.facial_protection(h,w)
    # All V5 changed paint is confined to the hand-redrawn facial rectangle.
    changed=latest!=before_redraw
    box=np.zeros((h,w),np.bool_)
    box[342:528,437:677]=True
    if np.any(changed & ~box):
        raise RuntimeError(f"V5 changed {int((changed&~box).sum())} outside its own face zone")
    swap=protected | changed
    exact=latest.copy()
    exact[swap]=previous[swap]
    if np.any(exact[swap]!=previous[swap]):
        raise RuntimeError("The previous face must be copied pixel for pixel")
    if np.any(exact[~swap]!=latest[~swap]):
        raise RuntimeError("Outside previous face, V5 must be unchanged")
    return exact,previous,latest,swap,protected

def audit_mask(work,prior,latest,swap,protected):
    return {
      "exact_previous_face_pixels_changed":int(np.count_nonzero((work!=prior)&protected)),
      "v5_outside_face_pixels_changed":int(np.count_nonzero((work!=latest)&~swap)),
      "face_pixels_swapped":int(np.count_nonzero((prior!=latest)&swap)),
      "face_mask_pixel_count":int(swap.sum()),
      "approved_previous_face_reference":str(PREV),
      "latest_rest_reference":str(V5)
    }

def write_sheet(name,indexed,palette,numbers,scale,source_bytes):
    dest=OUT/name
    dest.mkdir(exist_ok=True,parents=True)
    ids,colors,_=base.label_components(indexed,len(palette))
    n=int(ids.max())
    fits,centers,areas,dimensions,rotations=art.label_fit(ids,colors,numbers,scale)
    bad=[int(i) for i in np.flatnonzero(~fits[1:])+1]
    pixel_unassigned=int(np.count_nonzero(ids==0))
    if pixel_unassigned:raise RuntimeError("Unassigned pigment pixels")
    rgb=palette[indexed]
    Image.fromarray(rgb).save(dest/"color_segmentation_review.png")
    Image.fromarray(ids.astype(np.uint16)).save(dest/"region_ids_16bit.png")
    h,w=indexed.shape
    xoff=(CANVAS[0]-w*scale)/2
    yoff=(CANVAS[1]-h*scale)/2
    boundary,_=base.trace_smooth_shared_interfaces(indexed,scale,xoff,yoff)
    smooth=art.soften_single_shared_vector_path(boundary,scale)
    paths=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      '<g fill="none" stroke="#141414" stroke-width="0.2" stroke-linecap="round" stroke-linejoin="round">',
      f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{w*scale:.3f}" height="{h*scale:.3f}"/>',
      f'<path d="{smooth}"/>','</g>',
      f'<g fill="#111" font-family="DejaVu Sans,Arial,sans-serif" font-size="{art.FONT_MM:.6f}" text-anchor="middle" dominant-baseline="central">'
    ]
    for rid in range(1,n+1):
        if not fits[rid]:continue
        xx,yy=centers[rid]
        x=xoff+(xx+.5)*scale
        y=yoff+(yy+.5)*scale
        rotate=int(rotations[rid])
        extra=f' transform="rotate({rotate} {x:.4f} {y:.4f})"' if rotate else ""
        paths.append(f'<text x="{x:.4f}" y="{y:.4f}"{extra}>{numbers[int(colors[rid])]}</text>')
    paths+=['</g></svg>']
    svg="\n".join(paths)
    if svg.count("<text ")!=(n-len(bad)):
        raise RuntimeError("Number count not matching vector output")
    (dest/"lineart_review.svg").write_text(svg,encoding="utf-8")
    cairosvg.svg2png(url=str(dest/"lineart_review.svg"),write_to=str(dest/"lineart_preview.png"),output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(dest/"lineart_review.svg"),write_to=str(dest/"lineart_screen.png"),output_width=1200,output_height=1800)
    with (dest/"palette.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.writer(f);wr.writerow(["Номер","HEX","R","G","B"])
        for i in base.palette_indices(palette):
            wr.writerow([numbers[i],"#"+''.join(f"{int(v):02X}" for v in palette[i]),*map(int,palette[i])])
    audit={
      "source":"THE_CASE_ночные_фонари_19_красок.png",
      "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
      "face_source":"review_face_original/color_preview_face_untouched.png",
      "rest_source":"face_v5/color_segmentation_review.png",
      "artwork_version":name,
      "regions_total":n,
      "regions_numbered":int(fits[1:].sum()),
      "regions_without_number":len(bad),
      "all_regions_numbered":not bool(bad),
      "region_ids_without_number":bad[:200],
      "pixels_without_color_assignment":pixel_unassigned,
      "font_size_pt":5.3,
      "contour_width_mm":0.2,
      "colors_in_derived_image":len(np.unique(indexed)),
      "status":"PASS_NUMBERING" if not bad else "INCOMPLETE_MISSING_NUMBERS"
    }
    (dest/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return audit

def make_comparison(exact,previous,latest,palette):
    reg=(425,322,690,535)
    x0,y0,x1,y1=reg
    crops=[palette[a][y0:y1,x0:x1] for a in [previous,latest,exact]]
    ow=x1-x0;oh=y1-y0
    page=Image.new("RGB",(ow*3,oh))
    for i,im in enumerate(crops):
        page.paste(Image.fromarray(im),(ow*i,0))
    page.save(OUT/"comparison_face_previous_v5_composite.png")
    Image.fromarray(palette[exact]).save(OUT/"color_composite_exact_face.png")

def candidate_minimal_merge(exact,previous,latest,palette,numbers,scale,protected,swap,source_bytes):
    """Optional separate print-friendly candidate. Must not silently replace exact."""
    allowed=np.zeros(exact.shape,dtype=bool)
    allowed[304:556,412:712]=True
    allowed|=swap
    work=exact.copy()
    history=[]
    for rnd in range(32):
        ids,regcolors,_=base.label_components(work,19)
        fit,loc,areas,dims,rots=art.label_fit(ids,regcolors,numbers,scale)
        bad=[int(i) for i in np.flatnonzero(~fit[1:])+1]
        if not bad:break
        graph=art.adjacency(ids)
        lab=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0].astype(float)
        eligible=[]
        for rid in bad:
            patch=(ids==rid)
            if np.any(patch &~allowed):
                continue
            candidates=[]
            for nbr,shared in graph.get(rid,[]):
                col=int(regcolors[nbr]);old=int(regcolors[rid])
                if col==old:continue
                distance=float(np.linalg.norm(lab[old]-lab[col]))
                # Prefer retaining the original character of the hue and
                # the same semantic patch; avoid turning skin into dark hair.
                score=distance*2.6+25/(shared+1)
                if fit[nbr]:score-=18
                if shared>8:score-=6
                if int(areas[rid])>40:score+=distance
                candidates.append((score,nbr,col))
            if candidates:
                score,nbr,col=min(candidates)
                eligible.append((int(areas[rid]),score,rid,nbr,col))
        if not eligible:
            history.append({"round":rnd+1,"unfit":len(bad),"reason":"no eligible within face"})
            break
        # Constrained local, one at a time. Never touch another object.
        _,score,rid,nbr,col=min(eligible)
        work[ids==rid]=col
        history.append({"round":rnd+1,"unfit_before":len(bad),
                        "merged_id":rid,"into":nbr,"old_area":int(areas[rid]),
                        "perceptual_cost":round(float(score),2)})
    ids,rc,_=base.label_components(work,19)
    fits,loc,areas,dims,rots=art.label_fit(ids,rc,numbers,scale)
    missing=int(np.count_nonzero(~fits[1:]))
    changed=int(np.count_nonzero(work!=exact))
    outside=int(np.count_nonzero((work!=exact)&~allowed))
    summary={"regions":int(ids.max()),"unfit":missing,
             "changed_pixels_from_exact_face":changed,
             "changed_outside_editable_mask":outside,"merge_history":history}
    (OUT/"candidate_experiment.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if not missing and outside==0:
        res=write_sheet("numbered_minimal_candidate",work,palette,numbers,scale,source_bytes)
        summary["status"]="CANDIDATE_NUMBERED_PENDING_VISUAL_APPROVAL"
    else:
        summary["status"]="NOT_PRINT_READY"
        Image.fromarray(palette[work]).save(OUT/"numbered_candidate_incomplete.png")
    (OUT/"candidate_experiment.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def main():
    prior_idx,palette,numbers,original_bytes,scale=art.prepare()
    exact,previous,latest,mask,protected=blend_exact(palette)
    report=audit_mask(exact,previous,latest,mask,protected)
    make_comparison(exact,previous,latest,palette)
    exact_audit=write_sheet("exact_previous_face",exact,palette,numbers,scale,original_bytes)
    candidate=candidate_minimal_merge(exact,previous,latest,palette,numbers,scale,protected,mask,original_bytes)
    report.update({
      "exact_version_regions_total":exact_audit["regions_total"],
      "exact_version_regions_without_number":exact_audit["regions_without_number"],
      "print_candidate_status":candidate["status"],
      "print_candidate_regions_without_number":candidate["unfit"],
      "preexisting_approved_main_art_unchanged":True
    })
    (OUT/"fusion_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FUSION:",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
