#!/usr/bin/env python3
"""Artwork-first paint-by-numbers from the accepted 1501-region segmentation.

Never modifies input PNG. No hard-coded region count, no label pockets, no
synthesized pupils, and no artificial circles. A region is retained ONLY when
a real 5.3pt label plus safety clearance fits within its original paint fill.
Unpaintable fragments are absorbed into physically adjacent pigment shapes.
Every output numeral maps to exactly one connected paint region.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy import ndimage as ndi
import cairosvg
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base

SOURCE = Path("THE_CASE_ночные_фонари_19_красок.png")
REFERENCE = Path("review/color_segmentation_review.png")
PAGE_W, PAGE_H = 400., 600.
LINE_MM, FONT_PT = 0.2, 5.3
FONT_MM = FONT_PT * 25.4 / 72.
MAX_ROUNDS = 22
FONT_NAME = "DejaVu Sans"

def prepare():
    original_bytes = SOURCE.read_bytes()
    src=np.array(Image.open(SOURCE).convert("RGB"))
    ref=np.array(Image.open(REFERENCE).convert("RGB"))
    if ref.shape != src.shape: raise RuntimeError("Reference dimensions changed")
    colors, _=np.unique(src.reshape(-1,3),axis=0,return_inverse=True)
    if len(colors)!=19: raise RuntimeError("Source palette must have 19 colors")
    lookup={tuple(int(z) for z in rgb):i for i,rgb in enumerate(colors)}
    unique, inverse=np.unique(ref.reshape(-1,3),axis=0,return_inverse=True)
    if any(tuple(int(z) for z in rgb) not in lookup for rgb in unique):
        raise RuntimeError("Prior 1501 review has a color absent in original")
    indexed=np.asarray([lookup[tuple(int(z) for z in rgb)] for rgb in unique],
                       dtype=np.uint8)[inverse].reshape(ref.shape[:2])
    region_ids,_,_=base.label_components(indexed,19)
    if region_ids.max()!=1501 or not np.all(region_ids>0):
        raise RuntimeError("Expected accepted 1501 component source")
    h,w=indexed.shape
    scale=min(PAGE_W/w,PAGE_H/h)
    color_numbers={c:i+1 for i,c in enumerate(base.palette_indices(colors))}
    return indexed, colors, color_numbers, original_bytes, scale

def label_fit(ids,region_colors,color_numbers,scale):
    """Use a full text bounding rectangle including a one-source-pixel halo."""
    n=len(region_colors)-1
    measurements=np.bincount(ids.ravel(),minlength=n+1)
    boxes=ndi.find_objects(ids)
    font=ImageFont.truetype("DejaVuSans.ttf",max(6,int(np.ceil(FONT_MM/scale))))
    valid=np.zeros(n+1,np.bool_)
    locations=np.zeros((n+1,2),np.int32)
    widthheight=np.zeros((n+1,2),np.int16)
    for rid in range(1,n+1):
        box=boxes[rid-1]
        if box is None: continue
        yy,xx=box
        mask=(ids[box]==rid).astype(np.uint8)
        label=str(color_numbers[int(region_colors[rid])])
        bb=font.getbbox(label)
        bw,bh=bb[2]-bb[0],bb[3]-bb[1]
        # 1-px outline clearance on each side, plus half-pixel raster margin.
        width,height=bw+4,bh+4
        widthheight[rid]=(width,height)
        if mask.shape[1] < width or mask.shape[0] < height:continue
        eroded=cv2.erode(mask,np.ones((height,width),np.uint8),
                         borderType=cv2.BORDER_CONSTANT,borderValue=0)
        candidates=(eroded>0)
        if not np.any(candidates):continue
        # Favor a visually balanced position, with extra clearance.
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        distance[~candidates]=0
        _,_,_,(col,row)=cv2.minMaxLoc(distance)
        if distance[row,col]<=0:continue
        locations[rid]=(col+xx.start,row+yy.start)
        valid[rid]=True
    return valid,locations,measurements,widthheight

def adjacency(ids):
    pairsets=[]
    for left,right in ((ids[:,:-1],ids[:,1:]),(ids[:-1,:],ids[1:,:])):
        keep=left!=right
        aa=left[keep].astype(np.int32)
        bb=right[keep].astype(np.int32)
        pairsets.append(np.stack((np.minimum(aa,bb),np.maximum(aa,bb)),axis=1))
    pairs, count=np.unique(np.concatenate(pairsets),axis=0,return_counts=True)
    graph=defaultdict(list)
    for (a,b), shared in zip(pairs,count):
        graph[int(a)].append((int(b),int(shared)))
        graph[int(b)].append((int(a),int(shared)))
    return graph

def primary_color_per_region(ids,indexed):
    n=int(ids.max())
    color=np.zeros(n+1,np.uint8)
    _,starts=np.unique(ids.ravel(),return_index=True)
    color[np.arange(1,n+1)]=indexed.ravel()[starts[1:]]
    # np.unique(ids) includes no zero; starts indexes already ordered IDs.
    return color

def next_pass(indexed,palette,numbers,scale):
    ids,colors,sizes=base.label_components(indexed,len(palette))
    n=len(colors)-1
    fit,centers,areas,dimensions=label_fit(ids,colors,numbers,scale)
    remaining=np.flatnonzero(~fit[1:])+1
    if not len(remaining):
        return indexed, {"before":n,"after":n,"bad_before":0,"merged":0},(ids,colors,fit,centers,dimensions)
    graph=adjacency(ids)
    lab=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0].astype(float)
    target_color=colors.astype(np.uint8).copy()
    # Simultaneous color relabeling only (no raster drawing/pockets).
    # Prefer shortest color distance and long shared border over large edits.
    changes=0
    h,w=indexed.shape
    # Dynamic order: micro-patches are removed first; important larger shapes
    # are reconsidered on the next iteration after their neighbors merged.
    small_first=sorted((int(x) for x in remaining),key=lambda x:areas[x])
    batch_limit=max(5,int(len(remaining)*0.75))
    for r in small_first:
        if changes>=batch_limit:break
        candidates=graph.get(r,())
        if not candidates:continue
        c=int(colors[r])
        choices=[]
        for neigh,shared in candidates:
            other=int(colors[neigh])
            # Guard against ping-pong swaps of invalid neighboring islands.
            if neigh in remaining and areas[neigh]<=areas[r]:continue
            if target_color[neigh]!=colors[neigh]:continue
            distance=float(np.linalg.norm(lab[c]-lab[other]))
            # Large protected organic shapes are given a stricter hue rule.
            # All merges operate on old full components, never cut holes.
            score=distance + 28./(shared+1) - min(8.,shared*.055)
            if fit[neigh]:score-=18.
            if areas[neigh]>areas[r]*2:score-=4.
            if areas[r]>140:score+=distance*.5
            choices.append((score,neigh,other))
        if not choices:
            continue
        _,dst,newcol=min(choices)
        if newcol==c:continue
        target_color[r]=newcol
        changes+=1
    if changes==0:
        # Atypical disconnected micro-island: fall back to best real neighbor.
        for r in small_first[:min(len(small_first),20)]:
            candidates=graph.get(r,())
            if not candidates:continue
            c=int(colors[r])
            dst,_=min(candidates,key=lambda t:(
                float(np.linalg.norm(lab[c]-lab[int(colors[t[0]])])),
                -t[1]))
            if colors[dst]!=c:
                target_color[r]=colors[dst];changes+=1
    new_indexed=target_color[ids]
    newid,_,_=base.label_components(new_indexed,len(palette))
    return (new_indexed, {"before":n,"after":int(newid.max()),
                          "bad_before":len(remaining),"merged":changes},
            (ids,colors,fit,centers,dimensions))

def generate(source, dest):
    dest.mkdir(parents=True,exist_ok=True)
    original,palette,numbers,source_bytes,scale=prepare()
    work=original.copy()
    history=[]
    for i in range(MAX_ROUNDS):
        work,report,_=next_pass(work,palette,numbers,scale)
        report["round"]=i+1
        history.append(report)
        print("PASS",i+1,"regions",report["after"],
              "still unfit at beginning",report["bad_before"],
              "merges",report["merged"],flush=True)
        if report["bad_before"]==0:break
        if report["merged"]==0:break

    ids,colors,_=base.label_components(work,19)
    fit,centers,areas,dimensions=label_fit(ids,colors,numbers,scale)
    fail=[int(r) for r in np.flatnonzero(~fit[1:])+1]
    disconnected=[]
    for rid in range(1,int(ids.max())+1):
        count,_=cv2.connectedComponents((ids==rid).astype(np.uint8),connectivity=8)
        if count!=2:disconnected.append(rid)
    used=np.unique(work)
    # Original-to-derived pixel comparison by coarse artwork section.
    h,w=work.shape
    fore=(np.indices((h,w))[0]>=round(h*550/1402))
    maskface=np.zeros((h,w),np.uint8)
    cv2.ellipse(maskface,
      (round(545*w/1122),round(393*h/1402)),
      (round(180*w/1122),round(175*h/1402)),
      0,0,360,1,-1)
    protected=np.logical_or(fore,maskface>0)
    changed=(work!=original)
    xoff=(PAGE_W-work.shape[1]*scale)/2
    yoff=(PAGE_H-work.shape[0]*scale)/2
    audit={
      "target_regions":"artistic_quality",
      "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
      "reference_regions":1501,
      "regions_total":int(ids.max()),
      "regions_numbered":int(fit[1:].sum()),
      "regions_without_number":len(fail),
      "all_regions_numbered":not bool(fail),
      "label_fit_passed_regions":int(fit[1:].sum()),
      "label_fit_failed_regions":len(fail),
      "pixels_without_color_assignment":int(np.count_nonzero(ids==0)),
      "disconnected_region_ids":disconnected[:100],
      "unlabelled_region_ids":fail[:100],
      "font_size_pt":FONT_PT,
      "contour_width_mm":LINE_MM,
      "colors_in_derived_image":len(used),
      "source_colors":len(palette),
      "source_not_overwritten":True,
      "changed_pixels":int(changed.sum()),
      "changed_pixels_in_foreground_face_and_kimono":int(np.sum(changed&protected)),
      "changed_fraction_in_foreground_face_and_kimono":round(float(np.mean(changed[protected])),6),
      "changed_pixels_in_background":int(np.sum(changed&~protected)),
      "rounds":history,
      "status":"PASS" if (
         not fail and not disconnected and np.all(ids>0) and len(used)<=19
      ) else "NEEDS_ARTISTIC_REVIEW",
    }
    (dest/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    if audit["status"]!="PASS":
        raise RuntimeError("Quality gate: "+json.dumps({k:v for k,v in audit.items() if k not in ("rounds",)},ensure_ascii=False))

    # Region/color registry. Same fixed numbering as original 19 pigment palette.
    Image.fromarray(palette[work]).save(dest/"color_segmentation_review.png")
    Image.fromarray(ids.astype(np.uint16)).save(dest/"region_ids_16bit.png")
    order=base.palette_indices(palette)
    with (dest/"palette.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.writer(f);wr.writerow(["Номер","HEX","R","G","B"])
        for c in order:
            rgb=palette[c];wr.writerow([numbers[c],"#"+''.join(f"{int(v):02X}" for v in rgb),*map(int,rgb)])
    with (dest/"region_color_numbers.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.writer(f);wr.writerow(["region_id","color_number","HEX","area_px"])
        for r in range(1,int(ids.max())+1):
            c=int(colors[r])
            wr.writerow([r,numbers[c],"#"+''.join(f"{int(v):02X}" for v in palette[c]),int(areas[r])])

    # Trace a shared contour just once; rounded joins and soft line caps
    # eliminate pixel-doubled pen strokes without inventing paint patches.
    paths,_=base.trace_smooth_shared_interfaces(work,scale,xoff,yoff)
    width,height=work.shape[1]*scale,work.shape[0]*scale
    lines=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      f'<g fill="none" stroke="#141414" stroke-width="{LINE_MM}" stroke-linejoin="round" stroke-linecap="round">',
      f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{width:.3f}" height="{height:.3f}"/>',
      f'<path d="{paths}"/>','</g>',
      f'<g fill="#111" font-family="{FONT_NAME}, Arial, sans-serif" font-size="{FONT_MM:.5f}" text-anchor="middle" dominant-baseline="central">'
    ]
    for rid in range(1,int(ids.max())+1):
        xx,yy=centers[rid];x=xoff+(xx+0.5)*scale;y=yoff+(yy+0.5)*scale
        lines.append(f'<text x="{x:.4f}" y="{y:.4f}">{numbers[int(colors[rid])]}</text>')
    lines.append('</g></svg>')
    (dest/"lineart_review.svg").write_text("\n".join(lines),encoding="utf-8")
    cairosvg.svg2png(url=str(dest/"lineart_review.svg"),write_to=str(dest/"lineart_preview.png"),
                     output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(dest/"lineart_review.svg"),write_to=str(dest/"lineart_screen.png"),
                     output_width=1200,output_height=1800)

    # Montage of old and new artwork, to catch damage in important focal areas.
    original_rgb=palette[original]
    processed_rgb=palette[work]
    for name,(left,top,right,bottom) in {
        "face_hairpin":(290,285,700,560),
        "kimono":(195,520,765,1070),
        "horizon":(0,0,1122,300),
        "foreground_lantern":(745,885,1090,1375)}.items():
        reference=Image.fromarray(original_rgb[top:bottom,left:right])
        final=Image.fromarray(processed_rgb[top:bottom,left:right])
        side=Image.new("RGB",(reference.width*2,reference.height))
        side.paste(reference,(0,0));side.paste(final,(reference.width,0))
        side.save(dest/f"comparison_{name}.png")
    print("FINAL AUDIT:",json.dumps({k:v for k,v in audit.items() if k!="rounds"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("outputs_art"))
    args=ap.parse_args()
    generate(SOURCE,args.out)
