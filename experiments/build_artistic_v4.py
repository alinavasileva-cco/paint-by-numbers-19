#!/usr/bin/env python3
"""Artwork-first paint-by-numbers from the accepted 1501-region segmentation.

Never modifies input PNG. No hard-coded region count and no label pockets or artificial circles.
Both portrait and rotated real-size glyphs are checked before merging. A region is retained ONLY when
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

def label_fit(ids, region_colors, color_numbers, scale):
    """Validate real 5.3pt ink footprints at 0° or 90° before any merge.

    A 90° orientation is allowed for long slender strokes, preserving
    original color areas that otherwise would be erased.
    """
    n=len(region_colors)-1
    measurements=np.bincount(ids.ravel(),minlength=n+1)
    boxes=ndi.find_objects(ids)
    font=ImageFont.truetype("DejaVuSans.ttf",max(6,int(np.ceil(FONT_MM/scale))))
    valid=np.zeros(n+1,np.bool_)
    locations=np.zeros((n+1,2),np.int32)
    dimensions=np.zeros((n+1,2),np.int16)
    rotations=np.zeros(n+1,np.int16)
    silhouettes={}
    for label in set(str(x) for x in color_numbers.values()):
        bb=font.getbbox(label)
        bw,bh=bb[2]-bb[0],bb[3]-bb[1]
        canvas=Image.new("L",(bw+4,bh+4),0)
        ImageDraw.Draw(canvas).text((2-bb[0],2-bb[1]),label,font=font,fill=255)
        glyph=(np.asarray(canvas)>35).astype(np.uint8)
        halo=cv2.dilate(glyph,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
        silhouettes[label]=(halo,np.rot90(halo).copy())
    for rid in range(1,n+1):
        box=boxes[rid-1]
        if box is None: continue
        yy,xx=box
        mask=(ids[box]==rid).astype(np.uint8)
        label=str(color_numbers[int(region_colors[rid])])
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        for orientation, kernel in enumerate(silhouettes[label]):
            height,width=kernel.shape
            if mask.shape[1] < width or mask.shape[0] < height:
                continue
            eroded=cv2.erode(mask,kernel,
                         borderType=cv2.BORDER_CONSTANT,borderValue=0)
            if not np.any(eroded):continue
            candidate_clearance=distance.copy()
            candidate_clearance[eroded==0]=0
            _,best,_,(col,row)=cv2.minMaxLoc(candidate_clearance)
            if best<=0:continue
            locations[rid]=(col+xx.start,row+yy.start)
            valid[rid]=True
            rotations[rid]=90*orientation
            dimensions[rid]=(width,height)
            break
    return valid,locations,measurements,dimensions,rotations

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


def reconnect_close_same_color(indexed, ids, colors, fit, areas, palette):
    """Reconnect original same-color motifs across <=2 compatible pixels."""
    h,w=indexed.shape
    newer=indexed.copy()
    objects=ndi.find_objects(ids)
    labs=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0].astype(float)
    links=0; altered=0
    invalid=sorted(
        (int(r) for r in np.flatnonzero(~fit[1:])+1 if areas[r]<=75),
        key=lambda r:areas[r])
    for rid in invalid:
        box=objects[rid-1]
        if box is None:continue
        ysl,xsl=box
        pad=4
        x0=max(0,xsl.start-pad);x1=min(w,xsl.stop+pad)
        y0=max(0,ysl.start-pad);y1=min(h,ysl.stop+pad)
        local_ids=ids[y0:y1,x0:x1]
        col=int(colors[rid])
        our=np.argwhere(local_ids==rid)
        other=np.argwhere((newer[y0:y1,x0:x1]==col)&
             (local_ids!=rid)&fit[local_ids])
        if not len(our) or not len(other):continue
        if len(other)>2000:
            other=other[::max(1,len(other)//2000)]
        diff=our[:,None,:]-other[None,:,:]
        dist2=np.sum(diff*diff,axis=-1)
        rindex,oindex=np.unravel_index(int(np.argmin(dist2)),dist2.shape)
        if dist2[rindex,oindex]>13:continue
        start=our[rindex];end=other[oindex]
        yy,xx=int(start[0])+y0,int(start[1])+x0
        yend,xend=int(end[0])+y0,int(end[1])+x0
        nsteps=max(abs(xx-xend),abs(yy-yend))+1
        path_x=np.rint(np.linspace(xx,xend,nsteps)).astype(int)
        path_y=np.rint(np.linspace(yy,yend,nsteps)).astype(int)
        oldcolor=newer[path_y,path_x]
        if np.count_nonzero(oldcolor!=col)>2:continue
        if any(np.linalg.norm(labs[int(k)]-labs[col])>58 for k in oldcolor):
            continue
        newer[path_y,path_x]=col
        altered+=int(np.count_nonzero(oldcolor!=col))
        links+=1
    return newer,links,altered

def next_pass(indexed,palette,numbers,scale):
    ids,colors,sizes=base.label_components(indexed,len(palette))
    n=len(colors)-1
    fit,centers,areas,dimensions,rotations=label_fit(ids,colors,numbers,scale)
    remaining=np.flatnonzero(~fit[1:])+1
    if not len(remaining):
        return indexed, {"before":n,"after":n,"bad_before":0,"merged":0,"bridges":0,"bridge_pixels":0},(ids,colors,fit,centers,dimensions,rotations)
    # First try to restore the same-color continuity of a true paint motif.
    # This changes only <=2 bordering pixels instead of recoloring the island.
    indexed, bridges, bridge_pixels=reconnect_close_same_color(
        indexed,ids,colors,fit,areas,palette)
    if bridges:
        ids,colors,sizes=base.label_components(indexed,len(palette))
        n=len(colors)-1
        fit,centers,areas,dimensions,rotations=label_fit(ids,colors,numbers,scale)
        remaining=np.flatnonzero(~fit[1:])+1
    if not len(remaining):
        return indexed, {"before":n,"after":n,"bad_before":0,"merged":0,
                         "bridges":bridges,"bridge_pixels":bridge_pixels},(ids,colors,fit,centers,dimensions,rotations)
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
                          "bad_before":len(remaining),"merged":changes,
                         "bridges":bridges,"bridge_pixels":bridge_pixels},
            (ids,colors,fit,centers,dimensions,rotations))


# Carefully authored open, non-enclosing decorative INK marks for details
# that cannot host a 5.3-pt pigment number. They are printed linework,
# not new paint areas, and therefore cannot create unnumbered paint islands.

# Source-guided printed linework. The masks below are rebuilt from the
# accepted 1501-area reference; they do not introduce new paint zones.
# Only genuinely slender microdetails can become preprinted contour ink.
def source_microline_ink(original, reduced, palette):
    h,w=original.shape
    lum=cv2.cvtColor(palette[None,:,:].astype(np.uint8),cv2.COLOR_RGB2LAB)[0,:,0]
    src_l=lum[original]
    reduced_l=lum[reduced]
    marks=np.zeros((h,w),np.uint8)
    # Baseline art reference coordinates (1122 x 1402).
    protected_line_objects=[
       # near eye (long eyelash and iris contours)
       (540,372,603,420),
       # far eye
       (632,364,661,410),
       # ear folds
       (460,393,501,449),
       # flower stamens at shoulder
       (521,643,556,687),
    ]
    marked_by_zone={}
    for zone,(x0,y0,x1,y1) in enumerate(protected_line_objects):
        ref=original[y0:y1,x0:x1]
        now=reduced[y0:y1,x0:x1]
        darker=(lum[ref].astype(np.int16)+11 < lum[now].astype(np.int16))
        candidate=(ref!=now)&darker
        n,labels,stats,_=cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),connectivity=8)
        kept=0
        for region in range(1,n):
            x,y,cw,ch,area=map(int,stats[region,:5])
            if area<2 or area>160:continue
            local=(labels[y:y+ch,x:x+cw]==region).astype(np.uint8)
            radius=float(cv2.distanceTransform(local,cv2.DIST_L2,5).max())
            # Prefer thin ornament strokes; never print a whole replacement
            # patch of skin, cloth, water or iris.
            if radius>2.45 and max(cw,ch)/max(1,min(cw,ch)) < 2.7:
                continue
            marks[y0+y:y0+y+ch,x0+x:x0+x+cw][local>0]=255
            kept+=1
        marked_by_zone[["near_eye","far_eye","ear_folds","sakura_stamens"][zone]]=kept
    return marks,marked_by_zone

def source_microline_svg(marks,scale,xoff,yoff):
    paths=[]
    outlines,_=cv2.findContours(marks,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in outlines:
        if cv2.contourArea(cnt)>200:continue
        epsilon=.40
        poly=cv2.approxPolyDP(cnt,epsilon,True)[:,0,:]
        if len(poly)<2:continue
        def pos(p):
            return f'{xoff+(p[0]+.5)*scale:.3f},{yoff+(p[1]+.5)*scale:.3f}'
        if len(poly)==2:
            path='M'+pos(poly[0])+'L'+pos(poly[1])
            paths.append(f'<path d="{path}" fill="none" stroke="#171015" '
                         'stroke-width="0.15" stroke-linecap="round"/>')
        else:
            path='M'+pos(poly[0])+''.join('L'+pos(p) for p in poly[1:])+'Z'
            paths.append(f'<path d="{path}" fill="#171015" stroke="none"/>')
    return '\n'.join(paths)

def artist_ink_proof(flat,palette,marks):
    art=np.ascontiguousarray(flat.copy())
    ink=palette[np.argmin(palette.astype(np.int32).sum(axis=1))]
    art[marks>0]=ink
    return art


def soften_single_shared_vector_path(raw, scale):
    """Round tiny raster kinks on shared interfaces, without new paint polygons.

    The centerline is displaced by at most 0.75 reference-image pixels.
    Sharp architectural corners with long segments retain their angle.
    """
    import re
    out=[]
    for part in re.findall(r'M[^M]+',raw):
        pairs=re.findall(r'[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)',part)
        if len(pairs)<3:
            out.append(part)
            continue
        points=np.asarray([(float(x),float(y)) for x,y in pairs],float)
        closed=part.strip().endswith('Z')
        if closed and np.linalg.norm(points[0]-points[-1]) < 0.002:
            points=points[:-1]
        if len(points)<3:
            out.append(part)
            continue
        def coord(p):return f'{p[0]:.3f},{p[1]:.3f}'
        commands=['M'+coord(points[0])]
        for i in range(1,len(points)-1):
            a,b,c=points[i-1],points[i],points[i+1]
            v=b-a
            q=c-b
            d1=float(np.linalg.norm(v))
            d2=float(np.linalg.norm(q))
            if min(d1,d2)<.001:continue
            v/=d1;q/=d2
            dot=float(np.dot(v,q))
            if dot>.94 or dot<-.96:
                commands.append('L'+coord(b))
                continue
            # Strong angular architecture and lantern edges remain straight.
            if min(d1,d2)>8*scale and dot<.3:
                commands.append('L'+coord(b))
                continue
            radius=min(.75*scale,.21*d1,.21*d2)
            commands.append('L'+coord(b-v*radius)
                            +'Q'+coord(b)+' '+coord(b+q*radius))
        commands.append('L'+coord(points[-1]))
        if closed:commands.append('Z')
        out.append(''.join(commands))
    return ' '.join(out)

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
    fit,centers,areas,dimensions,rotations=label_fit(ids,colors,numbers,scale)
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
    # Per-object audit separates acceptable tiny edits from losses in
    # compositionally important face, embroidered kimono, and lanterns.
    objects={
        "face_and_hair":(290,195,700,565),
        "kimono_and_sakura":(165,520,758,1020),
        "foreground_left_lantern":(0,590,208,855),
        "foreground_right_lantern":(768,900,1070,1380),
        "distant_skyline":(0,0,1122,295)
    }
    region_deltas={}
    for object_name,(x0,y0,x1,y1) in objects.items():
        object_diff=changed[y0:y1,x0:x1]
        region_deltas[object_name]={
            "pixels_changed":int(object_diff.sum()),
            "fraction":round(float(object_diff.mean()),6),
            "region_pixels":int(object_diff.size)
        }
    microink, microink_by_zone=source_microline_ink(original,work,palette)
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
      "nonpaint_decorative_ink":True,
      "source_guided_ink_pixels":int(np.count_nonzero(microink)),
      "source_guided_ink_elements":microink_by_zone,
      "same_color_micro_bridges":sum(step.get("bridges",0) for step in history),
      "same_color_bridge_pixels":sum(step.get("bridge_pixels",0) for step in history),
      "pixel_changes_by_object":region_deltas,
      "label_fitting_method":"5.3pt actual-font glyph footprint, 1px halo, upright or rotated",
      "rotated_label_count":int(np.count_nonzero(rotations)),
      "shared_vector_contours_softened":True,
      "artistic_visual_approval":"PENDING",
      "based_on_original_1501_review":True,
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
    Image.fromarray(artist_ink_proof(palette[work],palette,microink)).save(dest/"color_artist_preview.png")
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
    paths=soften_single_shared_vector_path(paths,scale)
    width,height=work.shape[1]*scale,work.shape[0]*scale
    lines=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      f'<g fill="none" stroke="#141414" stroke-width="{LINE_MM}" stroke-linejoin="round" stroke-linecap="round">',
      f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{width:.3f}" height="{height:.3f}"/>',
      f'<path d="{paths}"/>',
      source_microline_svg(microink,scale,xoff,yoff),'</g>',
      f'<g fill="#111" font-family="{FONT_NAME}, Arial, sans-serif" font-size="{FONT_MM:.5f}" text-anchor="middle" dominant-baseline="central">'
    ]
    for rid in range(1,int(ids.max())+1):
        xx,yy=centers[rid];x=xoff+(xx+0.5)*scale;y=yoff+(yy+0.5)*scale
        rotation=rotations[rid]
        extra=f' transform="rotate({int(rotation)} {x:.4f} {y:.4f})"' if rotation else ''
        lines.append(f'<text x="{x:.4f}" y="{y:.4f}"{extra}>{numbers[int(colors[rid])]}</text>')
    lines.append('</g></svg>')
    vector_text="\n".join(lines)
    if vector_text.count("<text ") != int(ids.max()):
        raise RuntimeError("SVG label count mismatch; do not publish")
    (dest/"lineart_review.svg").write_text(vector_text,encoding="utf-8")
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
        final=Image.fromarray(artist_ink_proof(processed_rgb,palette,microink)[top:bottom,left:right])
        side=Image.new("RGB",(reference.width*2,reference.height))
        side.paste(reference,(0,0));side.paste(final,(reference.width,0))
        side.save(dest/f"comparison_{name}.png")
    print("FINAL AUDIT:",json.dumps({k:v for k,v in audit.items() if k!="rounds"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("outputs_art"))
    args=ap.parse_args()
    generate(SOURCE,args.out)
