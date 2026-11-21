#!/usr/bin/env python3
"""Feasibility audit for an immutable foreground and real 5.3-pt digits.
Uses the accepted 1501-cell PNG; never regenerates or edits source artwork.
"""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PT = 5.3
FONT_MM = FONT_PT * 25.4 / 72.0

def immutable_mask(h, w):
    """Conservative protected painting zone, in 1122x1402 source coords."""
    mask = np.zeros((h, w), dtype=np.uint8)
    def polygon(points):
        poly = np.rint(np.array(points, np.float64) * [w/1122, h/1402]).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
    def rect(x1,y1,x2,y2):
        x1,x2=round(x1*w/1122),round(x2*w/1122)
        y1,y2=round(y1*h/1402),round(y2*h/1402)
        mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=1
    # Girl's figure, face, hair, kimono and reflections, broadly protected
    polygon([(350,185),(610,170),(681,235),(694,345),(677,475),
             (724,590),(743,770),(730,970),(750,1140),
             (690,1355),(475,1390),(290,1290),(167,1070),(179,840),
             (255,624),(301,487),(284,345)])
    # All foreground water and primary reflections
    rect(0,850,1122,1402)
    # Foreground paper lanterns; including their illustrated decoration
    for bounds in [
        (0,590,204,847), (660,445,837,790), (777,894,1050,1322),
        (836,367,968,591), (96,274,212,479),(178,211,299,379),
        (664,206,749,387),(766,237,855,398),(1021,261,1122,460)
    ]: rect(*bounds)
    return mask.astype(bool)

def assess(indexed, palette):
    h,w=indexed.shape
    protect=immutable_mask(h,w)
    scale=min(400/w,600/h)
    order=np.argsort(palette[:,0])
    order=sorted(range(len(palette)),key=lambda i:float(i))
    # Match the existing SVG palette numbering; one- vs two-digit width matters.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
    import build
    order=build.palette_indices(palette)
    number_for_color={c:i+1 for i,c in enumerate(order)}
    total=0
    unfit=0
    protected_all=0
    protected_unfit=0
    partial_unfit=0
    failures=[]
    comps=[]
    for c in range(len(palette)):
        n, labels, stats, _=cv2.connectedComponentsWithStats(
            (indexed==c).astype(np.uint8), 8)
        for component in range(1,n):
            rid=total+1
            area=int(stats[component,cv2.CC_STAT_AREA])
            x,y,width,height=map(int,stats[component,:4])
            patch=(labels[y:y+height,x:x+width]==component)
            intersection=protect[y:y+height,x:x+width]
            overlap=int(np.count_nonzero(patch&intersection))
            wholly=(overlap==area)
            # 5.3pt with 1px physical clearance; same sizing as current renderer.
            number=str(number_for_color[c])
            rx=int(np.ceil((len(number)*FONT_MM/scale*.66+1.3)/2))
            ry=int(np.ceil((FONT_MM/scale+1.0)/2))
            kernel=np.ones((2*ry+1,2*rx+1),np.uint8)
            can_fit=(width>=2*rx+1 and height>=2*ry+1 and
                     cv2.erode(patch.astype(np.uint8),kernel,
                         borderType=cv2.BORDER_CONSTANT,borderValue=0).any())
            total+=1
            protected_all+=int(wholly)
            unfit+=int(not can_fit)
            protected_unfit+=int(wholly and not can_fit)
            partial_unfit+=int((0<overlap<area) and not can_fit)
            if not can_fit:
                failures.append({"region_id":rid,"color_id":c,"area":area,
                                 "bbox":[x,y,width,height],
                                 "protected_all":wholly,
                                 "protected_pixel_count":overlap})
            comps.append((rid,c,area,wholly,overlap,can_fit))
    out={
        "source":"review/color_segmentation_review.png",
        "source_segmentation_sha256":hashlib.sha256(
            Path("review/color_segmentation_review.png").read_bytes()).hexdigest(),
        "total_regions":total,"font_pt":FONT_PT,
        "protected_pixels":int(protect.sum()),
        "protected_whole_regions":protected_all,
        "total_areas_too_small_for_full_label":unfit,
        "fully_protected_areas_too_small_for_full_label":protected_unfit,
        "partially_protected_areas_too_small_for_full_label":partial_unfit,
        "editable_whole_regions":sum(not x[3] and x[4]==0 for x in comps),
        "immutable_shape_1300_labels_feasible":protected_unfit==0,
        "unfit_protected_samples":[a for a in failures if a["protected_all"]][:50],
        "smallest_areas":[a for a in sorted(failures,key=lambda a:a["area"])][:30],
        "protected_unfit_area_le_4px":sum(a["area"]<=4 for a in failures if a["protected_all"]),
        "protected_unfit_area_le_20px":sum(a["area"]<=20 for a in failures if a["protected_all"])
    }
    return out, [a for a in failures if a["protected_all"]],protect

def main():
    source=np.asarray(Image.open("review/color_segmentation_review.png").convert("RGB"))
    colors, inverse=np.unique(source.reshape(-1,3),axis=0,return_inverse=True)
    indexed=inverse.reshape(source.shape[:2]).astype(np.uint8)
    audit,blockers,protect=assess(indexed,colors)
    output=Path("study_1300")
    output.mkdir(parents=True,exist_ok=True)
    (output/"feasibility.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    # This visual is a diagnostic overlay, NOT a revised artistic print.
    artwork=Image.fromarray(source).convert("RGBA")
    transparent=Image.new("RGBA",artwork.size,(0,0,0,0))
    pencil=ImageDraw.Draw(transparent)
    for a in blockers:
        x,y,w,h=a["bbox"]
        cx=x+w/2; cy=y+h/2
        r=max(3,min(9,max(w,h)/2+2))
        pencil.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(241,45,59,105),outline=(255,70,70,230),width=1)
    overlay=Image.alpha_composite(artwork,transparent).convert("RGB")
    strip=Image.new("RGB",(overlay.width,overlay.height+75),(255,255,255))
    strip.paste(overlay,(0,75))
    ink=ImageDraw.Draw(strip)
    ink.text((18,13),f"{len(blockers)} protected regions cannot fit 5.3pt labels",fill=(20,20,20))
    ink.text((18,40),"Red markers = mathematical label-footprint blockers; NOT an edited painting",fill=(70,70,70))
    strip.save(output/"protected_label_blockers.png",optimize=True)
    print(json.dumps({key:value for key,value in audit.items() if not isinstance(value,list)},ensure_ascii=False))
if __name__=="__main__":
    main()
