#!/usr/bin/env python3
"""Smoothing only the shared vector outline; does NOT edit pigment assignments."""
import json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import cairosvg
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build

BASE=Path("review_art_v4")
MM_W,MM_H=400.,600.
NS="{http://www.w3.org/2000/svg}"

def revised_shared_path(raw,scale):
    chunks=re.findall(r'M[^M]+',raw)
    output=[]
    segments_before=0
    segments_after=0
    def pstr(p):return f'{p[0]:.3f},{p[1]:.3f}'
    for seg in chunks:
        coords=re.findall(r'[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)',seg)
        points=np.array([(float(x),float(y)) for x,y in coords],np.float64)
        if len(points)<3:
            output.append(seg);continue
        closed=seg.strip().endswith('Z')
        if closed and np.linalg.norm(points[0]-points[-1])<.002:
            points=points[:-1]
        if len(points)<3:
            output.append(seg);continue
        segments_before+=len(points)
        smoothed=points.copy()
        n=len(points)
        span=range(n) if closed else range(1,n-1)
        for i in span:
            before=points[i-1]
            current=points[i]
            following=points[(i+1)%n]
            v=current-before
            w=following-current
            d1=np.linalg.norm(v);d2=np.linalg.norm(w)
            if min(d1,d2)<1e-3:continue
            cosine=float(np.dot(v,w)/(d1*d2))
            # Important geometric right-angle architecture remains a true corner.
            if min(d1,d2)>4*scale and cosine<.35:continue
            weighted=.18*before + .64*current + .18*following
            shift=weighted-current
            length=np.linalg.norm(shift)
            if length>.65*scale:
                weighted=current+shift*(.65*scale/length)
            smoothed[i]=weighted
        # Geometric simplification: pixel stair steps -> one controlled curve,
        # while distinct corners remain because their displacement is large.
        approx=cv2.approxPolyDP(smoothed.astype(np.float32),
                               epsilon=.72*scale,closed=closed)[:,0,:].astype(float)
        if len(approx)<3:
            approx=np.array([smoothed[0],smoothed[len(smoothed)//2],smoothed[-1]])
        if not closed:
            approx[0]=points[0];approx[-1]=points[-1]
        segments_after+=len(approx)
        cmd=['M'+pstr(approx[0])]
        for j in range(1,len(approx)-1):
            a,b,c=approx[j-1],approx[j],approx[j+1]
            v=b-a;w=c-b
            d1=np.linalg.norm(v);d2=np.linalg.norm(w)
            if min(d1,d2)<.002:continue
            v/=d1;w/=d2
            dot=float(np.dot(v,w))
            if dot>.93 or (min(d1,d2)>4*scale and dot<.35):
                cmd.append('L'+pstr(b));continue
            rad=min(.72*scale,.26*d1,.26*d2)
            cmd.append('L'+pstr(b-v*rad)+'Q'+pstr(b)+' '+pstr(b+w*rad))
        cmd.append('L'+pstr(approx[-1]))
        if closed:cmd.append('Z')
        output.append(''.join(cmd))
    return ' '.join(output),segments_before,segments_after

def main():
    colored=np.asarray(Image.open(BASE/"color_segmentation_review.png").convert("RGB"))
    pal,inv=np.unique(colored.reshape(-1,3),axis=0,return_inverse=True)
    indexed=inv.reshape(colored.shape[:2]).astype(np.uint8)
    h,w=indexed.shape
    scale=min(MM_W/w,MM_H/h)
    xoff,yoff=(MM_W-w*scale)/2,(MM_H-h*scale)/2
    raw,_=build.trace_smooth_shared_interfaces(indexed,scale,xoff,yoff)
    revised,before,after=revised_shared_path(raw,scale)
    tree=ET.parse(BASE/"lineart_review.svg")
    root=tree.getroot()
    candidates=[x for x in root.iter(NS+"path") if len(x.attrib.get("d",""))>10000]
    if len(candidates)!=1:
        raise RuntimeError(f"Expected one long shared-path group: {len(candidates)}")
    candidates[0].set("d",revised)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg=BASE/"lineart_soft_review.svg"
    tree.write(svg,encoding="utf-8",xml_declaration=True)
    cairosvg.svg2png(url=str(svg),write_to=str(BASE/"lineart_soft_screen.png"),
                     output_width=1200,output_height=1800)
    cairosvg.svg2png(url=str(svg),write_to=str(BASE/"lineart_soft_preview.png"),
                     output_width=2400,output_height=3600)
    info={"source":"review_art_v4/lineart_review.svg",
          "type":"vector-only; pigment regions and numbers unchanged",
          "original_shared_vertices":before,"smoothed_shared_vertices":after,
          "font_size_pt":5.3,"line_width_mm":.2,
          "source_png_unchanged":True,
          "vector_art_approval":"PENDING"}
    (BASE/"contour_smoothing_audit.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(info))

if __name__=="__main__":
    main()
