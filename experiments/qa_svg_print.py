#!/usr/bin/env python3
"""Independently inspect both SVG alternatives at print-scale resolution."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json, math, xml.etree.ElementTree as ET
import cairosvg
import numpy as np
from PIL import Image

P=Path("review_art_v4")
SVG="{http://www.w3.org/2000/svg}"
WIDTH=2400;HEIGHT=3600

def ink_mask(root, retain_type):
    rt=deepcopy(root)
    for child in list(rt):
        if child.tag != SVG+"g":
            rt.remove(child)
            continue
        if retain_type=="text" and not list(child.iter(SVG+"text")):
            rt.remove(child)
        if retain_type=="lines" and list(child.iter(SVG+"text")):
            rt.remove(child)
    data=ET.tostring(rt,encoding="utf-8")
    png=cairosvg.svg2png(bytestring=data,output_width=WIDTH,output_height=HEIGHT)
    art=np.asarray(Image.open(BytesIO(png)).convert("RGBA"))
    return art[:,:,3]>80

def assess(path):
    root=ET.parse(path).getroot()
    texts=list(root.iter(SVG+"text"))
    if len(texts)!=945:
        raise RuntimeError(f"Unexpected SVG label count: {len(texts)}")
    labeled=np.asarray(Image.open(P/"region_ids_16bit.png"),np.uint16)
    h,w=labeled.shape
    scale=min(400/w,600/h)
    xoff=(400-w*scale)/2;yoff=(600-h*scale)/2
    centers_bad=[]
    for rid,t in enumerate(texts,1):
        xx=float(t.attrib["x"]); yy=float(t.attrib["y"])
        x=round((xx-xoff)/scale-0.5)
        y=round((yy-yoff)/scale-0.5)
        if not(0<=x<w and 0<=y<h and labeled[y,x]==rid):
            centers_bad.append(rid)
    line_pixels=ink_mask(root,"lines")
    label_pixels=ink_mask(root,"text")
    overlap=line_pixels & label_pixels
    from scipy.spatial import cKDTree
    centers_px=np.asarray([
        (float(t.attrib["x"])*WIDTH/400,
         float(t.attrib["y"])*HEIGHT/600) for t in texts],float)
    pixels_y,pixels_x=np.nonzero(overlap)
    collisions=[]
    if len(pixels_x):
        _,near=cKDTree(centers_px).query(
            np.column_stack((pixels_x,pixels_y)),k=1)
        ids,counts=np.unique(near,return_counts=True)
        for idx,count in sorted(zip(ids,counts),key=lambda x:-x[1]):
            t=texts[int(idx)]
            collisions.append({
               "region_id":int(idx)+1,
               "color_number":t.text,
               "overlap_pixels":int(count),
               "center_mm":[float(t.attrib["x"]),float(t.attrib["y"])],
               "rotation":t.attrib.get("transform","none")})
    return {
       "file":str(path),
       "colliding_numbered_regions":collisions,
       "label_count":len(texts),
       "center_outside_numbered_region":centers_bad[:100],
       "center_outside_count":len(centers_bad),
       "label_contour_overlap_px":int(overlap.sum()),
       "outline_ink_px":int(line_pixels.sum()),
       "number_ink_px":int(label_pixels.sum()),
       "font_pt":5.3,
       "line_width_mm":.2
    }

def main():
    result={
        "note":"Rasterized at 2400x3600; label-edge overlap detects line/ink collisions",
        "candidates":[
            assess(P/"lineart_review.svg"),
            assess(P/"lineart_soft_review.svg")
        ]
    }
    (P/"print_proof_qa.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
