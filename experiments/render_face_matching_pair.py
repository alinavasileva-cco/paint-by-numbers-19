#!/usr/bin/env python3
"""Render paired B/W line art for the COLOR proof with exactly restored face.

Crucial: 150 microscopic regions cannot fit 5.3pt labels. They remain blank,
and the output is visibly marked INCOMPLETE in the audit (not falsely final).
"""
from pathlib import Path
import json, sys, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cairosvg

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/"scripts"))
import restore_untouched_face as restore
import build_artistic_v4 as art
import build as base

OUT=Path("review_face_original")
def build():
    restore.main()
    color=np.asarray(Image.open(OUT/"color_preview_face_untouched.png").convert("RGB"))
    original, palette, number_by_color, source_bytes,scale=art.prepare()
    codebook={tuple(int(i) for i in c):n for n,c in enumerate(palette)}
    unique, inv=np.unique(color.reshape(-1,3),axis=0,return_inverse=True)
    if any(tuple(map(int,c)) not in codebook for c in unique):
        raise RuntimeError("Restored face preview changed palette")
    indexed=np.asarray([codebook[tuple(map(int,c))] for c in unique],dtype=np.uint8)[inv].reshape(color.shape[:2])
    ids, colors,_=base.label_components(indexed,len(palette))
    fits, centers, areas, dims, rotations=art.label_fit(ids,colors,number_by_color,scale)
    nregions=int(ids.max())
    bad=np.flatnonzero(~fits[1:])+1
    good=nregions-len(bad)
    h,w=indexed.shape
    xoff,yoff=(400-w*scale)/2,(600-h*scale)/2
    raw,_=base.trace_smooth_shared_interfaces(indexed,scale,xoff,yoff)
    softened=art.soften_single_shared_vector_path(raw,scale)
    lines=[
      '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
      '<rect width="400" height="600" fill="white"/>',
      '<g fill="none" stroke="#141414" stroke-width="0.2" stroke-linejoin="round" stroke-linecap="round">',
      f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{w*scale:.3f}" height="{h*scale:.3f}"/>',
      f'<path d="{softened}"/>','</g>',
      f'<g fill="#111" font-family="DejaVu Sans,Arial,sans-serif" font-size="{art.FONT_MM:.6f}" text-anchor="middle" dominant-baseline="central">'
    ]
    labeled=[]
    for rid in range(1,nregions+1):
        if not fits[rid]:continue
        px,py=centers[rid]
        x=xoff+(px+.5)*scale
        y=yoff+(py+.5)*scale
        angle=int(rotations[rid])
        rotated=f' transform="rotate({angle} {x:.4f} {y:.4f})"' if angle else ''
        lines.append(f'<text x="{x:.4f}" y="{y:.4f}"{rotated}>{number_by_color[int(colors[rid])]}</text>')
        labeled.append(rid)
    lines.append('</g></svg>')
    svg=OUT/"lineart_matching_review.svg"
    svg.write_text('\n'.join(lines),encoding="utf-8")
    root=ET.parse(svg).getroot()
    printed=list(root.iter("{http://www.w3.org/2000/svg}text"))
    if len(printed)!=good:raise RuntimeError("Number count disagrees with actual visible SVG labels")
    for rid in labeled:
        px,py=centers[rid]
        if ids[py,px]!=rid:raise RuntimeError("Number center not inside its own region")
    cairosvg.svg2png(url=str(svg),write_to=str(OUT/"lineart_matching_preview.png"),output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(svg),write_to=str(OUT/"lineart_matching_screen.png"),output_width=1200,output_height=1800)
    audit={
      "color_proof":"color_preview_face_untouched.png",
      "lineart_proof":"lineart_matching_preview.png",
      "same_color_segmentation":True,
      "regions_total":nregions,
      "numbered":good,
      "without_number":len(bad),
      "unlabelled_ids":[int(i) for i in bad],
      "colors":len(palette),
      "font_pt":5.3,
      "stroke_mm":0.2,
      "source_face_preserved":True,
      "status":"INCOMPLETE_NOT_A_FINAL_PAINT_BY_NUMBERS"
    }
    (OUT/"matching_pair_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if not isinstance(v,list)},ensure_ascii=False))
if __name__=="__main__":
    build()
