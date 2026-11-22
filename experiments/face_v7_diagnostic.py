from pathlib import Path
import sys, json
import numpy as np
from PIL import Image,ImageDraw
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_artistic_v4 as art
import draw_face_art_v5 as v5
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base
OUT=Path("face_v7_diagnostic")
OUT.mkdir(exist_ok=True)
idx,palette,numbers,source,scale=art.prepare()
seg=v5.indexed_png("face_v6/exact_previous_face/color_segmentation_review.png",palette)
ids,colors,stats=base.label_components(seg,19)
fits,centers,area,dims,rots=art.label_fit(ids,colors,numbers,scale)
bad=[int(i) for i in np.flatnonzero(~fits[1:])+1]
ys,xs=np.indices(ids.shape)
zones={
 "face_main":(420,300,710,555),
 "face_extended":(360,200,740,590),
 "main_face_only":(450,355,690,510),
 "whole_upper_person":(180,170,740,800)
}
records=[]
for rid in bad:
 y,x=np.nonzero(ids==rid); ymin,ymax=int(y.min()),int(y.max());xmin,xmax=int(x.min()),int(x.max())
 records.append(dict(id=rid,area=int(area[rid]),bbox=[xmin,ymin,xmax,ymax],color_number=int(numbers[int(colors[rid])]),
      centroid=[float(x.mean()),float(y.mean())]))
counts={}
for name,(x0,y0,x1,y1) in zones.items():
 counts[name]={"entirely_inside":sum(z["bbox"][0]>=x0 and z["bbox"][2]<x1 and z["bbox"][1]>=y0 and z["bbox"][3]<y1 for z in records),
              "intersects":sum(not(z["bbox"][2]<x0 or z["bbox"][0]>=x1 or z["bbox"][3]<y0 or z["bbox"][1]>=y1) for z in records)}
summary={"total_regions":int(ids.max()),"unfit":len(bad),"counts":counts,
    "outside_face_extended":[r for r in records if not (r["bbox"][0]>=360 and r["bbox"][2]<740 and r["bbox"][1]>=200 and r["bbox"][3]<590)],
    "bad_region_records":records}
(OUT/"diagnostic.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
preview=Image.fromarray(palette[seg]).convert("RGB")
d=ImageDraw.Draw(preview)
for r in records:
 x0,y0,x1,y1=r["bbox"]
 d.rectangle((x0,y0,x1,y1),outline=(255,0,122),width=1)
d.rectangle(zones["face_extended"],outline=(0,255,190),width=2)
preview.save(OUT/"unfit_regions.png")
print("REGION DIAGNOSTIC",json.dumps({k:v for k,v in summary.items() if k!="bad_region_records"},ensure_ascii=False)[:10000],flush=True)
