#!/usr/bin/env python3
"""Visual QA crops only; source and numbered work are unchanged."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
p=Path("face_v7")
old=Image.open("face_v6/exact_previous_face/color_segmentation_review.png").convert("RGB")
new=Image.open(p/"color_artist_preview.png").convert("RGB")
if old.size!=new.size:raise RuntimeError("Mismatched dimensions")
crop=(450,310,680,512)
src=old.crop(crop)
dst=new.crop(crop)
scale=3
src=src.resize((src.width*scale,src.height*scale),Image.Resampling.LANCZOS)
dst=dst.resize((dst.width*scale,dst.height*scale),Image.Resampling.LANCZOS)
page=Image.new("RGB",(src.width*2+24,src.height+65),"white")
page.paste(src,(0,65));page.paste(dst,(src.width+24,65))
d=ImageDraw.Draw(page)
d.text((12,20),"V6: face before",fill=(0,0,0))
d.text((src.width+36,20),"V7: cleaned face",fill=(0,0,0))
page.save(p/"face_before_after_final.png",optimize=True)
dst.save(p/"face_detail_final.png",optimize=True)
print(json.dumps({"cropped_area_px":crop,"same_background":True,"not_color_segmentation":True}))
