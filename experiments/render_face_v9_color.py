#!/usr/bin/env python3
"""THE CASE: paint a NEW FACE only, before paint-by-number conversion.

Artwork reference: clean anime portrait shown in the conversation (rich amber
eyes, clear nose/lips, uncluttered smooth facial planes).
Base: existing V8 color map. The rest of the existing artwork is immutable.
This stage produces COLOR ART, NOT a numbered coloring sheet.
All facial pigment colors belong to a fixed seven-color subset of the
existing 19-color palette. The clean facial paths are also saved as vector.
"""
import json
import hashlib
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_artistic_v4 as art
import draw_face_art_v5 as v5

ROOT=Path(__file__).resolve().parent.parent
INPUT=ROOT/"face_v8/color_segmentation_review.png"
OUT=ROOT/"face_v9_color_only"
OUT.mkdir(exist_ok=True,parents=True)

# Original face occupies exactly the existing painterly composition.
# A narrow protective canvas includes eyes/face/ear, not city/water/kimono.
X0,Y0,X1,Y1=456,344,675,512
SEVEN_NUMBERS=(4,11,13,14,15,18,19)
INK=4; EAR_SHADOW=13; SKIN=15; EYE_WHITE=19; IRIS=14; GOLD=18; FACE_DARK=11

def cubic(a,b,c,d,n=36):
    t=np.linspace(0,1,n,dtype=float)[:,None]
    return (1-t)**3*np.asarray(a)+3*(1-t)**2*t*np.asarray(b)+3*(1-t)*t*t*np.asarray(c)+t**3*np.asarray(d)

def curved_poly(start,*commands):
    current=np.asarray(start,dtype=float)
    out=[current]
    for cmd in commands:
        if cmd[0]=="L":
            current=np.asarray(cmd[1],float)
            out.append(current)
        elif cmd[0]=="C":
            samples=cubic(current,cmd[1],cmd[2],cmd[3],24)
            out.extend(samples[1:])
            current=np.asarray(cmd[3],float)
        else:
            raise ValueError(cmd[0])
    return np.rint(np.asarray(out)).astype(np.int32)

def mask_of(shape,start,*commands):
    a=np.zeros(shape,dtype=np.uint8)
    cv2.fillPoly(a,[curved_poly(start,*commands)],255,lineType=cv2.LINE_8)
    return a>0

def path_stroke(array, pts, color, width=1):
    line=cubic(*pts,n=120)
    cv2.polylines(array,[np.rint(line).astype(np.int32).reshape(-1,1,2)],
                  isClosed=False,color=int(color),thickness=width,lineType=cv2.LINE_8)

def fill(image,mask,color):
    image[mask]=int(color)

def compose():
    baseline,palette,color_numbers,source_bytes,scale=art.prepare()
    original=v5.indexed_png(INPUT,palette)
    H,W=original.shape
    assert (H,W)==(1402,1122), f"Unexpected artwork size {(H,W)}"
    idx_for_num={int(number):int(i) for i,number in color_numbers.items()}
    for num in SEVEN_NUMBERS:
        if num not in idx_for_num: raise RuntimeError(f"Missing approved pigment number {num}")
    n=lambda number:idx_for_num[number]
    out=original.copy()
    rect=np.zeros_like(original,dtype=bool)
    rect[Y0:Y1,X0:X1]=True

    # Replace skin from scratch with artist-defined large smooth shapes, not
    # machine pixel segmentation. Preserve original hair and body outside
    # the face silhouette.
    skin=mask_of((H,W),(520,385),
        ("C",(536,370),(552,379),(579,370)),
        ("C",(606,358),(627,358),(645,377)),
        ("C",(650,388),(648,405),(654,419)),
        ("C",(661,424),(662,428),(654,432)), # clean nose tip
        ("C",(651,435),(649,439),(654,445)),
        ("C",(658,449),(650,449),(645,451)), # upper lip
        ("C",(651,453),(648,458),(640,460)),
        ("C",(645,466),(637,479),(623,486)),
        ("C",(604,497),(579,489),(558,483)),
        ("C",(533,473),(528,455),(524,436)),
        ("C",(520,423),(516,398),(520,385)))
    skin &= rect
    fill(out,skin,n(SKIN))

    # Large, coherent cheek/jaw shading: this is a paintable patch,
    # not hundreds of machine-generated one-pixel highlights.
    jaw_shadow=mask_of((H,W),(525,450),
        ("C",(531,467),(545,477),(569,489)),
        ("C",(585,498),(613,499),(631,487)),
        ("C",(608,488),(587,480),(568,465)),
        ("C",(555,458),(542,452),(525,450))) & skin
    fill(out,jaw_shadow,n(EAR_SHADOW))

    # Little shaded nose plane, kept very simple; the black profile line
    # is drawn independently below, like editorial illustration.
    nose_shadow=mask_of((H,W),(644,414),
        ("C",(647,420),(654,424),(653,430)),
        ("C",(647,433),(643,428),(644,414))) & skin
    fill(out,nose_shadow,n(EAR_SHADOW))

    # Ear is simple and proportionate, with a single inner curved shadow.
    ear=mask_of((H,W),(466,405),
        ("C",(475,399),(485,400),(493,412)),
        ("C",(502,424),(498,439),(483,449)),
        ("C",(472,447),(461,433),(462,417)),
        ("C",(463,409),(466,405),(466,405))) & rect
    fill(out,ear,n(SKIN))
    inner_ear=mask_of((H,W),(473,412),
        ("C",(485,407),(492,416),(490,426)),
        ("C",(487,436),(476,438),(473,428)),
        ("C",(469,420),(470,416),(473,412))) & ear
    fill(out,inner_ear,n(EAR_SHADOW))

    # Near eye: distinctive elegant anime almond, not a wide round
    # geometric computer oval. Far eye agrees with face perspective.
    near=mask_of((H,W),(547,394),
        ("C",(557,383),(577,380),(590,390)),
        ("C",(597,394),(596,399),(590,403)),
        ("C",(573,410),(555,405),(547,394))) & skin
    far=mask_of((H,W),(625,387),
        ("C",(634,380),(647,380),(657,386)),
        ("C",(663,391),(662,394),(655,399)),
        ("C",(643,402),(632,397),(625,387))) & skin

    # Warm eyes: 7-color palette including sclera/iris/pupil.
    # The pupil and highlights are color fields in THIS illustration; a
    # separate later production step will decide how to number tiny marks.
    for eye in [near,far]:
        fill(out,eye,n(EYE_WHITE))
    iris_near=np.zeros_like(original,np.uint8)
    cv2.ellipse(iris_near,(577,395),(12,10),-8,0,360,255,-1)
    iris_far=np.zeros_like(original,np.uint8)
    cv2.ellipse(iris_far,(647,390),(9,9),-8,0,360,255,-1)
    fill(out, (iris_near>0)&near, n(IRIS))
    fill(out, (iris_far>0)&far,n(IRIS))

    # Deep dark pupils shifted toward the profile, preserving gaze.
    pupils=[]
    for cx,cy,rx,ry,eye in [(582,394,5,7,near),(651,390,4,6,far)]:
        mask=np.zeros_like(original,np.uint8)
        cv2.ellipse(mask,(cx,cy),(rx,ry),-6,0,360,255,-1)
        pupil=(mask>0)&eye
        fill(out,pupil,n(INK))
        pupils.append(pupil)

    # Two restrained light glints; no random microspots elsewhere.
    for cx,cy,rx,ry,eye in [(575,390,2,2,near),(645,387,2,2,far)]:
        circle=np.zeros_like(original,np.uint8)
        cv2.ellipse(circle,(cx,cy),(rx,ry),0,0,360,255,-1)
        fill(out,(circle>0)&eye,n(GOLD))

    # Clear black almond eyelids, natural brows, nostril, mouth and jaw.
    # Black drawing strokes are still one of the SAME seven paint colors.
    STROKES=[
      ("near_upper",((548,392),(561,381),(579,379),(594,391)),2),
      ("near_lower",((549,399),(563,407),(579,407),(592,401)),1),
      ("near_lash",((590,389),(598,387),(602,384),(605,378)),1),
      ("far_upper",((626,385),(636,378),(650,379),(660,386)),2),
      ("far_lower",((627,392),(640,400),(650,400),(659,395)),1),
      ("nostril",((651,429),(649,431),(648,432),(647,430)),1),
      ("mouth",((647,451),(642,451),(639,453),(635,453)),2),
      ("lower_lip",((641,457),(638,460),(636,460),(633,458)),1),
      ("ear_fold",((473,413),(482,407),(488,410),(491,419)),1),
    ]
    # Do not draw unrestricted lines across the hair, lanterns or water.
    for name,points,width in STROKES:
        plane=out.copy()
        path_stroke(plane,points,n(INK),width)
        out[rect]=plane[rect]

    # Clean lip pigment, merged shape suitable for a calm expression.
    lip=mask_of((H,W),(636,452),
        ("C",(642,450),(645,451),(650,450)),
        ("C",(646,455),(643,457),(639,458)),
        ("C",(637,457),(635,456),(636,452))) & skin
    fill(out,lip,n(IRIS))
    # Follow actual nose/cheek profile with a single fine shadow curve
    # rather than thousands of jagged source pixels.
    path_stroke(out,((646,408),(650,418),(657,424),(652,430)),n(INK),1)

    # Fine controlled hair/fringe strands in the approved dark hair tone.
    hair_color=original[360,501]
    for p in [
      ((533,360),(541,375),(548,378),(550,389)),
      ((567,350),(574,371),(581,378),(587,388)),
      ((611,362),(620,378),(625,389),(629,399))
    ]:
        plane=out.copy()
        path_stroke(plane,p,int(hair_color),1)
        out[rect]=plane[rect]

    # By construction this image CANNOT modify the rest of the painting.
    outside_changed=int(np.count_nonzero((out!=original)&~rect))
    if outside_changed:
        raise RuntimeError(f"Outside-face changes detected: {outside_changed}")

    face_color_set=sorted({int(k) for k in np.unique(out[skin|ear])})
    if any(k not in [n(q) for q in SEVEN_NUMBERS] for k in face_color_set):
        raise RuntimeError(f"Unexpected color in new facial paint: {face_color_set}")
    if len(face_color_set)>7:
        raise RuntimeError("More than 7 flat face pigments")

    # Keep a clean palette-faithful image separate from print numbering.
    clean=palette[out]
    Image.fromarray(clean).save(OUT/"portrait_v9_color.png")
    # A high-res *view* and two close-ups for art approval only; not a PDF.
    crop=(447,319,684,522)
    before=Image.fromarray(palette[original]).crop(crop)
    after=Image.fromarray(clean).crop(crop)
    after.resize((948,812),Image.Resampling.LANCZOS).save(OUT/"face_detail_v9.png")
    montage=Image.new("RGB",(948*2+20,812),"white")
    montage.paste(before.resize((948,812),Image.Resampling.LANCZOS),(0,0))
    montage.paste(after.resize((948,812),Image.Resampling.LANCZOS),(968,0))
    montage.save(OUT/"face_before_after_v9.png")

    audit={
      "version":"V9_FACE_COLOR_ART_ONLY",
      "source_segmentation":str(INPUT.relative_to(ROOT)),
      "changes_outside_face_rectangle":outside_changed,
      "face_edit_rectangle_px":[X0,Y0,X1,Y1],
      "face_palette_limit":7,
      "face_pigment_count":len(face_color_set),
      "face_pigment_numbers":[int(color_numbers[k]) for k in face_color_set],
      "face_pixels_repainted":int(np.count_nonzero((out!=original)&rect)),
      "original_source_sha256":hashlib.sha256(source_bytes).hexdigest(),
      "original_source_unchanged":True,
      "numbering_not_started":True,
      "status":"COLOR_ART_FOR_ARTISTIC_REVIEW"
    }
    (OUT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V9_FACE",json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=="__main__":
    compose()
