#!/usr/bin/env python3
"""Non-generative composite: exact full V9 artwork + approved V10 facial design.
The V9 original is never overwritten. V10 face is aligned by approved landmarks,
and only a constrained facial region is replaced. All output colors come from the
pre-existing 19-color palette (nearest-neighbour transform; zero generated art).
"""
import json, hashlib, csv
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent.parent
V9=ROOT/"face_v9_color_only/portrait_v9_smooth_preview.png"
V10=ROOT/"face_reference_v10/output/color_assignment_review.png"
PALETTE=ROOT/"review/palette.csv"
OUT=ROOT/"full_v11_v9_with_v10_face"
OUT.mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if not V9.is_file() or not V10.is_file():raise RuntimeError("Missing original V9 or approved V10 source")
    base=np.asarray(Image.open(V9).convert("RGB"))
    face=np.asarray(Image.open(V10).convert("RGB"))
    H,W=base.shape[:2]
    if face.shape[:2]!=(1402,1122) or (W,H)!=(1408,1760):
        raise RuntimeError(f"Unexpected dimensions: V9={W,H} V10={face.shape[:2]}")
    with PALETTE.open(encoding="utf-8-sig",newline="") as f:
        palette={int(row["Номер"]):tuple(int(row[k]) for k in ("R","G","B")) for row in csv.DictReader(f)}
    if set(palette)!=set(range(1,20)):raise RuntimeError("Palette must have exactly 19 colors")
    permissible=set(palette.values())
    for name,src in (("V9",base),("V10",face)):
        unique={tuple(v.tolist()) for v in np.unique(src.reshape(-1,3),axis=0)}
        if unique-permissible:raise RuntimeError(f"{name}: {len(unique-permissible)} non-palette colors")
    # Exact 3-landmark match: two eye centers and mouth, allowing affine scale
    # and shear to align the approved larger V10 face to the smaller V9 profile.
    source=np.float32([(515,822),(730,770),(720,1052)])
    target=np.float32([(717,484),(816,481),(804,573)])
    affine=cv2.getAffineTransform(source,target)
    warped=cv2.warpAffine(face,affine,(W,H),flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=palette[2])
    # Constrain edit to a facial patch only, leaving city, lanterns, water,
    # flower ornament, the kimono and the wider hair unchanged.
    facial_polygon=np.array([
        (360,748),(445,726),(555,710),(670,685),(750,712),(780,778),
        (805,878),(818,958),(804,1050),(751,1135),(676,1224),
        (579,1205),(483,1160),(394,1085),(362,980)],np.float32)
    source_matte=np.zeros((face.shape[0],face.shape[1]),np.uint8)
    cv2.fillPoly(source_matte,[facial_polygon.astype(np.int32)],255)
    matte=cv2.warpAffine(source_matte,affine,(W,H),flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    # Restrict to source V9 face vicinity; prevents its hair / background being changed.
    restrict=np.zeros((H,W),np.uint8)
    target_poly=np.array([(659,425),(737,420),(801,430),(839,466),
                          (851,523),(841,571),(813,614),(775,635),
                          (720,617),(671,574),(647,514)],np.int32)
    cv2.fillPoly(restrict,[target_poly],255)
    matte=cv2.bitwise_and(matte,restrict)
    merged=base.copy()
    merged[matte>0]=warped[matte>0]
    changed=np.any(merged!=base,axis=2)
    nchanged=int(changed.sum())
    if not 3000 <= nchanged <= 80000:
        raise RuntimeError(f"Too many/few pixels modified by face transplant: {nchanged}")
    ys,xs=np.where(changed)
    bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    if bbox[0]<635 or bbox[1]<410 or bbox[2]>860 or bbox[3]>645:
        raise RuntimeError(f"Changes escaped facial vicinity: {bbox}")
    merged_colors={tuple(c.tolist()) for c in np.unique(merged.reshape(-1,3),axis=0)}
    if merged_colors-permissible:raise RuntimeError("Merged painting has non-palette colors")
    art=OUT/"full_color_v11.png"
    Image.fromarray(merged).save(art,optimize=True)
    Image.fromarray(matte).save(OUT/"facial_edit_mask.png",optimize=True)
    # QA: side by side crops, original vs new, at native pixels.
    crop=(600,383,900,700)
    old=Image.fromarray(base).crop(crop)
    new=Image.fromarray(merged).crop(crop)
    proof=Image.new("RGB",(old.width*2,old.height+32),"white")
    proof.paste(old,(0,32));proof.paste(new,(old.width,32))
    d=ImageDraw.Draw(proof)
    d.text((12,8),"V9 ORIGINAL",fill="#18101B")
    d.text((old.width+12,8),"V11: APPROVED V10 FACE",fill="#18101B")
    proof.save(OUT/"face_before_after.png")
    audit={
      "status":"REVIEW_REQUIRED","composition":"V9 exact pixels outside face matte",
      "base_v9_path":str(V9.relative_to(ROOT)),"base_v9_sha256":sha(V9),
      "approved_v10_path":str(V10.relative_to(ROOT)),"approved_v10_sha256":sha(V10),
      "original_dimensions":[W,H],"colors_max":19,"colors_actual":len(merged_colors),
      "modified_pixels":nchanged,"modified_fraction":round(nchanged/(W*H),6),
      "changed_bbox_xyxy":bbox,"only_face_edited":True,
      "v10_under_right_eye_color":17,
      "source_landmarks":source.astype(int).tolist(),
      "target_landmarks":target.astype(int).tolist(),
      "affine":affine.tolist(),
      "approved_full_art":False,
      "note":"Automated compositing proof. Requires visual QA before releasing as final paint-by-numbers."
    }
    (OUT/"audit_composite.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))
if __name__=="__main__":main()
