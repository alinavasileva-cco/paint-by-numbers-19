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
    if face.shape[:2]!=(1402,1122) or (W,H)!=(2244,2804):
        raise RuntimeError(f"Unexpected dimensions: V9={W,H} V10={face.shape[:2]}")
    with PALETTE.open(encoding="utf-8-sig",newline="") as f:
        palette={int(row["Номер"]):tuple(int(row[k]) for k in ("R","G","B")) for row in csv.DictReader(f)}
    if set(palette)!=set(range(1,20)):raise RuntimeError("Palette must have exactly 19 colors")
    permissible=set(palette.values())
    for name,src in (("V9",base),("V10",face)):
        unique={tuple(v.tolist()) for v in np.unique(src.reshape(-1,3),axis=0)}
        if name=="V10" and unique-permissible:
            raise RuntimeError(f"{name}: {len(unique-permissible)} non-palette colors")
        # V9 is a SMOOTH visual preview with antialiased edge pixels.
        # Keep the exact unquantized V9 source for the conservation proof;
        # quantize only the separately printable 19-paint rendition.
    # Exact 3-landmark match: two eye centers and mouth, allowing affine scale
    # and shear to align the approved larger V10 face to the smaller V9 profile.
    source=np.float32([(515,822),(730,770),(720,1052)])
    # V9 landmarks were measured from a 1408x1760 scaled preview;
    # use native 2244x2804 source pixels without changing original resolution.
    sx,sy=W/1408.,H/1760.
    target=np.float32([(717*sx,484*sy),(816*sx,481*sy),(804*sx,573*sy)])
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
    cv2.fillPoly(restrict,[np.rint(target_poly * np.array([sx,sy])).astype(np.int32)],255)
    matte=cv2.bitwise_and(matte,restrict)
    # Preserve only native V9 DARK HAIR where it overlaps the very top
    # of the V10 replacement; never clip the new eye, nose, mouth or chin.
    # The V9 hair naturally hides the top of the transplanted face and
    # removes the artificial horizontal hair splice.
    old_rgb=base.astype(np.int16)
    dark_hair=(old_rgb[:,:,0]>old_rgb[:,:,1]*1.16)&(
        old_rgb[:,:,0]>old_rgb[:,:,2]*.84)&(old_rgb[:,:,0]<145)
    yy=np.arange(H)[:,None]
    hair_overdraw=dark_hair&(yy < round(485*sy))
    matte[hair_overdraw]=0
    merged=base.copy()
    merged[matte>0]=warped[matte>0]
    # QA fix: the first proof had a visible horizontal rectangle-shaped
    # color transition across the forehead. Extend the approved V10 skin
    # through contiguous V9 facial SKIN, stopping naturally at original hair
    # boundaries. This affects the face only, never scenery or clothing.
    v9_skin=base[round(540*sy),round(760*sx)].astype(np.int16)
    old_skin_distance=np.abs(base.astype(np.int16)-v9_skin[None,None,:]).sum(axis=2)
    old_skin=(old_skin_distance<=24).astype(np.uint8)
    forehead_region=np.zeros((H,W),np.uint8)
    forehead_poly=np.array([(653,424),(738,412),(807,429),(844,461),
                            (849,521),(832,574),(793,626),(718,632),
                            (659,572)],np.float32)
    forehead_poly=np.rint(forehead_poly*np.array([sx,sy])).astype(np.int32)
    cv2.fillPoly(forehead_region,[forehead_poly],255)
    extension=(forehead_region>0)&(old_skin>0)&(matte==0)
    merged[extension]=palette[17]
    matte[extension]=255
    changed=np.any(merged!=base,axis=2)
    nchanged=int(changed.sum())
    if not 3000 <= nchanged <= 200000:
        raise RuntimeError(f"Too many/few pixels modified by face transplant: {nchanged}")
    ys,xs=np.where(changed)
    bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    if bbox[0]<int(635*sx) or bbox[1]<int(410*sy) or bbox[2]>int(860*sx) or bbox[3]>int(645*sy):
        raise RuntimeError(f"Changes escaped facial vicinity: {bbox}")
    # Keep an exact V9-preserving visual composite as evidence: every pixel
    # outside the V10 face mask is unchanged from the user's V9 source.
    Image.fromarray(merged).save(OUT/"full_color_v11_source_composite.png",optimize=True)
    # V9 smooth preview contains antialias intermediary shades (thousands of
    # RGB combinations). Round only the printable copy to the EXISTING 19
    # THE CASE paint colors. No new colors or AI rendering.
    sample, inv=np.unique(merged.reshape(-1,3),axis=0,return_inverse=True)
    p19=np.array([palette[i] for i in range(1,20)],dtype=np.int32)
    d=((sample.astype(np.int32)[:,None,:]-p19[None,:,:])**2).sum(axis=2)
    lut=np.argmin(d,axis=1).astype(np.uint8)
    printed=p19[lut[inv]].astype(np.uint8).reshape(H,W,3)
    merged_colors={tuple(c.tolist()) for c in np.unique(printed.reshape(-1,3),axis=0)}
    if merged_colors-permissible or len(merged_colors)>19:
        raise RuntimeError("Print copy is not restricted to 19 palette colors")
    art=OUT/"full_color_v11.png"
    Image.fromarray(printed).save(art,optimize=True)
    Image.fromarray(matte).save(OUT/"facial_edit_mask.png",optimize=True)
    # QA: side by side crops, original vs new, at native pixels.
    crop=tuple(round(v*(sx if j%2==0 else sy)) for j,v in enumerate((600,383,900,700)))
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
      "skin_edge_extension_pixels":int(extension.sum()),
      "v9_exact_unmodified_outside_face_in":"full_color_v11_source_composite.png",
      "printable_19_color_render":"full_color_v11.png",
      "antialias_quantization_changed_pixels":int(np.any(printed!=merged,axis=2).sum()),
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
