#!/usr/bin/env python3
"""Restore the complete ORIGINAL face from the approved 1501-area reference.

This is a truth-in-advertising art experiment: the only acceptable outcome is
the original facial shape and gaze, not a mechanically invented eye/ear.
Never present the restored preview as a numbered SVG unless every cell is
independently rebuilt and checked.
"""
import sys, json, hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_artistic_v4 as art
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"scripts"))
import build as base

ROOT=Path("review_art_v4")
OUT=Path("review_face_original")
OUT.mkdir(parents=True,exist_ok=True)

def facial_protection(h,w):
    """Figure-specific mask; 1122x1402 artwork coordinates, source-derived."""
    sx=w/1122.; sy=h/1402.
    mask=np.zeros((h,w),np.uint8)
    def shape(points):
        coords=np.rint(np.array(points)*[sx,sy]).astype(np.int32)
        cv2.fillPoly(mask,[coords],255)
    # The WHOLE profile from forehead through lips, chin and neckline:
    shape([(523,358),(546,345),(580,333),(616,334),(644,352),
           (660,381),(672,415),(664,443),(647,465),(622,486),
           (597,507),(553,503),(521,480),(509,448),(501,421),
           (506,388)])
    # Ear, sideburns and connection to temple (not artificially drawn).
    shape([(450,405),(459,388),(482,383),(503,398),
           (515,432),(506,451),(484,457),(462,445),(448,427)])
    # Fringe overlaps the original upper eyelids; save the hairline as drawn.
    shape([(521,363),(542,331),(591,312),(640,314),
           (663,349),(667,384),(637,392),(598,375),(555,385)])
    return mask>0

def main():
    base_idx,palette,numbers,source_bytes,scale=art.prepare()
    img=np.asarray(Image.open(ROOT/"color_segmentation_review.png").convert("RGB"))
    look={tuple(map(int,color)):i for i,color in enumerate(palette)}
    unique,inverse=np.unique(img.reshape(-1,3),axis=0,return_inverse=True)
    if any(tuple(map(int,rgb)) not in look for rgb in unique):
        raise RuntimeError("V4 uses colors absent from original palette")
    color_idx=np.array([look[tuple(map(int,rgb))] for rgb in unique],np.uint8)[inverse].reshape(base_idx.shape)
    mask=facial_protection(*base_idx.shape)
    out=color_idx.copy()
    out[mask]=base_idx[mask]
    original_rgb=palette[base_idx]
    result_rgb=palette[out]
    draft_rgb=palette[color_idx]
    Image.fromarray(result_rgb).save(OUT/"color_preview_face_untouched.png")
    # Mark restoration boundary only on diagnostic image, not on artwork.
    guide=result_rgb.copy()
    boundary=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,
                              np.ones((3,3),np.uint8))
    guide[boundary>0]=(240,30,100)
    Image.fromarray(guide).save(OUT/"protected_face_boundary_diagnostic.png")
    # Preserve original colors and gaze on left and show facial restoration on right.
    left,top,right,bottom=405,292,706,544
    left=max(0,int(left*img.shape[1]/1122))
    right=min(img.shape[1],int(right*img.shape[1]/1122))
    top=max(0,int(top*img.shape[0]/1402))
    bottom=min(img.shape[0],int(bottom*img.shape[0]/1402))
    oh,ow=bottom-top,right-left
    comparison=Image.new("RGB",(ow*3,oh))
    comparison.paste(Image.fromarray(original_rgb[top:bottom,left:right]),(0,0))
    comparison.paste(Image.fromarray(draft_rgb[top:bottom,left:right]),(ow,0))
    comparison.paste(Image.fromarray(result_rgb[top:bottom,left:right]),(ow*2,0))
    comparison.save(OUT/"face_original_v4_restored_comparison.png")

    # Recompute connected regions AFTER restoration; the V4 SVG labels are no
    # longer valid and are NEVER reused for this image.
    ids,colors,_=base.label_components(out,len(palette))
    good,centers,areas,dims,rotation=art.label_fit(ids,colors,numbers,scale)
    failed=np.flatnonzero(~good[1:])+1
    within_face=set(int(x) for x in np.unique(ids[mask]))
    face_failed=[int(x) for x in failed if int(x) in within_face]
    meta={
      "source":"main/review/color_segmentation_review.png",
      "candidate":"art-minimal-merges-v4/review_art_v4/color_segmentation_review.png",
      "artifact_kind":"color proof; do not treat as final numbered paint-by-numbers",
      "facial_choice":"leave original face entirely untouched",
      "protected_original_face_pixels":int(mask.sum()),
      "changed_pixels_inside_face_vs_original":int(np.count_nonzero((out!=base_idx)&mask)),
      "changed_pixels_elsewhere_vs_v4":int(np.count_nonzero((out!=color_idx)&~mask)),
      "facial_pixels_restored_from_reference":int(np.count_nonzero((color_idx!=base_idx)&mask)),
      "new_connected_paint_regions":int(ids.max()),
      "regions_that_can_fit_5_3_pt":int(good[1:].sum()),
      "regions_that_cannot_fit_5_3_pt":int(len(failed)),
      "unfit_region_ids_touching_protected_face":face_failed[:80],
      "unfit_region_count_touching_protected_face":len(face_failed),
      "source_png_sha256":hashlib.sha256(source_bytes).hexdigest(),
      "source_png_unchanged":True,
      "requires_numbering_rebuild":True,
      "artistic_approval":"PENDING",
      "status":"FACE_PRESERVED_NUMBERING_PENDING" if len(failed) else "FACE_PRESERVED_READY_FOR_NEW_SVG"
    }
    (OUT/"audit.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in meta.items() if not isinstance(v,list)},ensure_ascii=False))

if __name__=="__main__":
    main()
