"""Register the approved V10 facial regions onto the existing V11 scene."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

REFERENCE = Path(__file__).with_name("approved_face_regions.png")
REFERENCE_SHA = "3db8b5b23149287831b5ba8fe9f3624fe24a9b5b"
SCALE = .272
OFFSET = (439, 168)
PATCH = (430, 278, 700, 557)
# V10 region order, including the approved 19 → 17 change under the far eye.
PAINTS = np.array([
    0, 5, 7, 2, 16, 5, 16, 16, 5, 7, 11, 16, 16, 16, 17, 2, 17, 16, 17, 4, 17,
    9, 9, 16, 9, 12, 1, 9, 16, 4, 4, 1, 4, 17, 18, 3, 19, 7, 14, 11, 7, 18, 17,
    16, 7, 5, 9, 17, 4, 14, 4, 10, 4, 13, 17, 4, 19, 19, 8, 6, 16, 17, 9, 4, 19,
    3, 17, 16, 17, 9, 15, 4, 3, 9, 16, 16, 4, 5, 9, 4, 5, 18, 16, 18, 12, 10, 17, 10,
], dtype=np.uint8)
HEAD = (
    19, 20, 21, 22, 24, 27, 29, 30, 32, 39, 40, 42, 46, 47, 48, 50, 51, 52,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73,
    74, 76, 78, 79,
)
# Local cut routed along the hair flow and the junction between face and collar.
JOIN = (
    (451, 393), (475, 378), (502, 360), (529, 342), (558, 323), (584, 307),
    (609, 290), (632, 282), (647, 285), (654, 303), (656, 325), (671, 340),
    (687, 354), (686, 379), (691, 413), (697, 448), (695, 482), (682, 515),
    (659, 543), (620, 555), (596, 552), (591, 526), (564, 512), (536, 507),
    (507, 507), (481, 500), (463, 488), (467, 462), (453, 444), (449, 420),
)
FACE_FIELDS = (42, 47, 54, 56, 57, 58, 59, 60, 61, 63, 64, 66, 67, 68, 70, 74)
EYE_MASKS = (
    ((356, 843), (409, 805), (477, 763), (529, 753), (579, 770), (597, 815),
     (567, 850), (551, 887), (485, 905), (443, 893), (410, 864), (356, 855)),
    ((701, 753), (724, 720), (748, 698), (782, 704), (821, 716), (811, 757),
     (781, 786), (773, 813), (778, 828), (747, 838), (715, 831), (700, 809)),
)
# The fine strokes carry the original nostril, lid crease and inner ear.
DETAIL_PATHS = (
    "M746 968 C750 962 759 956 765 954",
    "M443 779 C460 768 477 760 496 756",
    "M714 729 C725 707 747 686 762 694",
    "M713 1052 C723 1049 733 1047 741 1047",
    "M109 958 C95 936 88 911 96 894 C105 873 128 866 152 879",
)


def reference_planes():
    regions = np.asarray(Image.open(REFERENCE).convert("L"))
    if regions.shape != (1402, 1122) or set(np.unique(regions)) != set(range(len(PAINTS))):
        raise ValueError("Unexpected approved face region map")
    # Extend pigment to the center of each ink line. Keep the filled lashes.
    indices = distance_transform_edt(regions == 0, return_distances=False, return_indices=True)
    nearest = regions[tuple(indices)]
    paints = PAINTS[nearest]
    eye_mask = np.zeros(regions.shape, np.uint8)
    for polygon in EYE_MASKS:
        cv2.fillPoly(eye_mask, [np.asarray(polygon, dtype=np.int32)], 1)
    paints[(regions == 0) & (eye_mask != 0)] = 4
    # Rounded catchlights retain the approved locations at the smaller print scale.
    highlight = np.zeros(regions.shape, np.uint8)
    cv2.ellipse(highlight, (524, 812), (18, 17), 0, 0, 360, 1, -1)
    paints[(highlight > 0) & (eye_mask > 0)] = 19
    # Keep the smaller iris paintable without changing the outer eyelid.
    iris = cv2.dilate(np.uint8(regions == 60), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 15)))
    eye_interior = cv2.erode(
        np.uint8((eye_mask > 0) & np.isin(paints, (4, 8, 16, 19))),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    lower_iris = (iris > 0) & (eye_interior > 0) & (paints != 19)
    rim = cv2.dilate(
        np.uint8(lower_iris), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )
    paints[(rim > 0) & (eye_mask > 0) & ~np.isin(paints, (8, 19))] = 4
    paints[lower_iris] = 16
    upper = cv2.dilate(
        np.uint8(regions == 58), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 13))
    )
    paints[(upper > 0) & (eye_interior > 0) & ~np.isin(regions, (56, 57))] = 8
    far_glint = np.zeros(regions.shape, np.uint8)
    cv2.ellipse(far_glint, (743, 746), (19, 20), 0, 0, 360, 1, -1)
    paints[(far_glint > 0) & (eye_mask > 0)] = 19
    return paints, nearest, eye_mask


def transfer_face(base, samples):
    paints, areas, eye_mask = reference_planes()
    x0, y0, x1, y1 = PATCH
    part = base[y0*samples:y1*samples, x0*samples:x1*samples]
    size = (part.shape[1], part.shape[0])
    scale = SCALE * samples
    mapping = np.array([[scale, 0, (OFFSET[0] - x0) * samples],
                        [0, scale, (OFFSET[1] - y0) * samples]])
    projected = cv2.warpAffine(paints, mapping, size, flags=cv2.INTER_NEAREST)
    area = cv2.warpAffine(areas, mapping, size, flags=cv2.INTER_NEAREST)
    eye = cv2.warpAffine(eye_mask, mapping, size, flags=cv2.INTER_NEAREST) != 0
    gate = np.zeros(part.shape, np.uint8)
    polygon = (np.asarray(JOIN) - [x0, y0]) * samples
    cv2.fillPoly(gate, [polygon.astype(np.int32)], 1)
    row, col = np.indices(part.shape)
    art_y = row / samples + y0

    cold = np.isin(part, (1, 2, 3, 5, 7))
    # Preserve existing hair colors wherever the new drawing still shows hair.
    existing_hair = np.isin(part, (4, 5, 6, 7, 8, 9, 10, 11))
    anatomy = np.isin(area, FACE_FIELDS) | eye
    use_face = (gate > 0) & np.isin(area, HEAD) & (anatomy | ~existing_hair)
    # Preserve the outer head silhouette above the eyes and the neck at the collar.
    use_face &= ~((art_y < 350) & cold & (col / samples + x0 > 643))
    clear = (gate > 0) & (area == 15) & ~cold & (art_y > 370)
    preserve_neck = np.isin(part, (9, 10, 11, 12, 13, 15, 18))
    join_neck = (gate > 0) & (area == 75) & ~preserve_neck & (art_y < 498)
    candidate = part.copy()
    candidate[use_face] = projected[use_face]
    candidate[clear] = 2
    candidate[join_neck] = 13

    # Smooth the sampled seam locally. The previous indexed artwork is retained elsewhere.
    editable = ((candidate != part) & (gate > 0)).astype(np.uint8)
    editable = cv2.dilate(editable, np.ones((5, 5), np.uint8)) > 0
    editable &= gate > 0
    best = np.full(size[::-1], -1.0, np.float32)
    winner = candidate.copy()
    for paint in range(1, 20):
        blur = cv2.GaussianBlur(np.float32(candidate == paint), (0, 0), .56 * samples)
        better = blur > best
        best[better] = blur[better]
        winner[better] = paint
    revised = base.copy()
    patch = revised[y0*samples:y1*samples, x0*samples:x1*samples]
    patch[editable] = winner[editable]

    return revised
