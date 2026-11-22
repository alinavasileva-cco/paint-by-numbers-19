"""Apply the hand-drawn portrait, folds and embroidery to the V11 paint map."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from geometry import components, ink_interiors

SVG = "http://www.w3.org/2000/svg"
ART = Path(__file__).with_name("art_refinement.svg")
EDIT_BOUNDS = (259, 202, 698, 979)
SKIN_SEEDS = ((587, 449), (567, 371), (473, 410))
NEAR_EYE_SEEDS = ((548, 397), (572, 390), (574, 404), (581, 389))
FAR_EYE_SEEDS = ((654, 368), (641, 371), (636, 379), (638, 388))
# The earlier lip and concha shared pigments with other warm objects in the scene.
MOUTH_SEED = (636, 457)


def render(root, shape):
    png = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=shape[1], output_height=shape[0])
    return np.asarray(Image.open(BytesIO(png)).convert("RGBA"))


def selected_layer(root, id_name, shape):
    frame = ET.Element(f"{{{SVG}}}svg", root.attrib)
    frame.append(deepcopy(root.find(f"{{{SVG}}}defs")))
    item = root.find(f".//*[@id='{id_name}']")
    frame.append(deepcopy(item))
    return render(frame, shape)


def refine_artwork(base, rows, samples):
    original_ids, _, _ = components(base, len(rows))

    def fields_at(seeds):
        return np.isin(original_ids, [original_ids[y*samples, x*samples] for x, y in seeds])

    result = base.copy()
    skin = fields_at(SKIN_SEEDS)
    if not np.all(base[skin] == 17):
        raise ValueError("The V11 skin reference has changed")
    result[skin | fields_at(NEAR_EYE_SEEDS + (MOUTH_SEED,))] = 15

    root = ET.parse(ART).getroot()
    profile = selected_layer(root, "far-profile", base.shape)[..., 3] >= 128
    far_art = selected_layer(root, "far-eye", base.shape)[..., 3] >= 128
    outside_eye = int(np.count_nonzero(far_art & ~profile))
    if outside_eye:
        raise ValueError("The redrawn far eye exceeds the facial profile")
    clearance = distance_transform_edt(profile[345*samples:419*samples, 602*samples:663*samples])
    crop = far_art[345*samples:419*samples, 602*samples:663*samples]
    min_eye_clearance = float(clearance[crop].min()) / samples

    old_far = fields_at(FAR_EYE_SEEDS)
    y = np.arange(base.shape[0])[:, None] / samples
    result[old_far] = np.where(profile, 15, np.where(y < 378, 6, 2))[old_far]

    layer = render(root, base.shape)
    drawn = layer[..., 3] >= 128
    pigments, inverse = np.unique(layer[drawn, :3], axis=0, return_inverse=True)
    nearest = cKDTree([row[2] for row in rows]).query(pigments)[1]
    result[drawn] = nearest[inverse] + 1
    return result, {
        "far_eye_outside_profile_pixels": outside_eye,
        "far_eye_artist_profile_clearance_px": round(min_eye_clearance, 2),
        "base_skin_paint": 15,
    }


def ink_strokes():
    source = ET.parse(ART).getroot().find(f".//{{{SVG}}}g[@id='ink-details']")
    return [(path.attrib["d"], path.attrib["stroke"]) for path in source]


def ink_clip_path(arcs, rings, segments, origin, unit, samples):
    """Keep open drawing marks clear of shared boundaries and printed numbers."""
    from shapely import get_parts
    from shapely.geometry import box

    span = box(*(value * samples for value in EDIT_BOUNDS))
    glyphs = {}
    for segment in segments:
        x0, y0, x1, y1 = segment["bounds_mm"]
        glyphs[segment["segment_id"]] = box(
            (x0 - origin[0] - .23) / unit, (y0 - origin[1] - .23) / unit,
            (x1 - origin[0] + .23) / unit, (y1 - origin[1] + .23) / unit,
        )

    def ring_path(ring):
        coords = [f"{origin[0] + x*unit:.3f},{origin[1] + y*unit:.3f}"
                  for x, y in ring.coords]
        return "M" + "L".join(coords[:-1]) + "Z"

    paths = []
    for region, interior in ink_interiors(arcs, rings, .31 / unit):
        if interior.is_empty or not interior.intersects(span):
            continue
        safe = interior.intersection(span).difference(glyphs[region])
        for part in get_parts(safe):
            if part.geom_type == "Polygon" and not part.is_empty:
                paths.append(ring_path(part.exterior))
                paths.extend(ring_path(hole) for hole in part.interiors)
    return "".join(paths)
