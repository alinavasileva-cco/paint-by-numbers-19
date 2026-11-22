"""Trace shared boundaries and reuse each curve in both vector plates."""

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def components(paints, color_count):
    """Keep point contacts separate: paint cannot cross a diagonal corner."""
    ids = np.zeros(paints.shape, dtype=np.int32)
    colors = [0]
    boxes = [(0, 0, 0, 0, 0)]
    for paint in range(1, color_count + 1):
        count, local_ids, stats, _ = cv2.connectedComponentsWithStats(
            np.uint8(paints == paint), connectivity=4
        )
        occupied = local_ids > 0
        ids[occupied] = local_ids[occupied] + len(colors) - 1
        colors.extend([paint] * (count - 1))
        boxes.extend(map(tuple, stats[1:].tolist()))
    return ids, np.asarray(colors, dtype=np.uint8), boxes


def contacts(ids):
    """Measure the length of the border that each pair of fields shares."""
    sides = []
    for left, right in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])):
        changed = left != right
        a, b = left[changed], right[changed]
        sides.append(np.column_stack((np.minimum(a, b), np.maximum(a, b))))
    pairs, lengths = np.unique(np.concatenate(sides), axis=0, return_counts=True)
    neighbors = defaultdict(list)
    for (a, b), length in zip(pairs.tolist(), lengths.tolist()):
        neighbors[a].append((b, length))
        neighbors[b].append((a, length))
    return neighbors


Point = tuple[float, float]


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point
    handles: tuple[Point, Point] | None = None


@dataclass(frozen=True)
class Arc:
    start: int
    end: int
    right: int
    left: int
    start_direction: int
    end_direction: int
    segments: tuple[Segment, ...]


def soften(points, closed, radius):
    """Round turns with tangential cubic curves and preserve junctions."""
    steps = []

    def line(start, end):
        if np.linalg.norm(end - start) > 0.01:
            steps.append(Segment(tuple(start), tuple(end)))

    def turn(index):
        before, middle, after = points[index - 1], points[index], points[(index + 1) % len(points)]
        incoming, outgoing = before - middle, after - middle
        in_length, out_length = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        distance = min(radius, in_length * 0.46, out_length * 0.46)
        entry = middle + incoming * (distance / in_length)
        exit_point = middle + outgoing * (distance / out_length)
        handles = (tuple(entry + (middle - entry) * (2 / 3)),
                   tuple(exit_point + (middle - exit_point) * (2 / 3)))
        return entry, exit_point, handles

    indices = range(len(points)) if closed else range(1, len(points) - 1)
    start = turn(0)[0] if closed else points[0]
    last = start
    for index in indices:
        entry, exit_point, handles = turn(index)
        line(last, entry)
        steps.append(Segment(tuple(entry), tuple(exit_point), handles))
        last = exit_point
    line(last, start if closed else points[-1])
    return tuple(steps)


def vectorize(ids, samples_per_pixel):
    """Find chains of grid edges between pinned multi-region junctions."""
    height, width = ids.shape
    stride = width + 1
    padded = np.pad(ids, 1)
    upper, lower = padded[:-1, 1:-1], padded[1:, 1:-1]
    left, right = padded[1:-1, :-1], padded[1:-1, 1:]
    edges = {}
    adjacent = defaultdict(list)
    for side_a, side_b, step in ((lower, upper, 1), (left, right, stride)):
        rows, columns = np.where(side_a != side_b)
        for y, x, a, b in zip(rows.tolist(), columns.tolist(),
                              side_a[rows, columns].tolist(), side_b[rows, columns].tolist()):
            first = y * stride + x
            second = first + step
            edges[first, second] = (a, b)
            adjacent[first].append(second)
            adjacent[second].append(first)

    original_edges = len(edges)

    def pop_edge(first, second):
        if first < second:
            return edges.pop((first, second))
        return edges.pop((second, first))[::-1]

    def direction(first, second):
        return {1: 0, stride: 1, -1: 2, -stride: 3}[second - first]

    arcs = []
    for start in sorted(adjacent, key=lambda vertex: len(adjacent[vertex]) == 2):
        for next_node in adjacent[start]:
            if (min(start, next_node), max(start, next_node)) not in edges:
                continue
            sides = pop_edge(start, next_node)
            nodes = [start, next_node]
            while nodes[-1] != start and len(adjacent[nodes[-1]]) == 2:
                previous, current = nodes[-2:]
                a, b = adjacent[current]
                following = a if b == previous else b
                if pop_edge(current, following) != sides:
                    raise ValueError("A shared boundary changed regions between junctions")
                nodes.append(following)

            points = np.asarray([(node % stride, node // stride) for node in nodes], np.float32)
            closed = nodes[-1] == start and len(adjacent[start]) == 2
            center_x, center_y = points.mean(axis=0) / samples_per_pixel
            face = 445 < center_x < 693 and 325 < center_y < 550
            if closed:
                points = points[:-1]
            if len(points) > 7:
                sigma = (1.20 if face else 1.75) * samples_per_pixel
                filtered = gaussian_filter1d(points, sigma, axis=0, mode="wrap" if closed else "nearest")
                if not closed:
                    distances = np.minimum(np.arange(len(points)), np.arange(len(points) - 1, -1, -1))
                    blend = np.minimum(distances[:, None] / (2 * sigma), 1)
                    filtered = np.float32(points + blend * (filtered - points))
                points = filtered
            tolerance = (0.67 if face else 0.92) * samples_per_pixel
            simplified = cv2.approxPolyDP(points[:, None], tolerance, closed).reshape(-1, 2)
            if not closed:
                simplified[0], simplified[-1] = points[0], points[-1]
            if len(simplified) < (3 if closed else 2):
                simplified = points
            segments = soften(simplified, closed, (2.00 if face else 3.00) * samples_per_pixel)
            arcs.append(Arc(start, nodes[-1], *sides,
                            direction(*nodes[:2]), direction(*nodes[-2:]), segments))

    if edges:
        raise ValueError("Some grid edges were left untraced")
    return arcs, original_edges


def region_rings(arcs, region_count):
    """Stitch the directed sides of the arcs into closed rings per region."""
    outgoing = defaultdict(list)
    directed = defaultdict(list)
    for index, arc in enumerate(arcs):
        for region, node, reverse in ((arc.right, arc.start, False), (arc.left, arc.end, True)):
            if region == 0:
                continue
            edge = index, reverse
            outgoing[region, node].append(edge)
            directed[region].append(edge)

    def incoming_direction(edge):
        index, reverse = edge
        arc = arcs[index]
        return (arc.end_direction if not reverse else arc.start_direction + 2) % 4

    def outgoing_direction(edge):
        index, reverse = edge
        arc = arcs[index]
        return (arc.start_direction if not reverse else arc.end_direction + 2) % 4

    rings = {}
    for region in range(1, region_count + 1):
        seen = set()
        rings[region] = []
        for edge in directed[region]:
            if edge in seen:
                continue
            ring = []
            current = edge
            while current not in seen:
                seen.add(current)
                ring.append(current)
                index, reverse = current
                arc = arcs[index]
                vertex = arc.start if reverse else arc.end
                incoming = incoming_direction(current)
                current = min(outgoing[region, vertex], key=lambda following:
                              (1, 0, 3, 2).index((outgoing_direction(following) - incoming) % 4))
            if current != edge:
                raise ValueError(f"Open or duplicate vector ring in region {region}")
            rings[region].append(ring)
    return rings


def svg_paths(arcs, rings_by_region, origin, mm_per_sample):
    """Serialize both plates from identical Bézier segments."""
    def coordinate(point):
        x, y = point
        return f"{origin[0] + x * mm_per_sample:.3f},{origin[1] + y * mm_per_sample:.3f}"

    def commands(segments, reverse):
        steps = reversed(segments) if reverse else segments
        result = []
        for step in steps:
            destination = coordinate(step.start if reverse else step.end)
            if step.handles is None:
                result.append("L" + destination)
            else:
                handles = step.handles[::-1] if reverse else step.handles
                result.append("C" + " ".join(map(coordinate, handles)) + " " + destination)
        return "".join(result)

    paths = []
    for arc in arcs:
        forward_start = coordinate(arc.segments[0].start)
        backward_start = coordinate(arc.segments[-1].end)
        paths.append(((forward_start, commands(arc.segments, False)),
                      (backward_start, commands(arc.segments, True))))

    outline = "".join(
        "M" + start + path for (start, path), _ in paths
    )
    fills = {}
    for region, rings in rings_by_region.items():
        parts = []
        for ring in rings:
            pieces = [paths[index][reverse] for index, reverse in ring]
            parts.append("M" + pieces[0][0] + "".join(path for _, path in pieces) + "Z")
        fills[region] = "".join(parts)
    return outline, fills


def ink_interiors(arcs, rings, radius):
    """Calculate each field's interior after the printed stroke takes its width."""
    import shapely
    from shapely.geometry import Polygon

    def sample(arc):
        points = []
        for segment in arc.segments:
            points.append(segment.start)
            if segment.handles:
                a, b, c, d = map(np.asarray, (segment.start, *segment.handles, segment.end))
                t = np.arange(1, 10)[:, None] / 10
                cubic = (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t**2*c + t**3*d
                points.extend(map(tuple, cubic))
        return points + [arc.segments[-1].end]

    curves = [sample(arc) for arc in arcs]
    for region, region_loops in rings.items():
        polygons = []
        for loop in region_loops:
            path = []
            for arc_index, reverse in loop:
                curve = curves[arc_index]
                path.extend(curve[-1:0:-1] if reverse else curve[:-1])
            polygon = Polygon(path)
            polygons.append(polygon if polygon.is_valid else shapely.make_valid(polygon))
        # The fill rule must match the SVG, including any hole in a field.
        solid = shapely.symmetric_difference_all(polygons)
        yield region, solid.buffer(-radius, quad_segs=8, join_style="round")


def open_narrow_fields(indexed, ids, numbered, arcs, rings, unit, outer_clearance):
    """Make each printed cell usable past its stroke, including narrow necks."""
    import shapely
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    from scipy.ndimage import distance_transform_edt

    edits = indexed.copy()
    locations = {region: (paint, x, y) for region, paint, x, y in numbered}
    defects = 0
    for region, interior in ink_interiors(arcs, rings, outer_clearance / unit):
        paint, x, y = locations[region]
        parts = [part for part in shapely.get_parts(interior) if not part.is_empty]
        if len(parts) == 1:
            continue
        core = next((part for part in parts if part.covers(Point(x, y))), None)
        if core is None:
            raise ValueError(f"Region {region} lost its number after tracing")
        for pocket in parts:
            if pocket is core:
                continue
            defects += 1
            if pocket.area * unit**2 >= 2:
                start, end = nearest_points(core, pocket)
                a, b = (tuple(map(round, node.coords[0])) for node in (start, end))
                cv2.line(edits, a, b, paint, 7)
                continue

            # A small split-off end can share the paint of its nearest adjoining area.
            trimmed = pocket.buffer(2.8, quad_segs=6)
            xmin, ymin, xmax, ymax = trimmed.bounds
            left, top = max(0, int(xmin) - 5), max(0, int(ymin) - 5)
            right = min(indexed.shape[1], int(np.ceil(xmax)) + 6)
            bottom = min(indexed.shape[0], int(np.ceil(ymax)) + 6)
            local = indexed[top:bottom, left:right]
            should_trim = np.zeros(local.shape, np.uint8)
            for polygon in shapely.get_parts(trimmed):
                xy = np.array(polygon.exterior.coords) - [left, top]
                cv2.fillPoly(should_trim, [np.rint(xy * 16).astype(np.int32)], 1, shift=4)
            remove = (should_trim > 0) & (ids[top:bottom, left:right] == region)
            distance_indices = distance_transform_edt(
                local == paint, return_distances=False, return_indices=True
            )
            adjoining = local[tuple(distance_indices)]
            edits[top:bottom, left:right][remove] = adjoining[remove]
    return edits, defects
