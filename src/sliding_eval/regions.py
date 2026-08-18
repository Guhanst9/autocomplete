from collections import defaultdict
from dataclasses import dataclass


DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass
class Region:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class RegionMap:
    regions: list[Region]
    status: str


def reverse_complement(sequence: str) -> str:
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def region_map_from_repeats(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
    genome_length: int,
    status: str,
    first_name: str = "IRB",
    second_name: str = "IRA",
) -> RegionMap:
    first = Region(first_name, first_start, first_end)
    second = Region(second_name, second_start, second_end)
    gap_between = Region("single_copy", first_end, second_start)
    gap_wrap = Region("single_copy", second_end, genome_length + first_start)
    lsc_gap, ssc_gap = sorted([gap_between, gap_wrap], key=lambda region: region.length, reverse=True)
    return RegionMap(
        [
            first,
            second,
            Region("LSC", lsc_gap.start, lsc_gap.end),
            Region("SSC", ssc_gap.start, ssc_gap.end),
        ],
        status,
    )


def infer_regions(
    sequence: str,
    seed_length: int = 31,
    scan_step: int = 10,
    min_ir_length: int = 10000,
    min_ir_spacing: int = 10000,
    max_diagonal_shift: int = 10,
) -> RegionMap:
    positions: dict[str, list[int]] = defaultdict(list)
    n = len(sequence)
    for start in range(n - seed_length + 1):
        seed = sequence[start : start + seed_length]
        if "N" not in seed:
            positions[seed].append(start)

    diagonals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for left_start in range(0, n - seed_length + 1, scan_step):
        seed = sequence[left_start : left_start + seed_length]
        if "N" in seed:
            continue
        for right_start in positions.get(reverse_complement(seed), []):
            if right_start - left_start < min_ir_spacing:
                continue
            diagonals[left_start + right_start].append((left_start, right_start))

    eligible: list[tuple[int, list[tuple[int, int]]]] = []
    minimum_anchor_span = max(200, min(1000, min_ir_length // 5))
    for diagonal, anchors in diagonals.items():
        if len(anchors) < 20:
            continue
        left_positions = [left for left, _ in anchors]
        span = max(left_positions) - min(left_positions) + seed_length
        if span >= minimum_anchor_span:
            eligible.append((diagonal, anchors))
    eligible.sort(key=lambda item: item[0])

    groups: list[list[tuple[int, list[tuple[int, int]]]]] = []
    for diagonal, anchors in eligible:
        if not groups or diagonal - groups[-1][-1][0] > max_diagonal_shift:
            groups.append([])
        groups[-1].append((diagonal, anchors))

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for group in groups:
        anchors = [anchor for _, items in group for anchor in items]
        left_positions = [left for left, _ in anchors]
        right_positions = [right for _, right in anchors]
        left_start = min(left_positions)
        left_end = max(left_positions) + seed_length
        right_start = min(right_positions)
        right_end = max(right_positions) + seed_length
        span = left_end - left_start
        if span >= min_ir_length:
            candidates.append((span, len(anchors), left_start, left_end, right_start, right_end))

    if not candidates:
        return RegionMap([Region("unknown", 0, n)], "unknown")

    _, _, first_start, first_end, second_start, second_end = max(candidates)
    return region_map_from_repeats(
        first_start,
        first_end,
        second_start,
        second_end,
        n,
        "sequence-inferred",
    )


def coordinate_in_region(position: int, region: Region, genome_length: int) -> bool:
    normalized = position % genome_length
    start = region.start % genome_length
    end = (region.end - 1) % genome_length
    if region.end <= genome_length:
        return start <= normalized <= end
    return normalized >= start or normalized <= end


def label_interval(start: int, end: int, region_map: RegionMap, genome_length: int) -> str:
    if region_map.status == "unknown":
        return "unknown"

    labels = set()
    for pos in (start, end - 1):
        for region in region_map.regions:
            if coordinate_in_region(pos, region, genome_length):
                labels.add(region.name)
                break

    return labels.pop() if len(labels) == 1 else "boundary"
