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


def extend_reverse_repeat(sequence: str, left_start: int, right_seed_start: int, seed_length: int) -> Region:
    n = len(sequence)
    left_extra = 0
    while (
        left_start - 1 - left_extra >= 0
        and right_seed_start + seed_length + left_extra < n
        and sequence[left_start - 1 - left_extra]
        == sequence[right_seed_start + seed_length + left_extra].translate(DNA_COMPLEMENT)
    ):
        left_extra += 1

    right_extra = 0
    while (
        left_start + seed_length + right_extra < n
        and right_seed_start - 1 - right_extra >= 0
        and sequence[left_start + seed_length + right_extra]
        == sequence[right_seed_start - 1 - right_extra].translate(DNA_COMPLEMENT)
    ):
        right_extra += 1

    start = left_start - left_extra
    end = left_start + seed_length + right_extra
    return Region("repeat", start, end)


def infer_regions(
    sequence: str,
    seed_length: int = 80,
    scan_step: int = 50,
    min_ir_length: int = 10000,
    min_ir_spacing: int = 10000,
) -> RegionMap:
    positions: dict[str, list[int]] = {}
    n = len(sequence)
    for start in range(0, n - seed_length + 1):
        seed = sequence[start : start + seed_length]
        if "N" in seed:
            continue
        positions.setdefault(seed, []).append(start)

    best_pair: tuple[int, int, int, int] | None = None
    best_length = 0
    for left_start in range(0, n - seed_length + 1, scan_step):
        seed = sequence[left_start : left_start + seed_length]
        if "N" in seed:
            continue
        rc_seed = reverse_complement(seed)
        for right_start in positions.get(rc_seed, []):
            if abs(left_start - right_start) < min_ir_spacing:
                continue
            left = extend_reverse_repeat(sequence, left_start, right_start, seed_length)
            right = extend_reverse_repeat(sequence, right_start, left_start, seed_length)
            if left.length < min_ir_length or right.length < min_ir_length:
                continue
            first, second = sorted([left, right], key=lambda region: region.start)
            if first.length > best_length:
                best_pair = (first.start, first.end, second.start, second.end)
                best_length = first.length

    if best_pair is None:
        return RegionMap([Region("unknown", 0, n)], "unknown")

    ir_a_start, ir_a_end, ir_b_start, ir_b_end = best_pair
    gap_between = Region("single_copy", ir_a_end, ir_b_start)
    gap_wrap = Region("single_copy", ir_b_end, n + ir_a_start)
    lsc_gap, ssc_gap = sorted([gap_between, gap_wrap], key=lambda region: region.length, reverse=True)

    regions = [
        Region("IRA", ir_a_start, ir_a_end),
        Region("IRB", ir_b_start, ir_b_end),
        Region("LSC", lsc_gap.start, lsc_gap.end),
        Region("SSC", ssc_gap.start, ssc_gap.end),
    ]
    return RegionMap(regions, "inferred")


def coordinate_in_region(position: int, region: Region, genome_length: int) -> bool:
    normalized = position % genome_length
    start = region.start % genome_length
    end = (region.end - 1) % genome_length
    if region.end <= genome_length:
        return region.start <= position < region.end
    return normalized >= start or normalized <= end


def label_interval(start: int, end: int, region_map: RegionMap, genome_length: int) -> str:
    if region_map.status != "inferred":
        return "unknown"

    labels = set()
    for pos in (start, end - 1):
        for region in region_map.regions:
            if coordinate_in_region(pos, region, genome_length):
                labels.add(region.name)
                break

    return labels.pop() if len(labels) == 1 else "boundary"
