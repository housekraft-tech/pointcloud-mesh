"""Verify Feima SLAM2000 ``.fmlidar`` packet continuity and decode its layout.

The file is a Feima container around Livox MID-360 spherical packets.  Each
point packet contains 96 records with the official Livox data-type-3 layout::

    depth_mm:u32, theta_0p01deg:u16, phi_0p01deg:u16,
    reflectivity:u8, tag:u8

This probe is deliberately read-only.  It scans by marker rather than assuming
a fixed record stride because Feima inserts a 24-byte index record between some
packets.  The six-byte marker is the captured sensor IP (192.168.1.100) and
stream id found in this SLAM2000 project.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


MARKER = bytes.fromhex("c0 a8 01 64 08 02")
PACKET_MAGIC = bytes.fromhex("a2 a7 18")
HEADER_BYTES = 26
POINTS_PER_PACKET = 96
POINT_BYTES = 10
EXPECTED_PAYLOAD_BYTES = POINTS_PER_PACKET * POINT_BYTES
OVERLAP_BYTES = 64

POINT_DTYPE = np.dtype(
    [
        ("depth_mm", "<u4"),
        ("theta_0p01deg", "<u2"),
        ("phi_0p01deg", "<u2"),
        ("reflectivity", "u1"),
        ("tag", "u1"),
    ]
)


def _top(counter: Counter[int], n: int = 8) -> list[dict[str, int]]:
    return [{"value": int(value), "count": int(count)} for value, count in counter.most_common(n)]


def scan(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> dict:
    """Return container and continuity statistics without loading the file."""
    marker_re = re.compile(re.escape(MARKER))
    carry = b""
    offset = 0
    last_pos = -1
    last_ts = None
    last_seq = None
    first_ts = None
    first_offset = None
    packet_count = 0
    invalid_markers = 0
    sequence_discontinuities = 0
    nonpositive_timestamp_steps = 0
    strides: Counter[int] = Counter()
    timestamp_steps: Counter[int] = Counter()
    data_types: Counter[int] = Counter()
    payload_lengths: Counter[int] = Counter()
    sample_points: list[dict[str, int]] = []
    started = time.perf_counter()

    with path.open("rb") as source:
        while True:
            data = source.read(chunk_bytes)
            if not data:
                break
            buffer = carry + data
            base = offset - len(carry)

            for match in marker_re.finditer(buffer):
                local = match.start()
                absolute = base + local
                if absolute <= last_pos or local + HEADER_BYTES > len(buffer):
                    continue

                header = buffer[local : local + HEADER_BYTES]
                magic = header[11:14]
                sequence = int.from_bytes(header[14:16], "little")
                data_type = int(header[16])
                payload_length = int.from_bytes(header[17:19], "little")
                timestamp_ns = int.from_bytes(header[6:11], "little")
                if magic != PACKET_MAGIC or data_type != 3 or payload_length != EXPECTED_PAYLOAD_BYTES:
                    invalid_markers += 1
                    continue

                if first_ts is None:
                    first_ts = timestamp_ns
                    first_offset = absolute
                if last_pos >= 0:
                    strides[absolute - last_pos] += 1
                if last_ts is not None:
                    step = timestamp_ns - last_ts
                    timestamp_steps[step] += 1
                    nonpositive_timestamp_steps += int(step <= 0)
                    sequence_discontinuities += int(sequence != ((last_seq + 1) & 0xFFFF))

                if len(sample_points) < 8 and local + HEADER_BYTES + EXPECTED_PAYLOAD_BYTES <= len(buffer):
                    raw = buffer[
                        local + HEADER_BYTES : local + HEADER_BYTES + EXPECTED_PAYLOAD_BYTES
                    ]
                    points = np.frombuffer(raw, dtype=POINT_DTYPE)
                    for point in points[points["depth_mm"] > 0]:
                        sample_points.append(
                            {
                                name: int(point[name])
                                for name in POINT_DTYPE.names
                            }
                        )
                        if len(sample_points) == 8:
                            break

                data_types[data_type] += 1
                payload_lengths[payload_length] += 1
                packet_count += 1
                last_pos = absolute
                last_ts = timestamp_ns
                last_seq = sequence

            carry = buffer[-OVERLAP_BYTES:]
            offset += len(data)

    if packet_count == 0:
        raise ValueError(f"no valid MID-360 point packets found in {path}")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "packet_count": packet_count,
        "raw_return_slots": packet_count * POINTS_PER_PACKET,
        "first_packet_offset": first_offset,
        "last_packet_offset": last_pos,
        "first_lidar_timestamp_ns": first_ts,
        "last_lidar_timestamp_ns": last_ts,
        "duration_s": (last_ts - first_ts) / 1e9,
        "sequence_discontinuities": sequence_discontinuities,
        "nonpositive_timestamp_steps": nonpositive_timestamp_steps,
        "invalid_markers": invalid_markers,
        "data_types": {str(key): int(value) for key, value in sorted(data_types.items())},
        "payload_lengths": {
            str(key): int(value) for key, value in sorted(payload_lengths.items())
        },
        "stride_bytes_top": _top(strides),
        "timestamp_step_ns_top": _top(timestamp_steps),
        "sample_nonzero_points": sample_points,
        "scan_elapsed_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    report = scan(args.path, chunk_bytes=args.chunk_mib * 1024 * 1024)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"{report['path']}")
    print(
        f"  {report['packet_count']:,} packets; "
        f"{report['raw_return_slots']:,} raw return slots; "
        f"{report['duration_s']:.3f} s"
    )
    print(
        f"  sequence discontinuities={report['sequence_discontinuities']}; "
        f"nonpositive timestamp steps={report['nonpositive_timestamp_steps']}; "
        f"invalid markers={report['invalid_markers']}"
    )
    print(f"  data types={report['data_types']}; payload lengths={report['payload_lengths']}")
    print(f"  packet strides={report['stride_bytes_top']}")
    print(f"  timestamp steps={report['timestamp_step_ns_top']}")
    print(f"  scan elapsed={report['scan_elapsed_s']:.2f} s")


if __name__ == "__main__":
    main()
