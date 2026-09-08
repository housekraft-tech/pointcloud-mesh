"""Validate every named room against independent site controls.

The comparison is deliberately by stable ``room_id``.  Matching a measured
room to whichever control happens to be closest can manufacture a good score
from the wrong room, so missing semantic identity is a hard failure.

Both inputs use this small schema::

    {"rooms": [
      {"room_id": "living", "width_mm": 4947,
       "length_mm": 6540, "height_mm": 2697}
    ]}

Width and length are compared as the short and long clear spans, so a 90-degree
registration does not change the result.  Height is always compared directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = ("width_mm", "length_mm", "height_mm")


def _by_id(payload, label):
    rooms = payload.get("rooms", [])
    out = {}
    for room in rooms:
        room_id = room.get("room_id")
        if not room_id:
            raise ValueError(f"{label} contains a room without room_id")
        if room_id in out:
            raise ValueError(f"{label} contains duplicate room_id {room_id!r}")
        out[room_id] = room
    return out


def validate_rooms(measured_payload, control_payload, tolerance_mm=10.0):
    """Return a strict all-room accuracy report."""
    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive")
    measured = _by_id(measured_payload, "measurements")
    controls = _by_id(control_payload, "controls")
    rows = []

    for room_id, control in controls.items():
        actual = measured.get(room_id)
        if actual is None:
            rows.append({
                "room_id": room_id, "status": "missing", "dimensions": {},
                "reason": "no measured room with this stable room_id",
            })
            continue

        # Clear-plan spans are orientation-independent.  Assign the shorter
        # measured span to the shorter control span before comparing.
        c_spans = sorted([control.get("width_mm"), control.get("length_mm")],
                         key=lambda value: float("inf") if value is None else value)
        m_spans = sorted([actual.get("width_mm"), actual.get("length_mm")],
                         key=lambda value: float("inf") if value is None else value)
        comparisons = {
            "short_span_mm": (m_spans[0], c_spans[0]),
            "long_span_mm": (m_spans[1], c_spans[1]),
            "height_mm": (actual.get("height_mm"), control.get("height_mm")),
        }
        dimensions = {}
        complete = True
        passed = True
        for name, (value, expected) in comparisons.items():
            if value is None or expected is None:
                complete = False
                passed = False
                dimensions[name] = {
                    "measured_mm": value, "control_mm": expected,
                    "delta_mm": None, "pass": False,
                }
                continue
            delta = float(value) - float(expected)
            ok = abs(delta) <= tolerance_mm
            passed = passed and ok
            dimensions[name] = {
                "measured_mm": float(value), "control_mm": float(expected),
                "delta_mm": delta, "pass": ok,
            }
        rows.append({
            "room_id": room_id,
            "status": "pass" if passed and complete else "fail",
            "dimensions": dimensions,
        })

    extra = sorted(set(measured) - set(controls))
    n_pass = sum(row["status"] == "pass" for row in rows)
    status = "pass" if rows and n_pass == len(rows) and not extra else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "tolerance_mm": float(tolerance_mm),
        "criterion": (
            f"every named room width, length and clear height must be within "
            f"+/-{tolerance_mm:g} mm of independent site controls"
        ),
        "rooms_total": len(rows),
        "rooms_passed": n_pass,
        "rooms_failed_or_missing": len(rows) - n_pass,
        "uncontrolled_measured_room_ids": extra,
        "reason": (
            "all controlled rooms pass"
            if status == "pass"
            else "one or more rooms are outside tolerance, missing, incomplete, or uncontrolled"
        ),
        "rooms": rows,
    }


def markdown_report(report):
    lines = [
        f"# Every-room +/-{report['tolerance_mm']:g} mm validation",
        "",
        f"- Outcome: **{report['status'].upper()}**",
        f"- Rooms passed: {report['rooms_passed']}/{report['rooms_total']}",
        f"- Rule: {report['criterion']}",
        "",
        "| room | status | short span delta mm | long span delta mm | height delta mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for room in report["rooms"]:
        dims = room.get("dimensions", {})
        def delta(name):
            value = dims.get(name, {}).get("delta_mm")
            return "-" if value is None else f"{value:+.1f}"
        lines.append(
            f"| {room['room_id']} | {room['status']} | {delta('short_span_mm')} | "
            f"{delta('long_span_mm')} | {delta('height_mm')} |"
        )
    if report["uncontrolled_measured_room_ids"]:
        lines += [
            "",
            "Uncontrolled measured rooms (also block release): "
            + ", ".join(report["uncontrolled_measured_room_ids"]),
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--tolerance-mm", type=float, default=10.0)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--report-md", default=None)
    ap.add_argument("--require-pass", action="store_true")
    args = ap.parse_args()

    measured = json.loads(Path(args.measured).read_text(encoding="utf-8"))
    controls = json.loads(Path(args.controls).read_text(encoding="utf-8"))
    report = validate_rooms(measured, controls, args.tolerance_mm)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_md:
        md = Path(args.report_md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(markdown_report(report), encoding="utf-8")
    print(f"room accuracy: {report['status']} -- "
          f"{report['rooms_passed']}/{report['rooms_total']} pass -> {out}")
    if args.require_pass and report["status"] != "pass":
        raise SystemExit("every-room accuracy gate failed")


if __name__ == "__main__":
    main()
