import argparse
import csv
import json
import re
from collections import Counter
from math import ceil
from pathlib import Path


TBOX_TRACK_MARKER = '"track": "T_BOX"'
PROPERTY_REVISION_RE = re.compile(r'"property_revision_id"\s*:\s*(\d+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Stage 4 benchmark JSONL file and count how many T-BOX cases "
            "map to the same property revision."
        )
    )
    parser.add_argument(
        "--input",
        default="data/04_classified_benchmark.jsonl",
        help="Path to the classified benchmark JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tbox_update_analysis",
        help="Directory for summary outputs, CSV, and SVG chart.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of highest-frequency T-BOX updates to include in the chart and summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print a progress update every N lines.",
    )
    parser.add_argument(
        "--max-cap",
        type=int,
        default=1000,
        help="Largest per-update repair cap N to include in the survival curve.",
    )
    return parser.parse_args()


def _extract_metadata(record):
    repair_target = record.get("repair_target") or {}
    constraint_delta = repair_target.get("constraint_delta") or {}
    labels_en = record.get("labels_en") or {}
    property_label = (labels_en.get("property") or {}).get("label")
    property_description = (labels_en.get("property") or {}).get("description")
    violation_context = record.get("violation_context") or {}
    readable = constraint_delta.get("constraints_readable_en") or {}
    return {
        "property": record.get("property"),
        "property_label": property_label,
        "property_description": property_description,
        "author": repair_target.get("author"),
        "repair_id_example": record.get("id"),
        "report_violation_type": violation_context.get("report_violation_type_normalized")
        or violation_context.get("report_violation_type"),
        "changed_constraint_types": json.dumps(
            constraint_delta.get("changed_constraint_types") or [], ensure_ascii=False
        ),
        "hash_before": constraint_delta.get("hash_before"),
        "hash_after": constraint_delta.get("hash_after"),
        "before_constraint_count": len(readable.get("before") or []),
        "after_constraint_count": len(readable.get("after") or []),
    }


def analyze_tbox_updates(input_path, progress_every=100000):
    counts = Counter()
    metadata_by_revision = {}
    total_lines = 0
    total_tbox_cases = 0

    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            if progress_every > 0 and total_lines % progress_every == 0:
                print(
                    f"[progress] lines={total_lines:,} "
                    f"tbox_cases={total_tbox_cases:,} "
                    f"distinct_updates={len(counts):,}"
                )

            if TBOX_TRACK_MARKER not in line:
                continue

            match = PROPERTY_REVISION_RE.search(line)
            if not match:
                continue

            revision_id = int(match.group(1))
            total_tbox_cases += 1
            counts[revision_id] += 1

            if revision_id in metadata_by_revision:
                continue

            record = json.loads(line)
            metadata_by_revision[revision_id] = _extract_metadata(record)

    rows = []
    for revision_id, count in counts.most_common():
        row = {"property_revision_id": revision_id, "case_count": count}
        row.update(metadata_by_revision.get(revision_id, {}))
        rows.append(row)

    return {
        "total_lines": total_lines,
        "total_tbox_cases": total_tbox_cases,
        "distinct_tbox_updates": len(counts),
        "rows": rows,
    }


def write_csv(rows, out_path):
    fieldnames = [
        "property_revision_id",
        "case_count",
        "property",
        "property_label",
        "property_description",
        "author",
        "repair_id_example",
        "report_violation_type",
        "changed_constraint_types",
        "hash_before",
        "hash_after",
        "before_constraint_count",
        "after_constraint_count",
    ]
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(summary, out_path, top_n):
    rows = summary["rows"]
    payload = {
        "total_lines": summary["total_lines"],
        "total_tbox_cases": summary["total_tbox_cases"],
        "distinct_tbox_updates": summary["distinct_tbox_updates"],
        "top_updates": rows[:top_n],
        "survival_curve": summary["survival_curve"],
        "highlighted_caps": summary["highlighted_caps"],
    }
    with Path(out_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _row_label(row):
    property_id = row.get("property") or "?"
    property_label = row.get("property_label") or "unlabeled property"
    revision_id = row.get("property_revision_id")
    return f"{property_id} {property_label} (rev {revision_id})"


def build_survival_curve(rows, max_cap):
    counts = [row["case_count"] for row in rows]
    highlight_caps = [0, 2, 5, 10, 20, 50, 100, 500, 1000]
    highlight_caps = sorted({cap for cap in highlight_caps if 0 <= cap <= max_cap})
    if max_cap not in highlight_caps:
        highlight_caps.append(max_cap)
        highlight_caps.sort()

    curve = []
    highlighted = {}
    for cap in range(max_cap, -1, -1):
        surviving = sum(min(count, cap) for count in counts)
        point = {"cap": cap, "surviving_tbox_repairs": surviving}
        curve.append(point)
        if cap in highlight_caps:
            highlighted[cap] = surviving

    return curve, highlighted


def write_survival_curve_csv(curve, out_path):
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cap", "surviving_tbox_repairs"])
        writer.writeheader()
        writer.writerows(curve)


def _format_int(value):
    return f"{int(value):,}"


def _nice_upper_bound(value):
    if value <= 0:
        return 1
    magnitude = 10 ** (len(str(int(value))) - 1)
    return int(ceil(value / magnitude) * magnitude)


def write_svg_chart(rows, out_path, top_n):
    top_rows = rows[:top_n]
    if not top_rows:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120">'
            '<rect width="100%" height="100%" fill="#fffaf0"/>'
            '<text x="24" y="64" font-family="Verdana, sans-serif" font-size="24" fill="#333">'
            "No T-BOX updates found."
            "</text></svg>"
        )
        Path(out_path).write_text(svg, encoding="utf-8")
        return

    bar_area_width = 520
    left_margin = 360
    right_margin = 60
    top_margin = 70
    row_height = 28
    chart_height = top_margin + row_height * len(top_rows) + 40
    chart_width = left_margin + bar_area_width + right_margin
    max_count = max(row["case_count"] for row in top_rows) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="{chart_height}">',
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        '<text x="24" y="32" font-family="Verdana, sans-serif" font-size="24" fill="#222">'
        "T-BOX update frequency"
        "</text>",
        '<text x="24" y="54" font-family="Verdana, sans-serif" font-size="12" fill="#555">'
        f"Top {len(top_rows)} property revisions ranked by number of Stage 4 T-BOX cases"
        "</text>",
    ]

    for index, row in enumerate(top_rows):
        y = top_margin + index * row_height
        bar_width = int((row["case_count"] / max_count) * bar_area_width)
        label = _svg_escape(_row_label(row))
        count_text = _svg_escape(row["case_count"])
        parts.append(
            f'<text x="{left_margin - 12}" y="{y + 16}" text-anchor="end" '
            'font-family="Verdana, sans-serif" font-size="12" fill="#333">'
            f"{label}</text>"
        )
        parts.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_width}" height="18" '
            'rx="4" fill="#c26d2d"/>'
        )
        parts.append(
            f'<text x="{left_margin + bar_width + 8}" y="{y + 14}" '
            'font-family="Verdana, sans-serif" font-size="12" fill="#333">'
            f"{count_text}</text>"
        )

    parts.append("</svg>")
    Path(out_path).write_text("".join(parts), encoding="utf-8")


def write_survival_svg(curve, highlighted_caps, out_path, max_cap):
    width = 1100
    height = 640
    left_margin = 90
    right_margin = 340
    top_margin = 70
    bottom_margin = 90
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    max_y = max((point["surviving_tbox_repairs"] for point in curve), default=0)
    y_axis_max = _nice_upper_bound(max_y)

    def x_for(cap):
        if max_cap <= 0:
            return left_margin
        return left_margin + ((max_cap - cap) / max_cap) * plot_width

    def y_for(value):
        if y_axis_max <= 0:
            return top_margin + plot_height
        return top_margin + plot_height - (value / y_axis_max) * plot_height

    y_ticks = 5
    x_ticks = [max_cap, 800, 600, 400, 200, 0]
    x_ticks = [tick for tick in x_ticks if 0 <= tick <= max_cap]
    if max_cap not in x_ticks:
        x_ticks.insert(0, max_cap)
    x_ticks = sorted(set(x_ticks), reverse=True)

    path_parts = []
    for index, point in enumerate(curve):
        x = x_for(point["cap"])
        y = y_for(point["surviving_tbox_repairs"])
        path_parts.append(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        '<text x="24" y="32" font-family="Verdana, sans-serif" font-size="24" fill="#222">'
        "Surviving T-BOX repairs by per-update cap"
        "</text>",
        '<text x="24" y="54" font-family="Verdana, sans-serif" font-size="12" fill="#555">'
        f"Each point is sum(min(case_count_for_revision, N)) for N from {max_cap} down to 0"
        "</text>",
    ]

    parts.append(
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" '
        f'x2="{left_margin + plot_width}" y2="{top_margin + plot_height}" '
        'stroke="#555" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left_margin}" y1="{top_margin}" '
        f'x2="{left_margin}" y2="{top_margin + plot_height}" '
        'stroke="#555" stroke-width="1.5"/>'
    )

    for index in range(y_ticks + 1):
        value = int(round((y_axis_max / y_ticks) * index))
        y = y_for(value)
        parts.append(
            f'<line x1="{left_margin}" y1="{y:.2f}" '
            f'x2="{left_margin + plot_width}" y2="{y:.2f}" '
            'stroke="#e8dccd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left_margin - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Verdana, sans-serif" font-size="11" fill="#555">'
            f"{_svg_escape(_format_int(value))}</text>"
        )

    for tick in x_ticks:
        x = x_for(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top_margin + plot_height}" '
            f'x2="{x:.2f}" y2="{top_margin + plot_height + 6}" '
            'stroke="#555" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top_margin + plot_height + 24}" text-anchor="middle" '
            'font-family="Verdana, sans-serif" font-size="11" fill="#555">'
            f"{_svg_escape(_format_int(tick))}</text>"
        )

    parts.append(
        f'<path d="{" ".join(path_parts)}" fill="none" stroke="#c26d2d" stroke-width="3"/>'
    )

    label_box_x = left_margin + plot_width + 24
    label_box_y = top_margin + 8
    label_box_width = right_margin - 48
    label_box_height = 22 + len(highlighted_caps) * 24
    parts.append(
        f'<rect x="{label_box_x}" y="{label_box_y}" width="{label_box_width}" '
        f'height="{label_box_height}" rx="10" fill="#fff4e8" stroke="#e3c29e"/>'
    )
    parts.append(
        f'<text x="{label_box_x + 16}" y="{label_box_y + 22}" '
        'font-family="Verdana, sans-serif" font-size="13" fill="#7a4618">'
        "Explicit cap values"
        "</text>"
    )

    for index, cap in enumerate(sorted(highlighted_caps)):
        surviving = highlighted_caps[cap]
        x = x_for(cap)
        y = y_for(surviving)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#7a4618"/>')

        if cap in {0, max_cap, 100, 500}:
            anchor = "start" if cap == max_cap else "end" if cap == 0 else "middle"
            dx = 10 if anchor == "start" else -10 if anchor == "end" else 0
            parts.append(
                f'<text x="{x + dx:.2f}" y="{y - 10:.2f}" text-anchor="{anchor}" '
                'font-family="Verdana, sans-serif" font-size="11" fill="#7a4618">'
                f"N={cap}</text>"
            )

        label_y = label_box_y + 46 + index * 24
        label_text = f"N={cap}: {_format_int(surviving)}"
        parts.append(
            f'<text x="{label_box_x + 16}" y="{label_y}" '
            'font-family="Verdana, sans-serif" font-size="12" fill="#333">'
            f"{_svg_escape(label_text)}</text>"
        )

    parts.append(
        f'<text x="{left_margin + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" '
        'font-family="Verdana, sans-serif" font-size="13" fill="#333">'
        "N repairs kept per property revision"
        "</text>"
    )
    parts.append(
        f'<text x="22" y="{top_margin + plot_height / 2:.2f}" '
        'font-family="Verdana, sans-serif" font-size="13" fill="#333" '
        f'transform="rotate(-90 22 {top_margin + plot_height / 2:.2f})">'
        "Surviving T-BOX repairs"
        "</text>"
    )
    parts.append("</svg>")
    Path(out_path).write_text("".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] input={input_path}")
    summary = analyze_tbox_updates(input_path, progress_every=args.progress_every)
    curve, highlighted_caps = build_survival_curve(summary["rows"], args.max_cap)
    summary["survival_curve"] = curve
    summary["highlighted_caps"] = highlighted_caps

    csv_path = output_dir / "tbox_update_frequency.csv"
    survival_csv_path = output_dir / "tbox_survival_by_cap.csv"
    summary_path = output_dir / "summary.json"
    chart_path = output_dir / "tbox_update_frequency_top.svg"
    survival_chart_path = output_dir / "tbox_survival_by_cap.svg"

    write_csv(summary["rows"], csv_path)
    write_survival_curve_csv(curve, survival_csv_path)
    write_summary(summary, summary_path, args.top_n)
    write_svg_chart(summary["rows"], chart_path, args.top_n)
    write_survival_svg(curve, highlighted_caps, survival_chart_path, args.max_cap)

    print(f"[done] total_lines={summary['total_lines']:,}")
    print(f"[done] total_tbox_cases={summary['total_tbox_cases']:,}")
    print(f"[done] distinct_tbox_updates={summary['distinct_tbox_updates']:,}")
    print(f"[done] csv={csv_path}")
    print(f"[done] survival_csv={survival_csv_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] chart={chart_path}")
    print(f"[done] survival_chart={survival_chart_path}")


if __name__ == "__main__":
    main()
