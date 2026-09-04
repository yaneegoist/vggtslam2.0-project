import argparse
import csv
import math
import re
from pathlib import Path


METRIC_FILES = {
    "ate_se3_rmse_m": "ape_se3.txt",
    "ate_sim3_rmse_m": "ape_sim3.txt",
    "rpe_translation_rmse_m": "rpe_translation.txt",
    "rpe_rotation_rmse_deg": "rpe_rotation_deg.txt",
}


def sequence_group(sequence):
    if sequence.startswith("MH_"):
        return "Machine Hall"
    if sequence.startswith("V1_"):
        return "Vicon Room 1"
    if sequence.startswith("V2_"):
        return "Vicon Room 2"
    return "Other"


def parse_rmse(path):
    if not path.is_file():
        return None
    match = re.search(
        r"^\s*rmse\s+([-+0-9.eE]+)\s*$",
        path.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def parse_log(path):
    fields = {"keyframes": None, "total_time_s": None, "fps": None}
    if not path.is_file():
        return fields
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "keyframes": r"(?m)^(\d+) frames processed\s*$",
        "total_time_s": r"(?m)^Total time:\s*([-+0-9.eE]+)\s*$",
        "fps": r"(?m)^Average FPS:\s*([-+0-9.eE]+)\s*$",
    }
    for field, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            fields[field] = int(matches[-1]) if field == "keyframes" else float(matches[-1])
    return fields


def load_run(results_root, sequence, method):
    result_family = "da3_baseline_euroc" if method == "baseline" else "da3_gt_euroc"
    result_dir = results_root / result_family / sequence
    row = {
        "sequence": sequence,
        "group": sequence_group(sequence),
        "method": method,
        "status": "complete",
        "trajectory_poses": None,
        "keyframes": None,
        "ate_se3_rmse_m": None,
        "ate_sim3_rmse_m": None,
        "rpe_translation_rmse_m": None,
        "rpe_rotation_rmse_deg": None,
        "total_time_s": None,
        "fps": None,
        "dense_map": (result_dir / "trajectory_points.pcd").is_file(),
    }

    trajectory = result_dir / "trajectory.txt"
    if trajectory.is_file():
        row["trajectory_poses"] = sum(
            1
            for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    else:
        row["status"] = "missing"

    for field, filename in METRIC_FILES.items():
        row[field] = parse_rmse(result_dir / filename)
    if any(row[field] is None for field in (
        "ate_se3_rmse_m",
        "rpe_translation_rmse_m",
        "rpe_rotation_rmse_deg",
    )):
        row["status"] = "incomplete"

    row.update(parse_log(result_dir / "run.log"))
    return row


def fmt(value, digits=4):
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def mean(rows, field):
    values = [row[field] for row in rows if row[field] is not None and math.isfinite(row[field])]
    return sum(values) / len(values) if values else None


def write_csv(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(rows, sequences):
    lookup = {(row["sequence"], row["method"]): row for row in rows}
    return [(sequence, lookup[(sequence, "baseline")], lookup[(sequence, "gt")]) for sequence in sequences]


def write_markdown(path, rows, sequences):
    comparisons = comparison_rows(rows, sequences)
    lines = [
        "# EuRoC DA3 factor-graph ablation",
        "",
        "Comparison of the unchanged DA3/SL(4) backend against the same pipeline with exact",
        "EuRoC camera GT added as parallel SE(3) factors. GT is used by the baseline only for",
        "post-run evaluation. SE(3) ATE is aligned by one rigid transform without scale correction.",
        "The GT-factor run is an oracle ablation, not a deployable SLAM result.",
        "",
        "## Protocol",
        "",
        "- Input: monocular EuRoC cam0 images; the same prepared image list is used for both runs.",
        "- Frontend: Depth Anything 3 with the repository's fixed preprocessing and keyframe policy.",
        "- Baseline: original SL(4) graph, with loop closure disabled in DA3 mode.",
        "- Oracle: the same graph plus exact EuRoC cam0 relative poses as parallel SE(3) factors.",
        "- Metric-factor noise: 0.01 m translation and 0.1 degree rotation.",
        "- The batch runner rejects a pair if baseline and oracle trajectory timestamps differ.",
        "",
        "## Per-sequence results",
        "",
        "| Sequence | Poses (base/GT) | ATE SE3 baseline [m] | ATE SE3 +GT [m] | ATE Sim3 baseline [m] | RPE trans baseline [m] | RPE trans +GT [m] | RPE rot baseline [deg] | RPE rot +GT [deg] | Time base/GT [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence, baseline, gt in comparisons:
        lines.append(
            "| {sequence} | {bp}/{gp} | {ba} | {ga} | {bs} | {bt} | {gt_} | {br} | {gr} | {btime}/{gtime} |".format(
                sequence=sequence,
                bp=baseline["trajectory_poses"] or "—",
                gp=gt["trajectory_poses"] or "—",
                ba=fmt(baseline["ate_se3_rmse_m"]),
                ga=fmt(gt["ate_se3_rmse_m"]),
                bs=fmt(baseline["ate_sim3_rmse_m"]),
                bt=fmt(baseline["rpe_translation_rmse_m"]),
                gt_=fmt(gt["rpe_translation_rmse_m"]),
                br=fmt(baseline["rpe_rotation_rmse_deg"]),
                gr=fmt(gt["rpe_rotation_rmse_deg"]),
                btime=fmt(baseline["total_time_s"], 1),
                gtime=fmt(gt["total_time_s"], 1),
            )
        )

    lines.extend([
        "",
        "## Macro averages",
        "",
        "| Group | Method | ATE SE3 [m] | RPE trans [m] | RPE rot [deg] | Time [s] |",
        "|---|---|---:|---:|---:|---:|",
    ])
    groups = []
    for sequence in sequences:
        group = sequence_group(sequence)
        if group not in groups:
            groups.append(group)
    groups.append("Overall")

    for group in groups:
        for method in ("baseline", "gt"):
            selected = [
                row for row in rows
                if row["method"] == method and (group == "Overall" or row["group"] == group)
            ]
            label = "DA3 baseline" if method == "baseline" else "DA3 + exact GT factors"
            lines.append(
                f"| {group} | {label} | {fmt(mean(selected, 'ate_se3_rmse_m'))} | "
                f"{fmt(mean(selected, 'rpe_translation_rmse_m'))} | "
                f"{fmt(mean(selected, 'rpe_rotation_rmse_deg'))} | "
                f"{fmt(mean(selected, 'total_time_s'), 1)} |"
            )

    incomplete = [row for row in rows if row["status"] != "complete"]
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Primary trajectory metric: ATE SE(3) RMSE in metres (no scale correction).",
        "- ATE Sim(3) is a baseline diagnostic showing error after scale correction.",
        "- RPE reports local translation and rotation consistency.",
        "- Dense-map geometry is not validated by ATE/RPE and requires a reference scan.",
        "- Exact-GT factors test graph wiring, coordinate conventions, and map propagation; DAVIO",
        "  poses must later replace GT for the deployable experiment.",
        "- The original V1_01_easy orientation GT has a reported accuracy issue; identify whether",
        "  the official or corrected trajectory was used before interpreting its rotation metric:",
        "  https://docs.openvins.com/gs-datasets.html#groundtruth-on-v1_01_easy",
    ])
    if incomplete:
        lines.extend(["", "## Incomplete runs", ""])
        lines.extend(f"- {row['sequence']} / {row['method']}: {row['status']}" for row in incomplete)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate EuRoC DA3 ablation results")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    args = parser.parse_args()

    rows = [
        load_run(args.results_root, sequence, method)
        for sequence in args.sequences
        for method in ("baseline", "gt")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "summary.md", rows, args.sequences)
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
