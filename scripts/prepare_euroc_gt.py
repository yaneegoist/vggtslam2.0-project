import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_t_bs(sensor_yaml):
    """Load the 4x4 sensor-to-body transform from an ASL sensor file."""
    text = Path(sensor_yaml).read_text(encoding="utf-8")
    match = re.search(
        r"T_BS\s*:.*?data\s*:\s*\[(.*?)\]",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"T_BS.data was not found in {sensor_yaml}")

    values = ast.literal_eval("[" + match.group(1) + "]")
    if len(values) != 16:
        raise ValueError(
            f"Expected 16 T_BS values in {sensor_yaml}, got {len(values)}"
        )
    return np.asarray(values, dtype=float).reshape(4, 4)


def load_camera_index(data_csv):
    entries = []
    with Path(data_csv).open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("#"):
                continue
            entries.append((int(row[0]), row[1].strip()))

    if not entries:
        raise ValueError(f"No camera entries found in {data_csv}")
    return entries


def load_ground_truth(data_csv):
    timestamps_ns = []
    states = []
    with Path(data_csv).open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError(f"Malformed GT row in {data_csv}: {row}")
            timestamps_ns.append(int(row[0]))
            states.append([float(value) for value in row[1:8]])

    if len(states) < 2:
        raise ValueError(f"At least two GT poses are required in {data_csv}")

    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    states = np.asarray(states, dtype=float)
    positions = states[:, 0:3]
    quaternions_xyzw = states[:, [4, 5, 6, 3]]
    return timestamps_ns, positions, quaternions_xyzw


def prepare_sequence(sequence_root, output_dir, container_sequence_root=None):
    sequence_root = Path(sequence_root)
    output_dir = Path(output_dir)
    cam0_root = sequence_root / "cam0"
    gt_root = sequence_root / "state_groundtruth_estimate0"

    t_body_cam0 = load_t_bs(cam0_root / "sensor.yaml")
    camera_entries = load_camera_index(cam0_root / "data.csv")
    gt_ns, gt_positions, gt_quaternions = load_ground_truth(
        gt_root / "data.csv"
    )

    valid_camera_entries = [
        entry
        for entry in camera_entries
        if gt_ns[0] <= entry[0] <= gt_ns[-1]
    ]
    if not valid_camera_entries:
        raise ValueError("No cam0 timestamps overlap the GT time range.")

    origin_ns = int(gt_ns[0])
    gt_times = (gt_ns - origin_ns) * 1e-9
    camera_ns = np.asarray(
        [entry[0] for entry in valid_camera_entries],
        dtype=np.int64,
    )
    camera_times = (camera_ns - origin_ns) * 1e-9

    interpolated_positions = np.column_stack(
        [
            np.interp(camera_times, gt_times, gt_positions[:, axis])
            for axis in range(3)
        ]
    )
    interpolated_rotations = Slerp(
        gt_times,
        Rotation.from_quat(gt_quaternions),
    )(camera_times)

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_path = output_dir / "gt_cam0_tum.txt"
    image_list_path = output_dir / "cam0_images.txt"

    if container_sequence_root is None:
        image_root = cam0_root / "data"
    else:
        image_root = Path(container_sequence_root) / "cam0" / "data"

    with pose_path.open("w", encoding="utf-8") as pose_file, image_list_path.open(
        "w", encoding="utf-8"
    ) as image_list_file:
        pose_file.write("# timestamp tx ty tz qx qy qz qw\n")

        for index, (timestamp_ns, filename) in enumerate(valid_camera_entries):
            source_image = cam0_root / "data" / filename
            if not source_image.is_file():
                raise FileNotFoundError(source_image)

            t_ref_body = np.eye(4)
            t_ref_body[:3, :3] = interpolated_rotations[index].as_matrix()
            t_ref_body[:3, 3] = interpolated_positions[index]
            t_ref_cam0 = t_ref_body @ t_body_cam0

            quaternion = Rotation.from_matrix(
                t_ref_cam0[:3, :3]
            ).as_quat()
            translation = t_ref_cam0[:3, 3]
            timestamp_seconds = timestamp_ns * 1e-9

            pose_values = [timestamp_seconds, *translation, *quaternion]
            pose_file.write(
                " ".join(f"{value:.9f}" for value in pose_values) + "\n"
            )
            image_list_file.write(str(image_root / filename) + "\n")

    return {
        "camera_rows": len(camera_entries),
        "gt_rows": len(gt_ns),
        "selected_images": len(valid_camera_entries),
        "discarded_images": len(camera_entries) - len(valid_camera_entries),
        "image_list": str(image_list_path),
        "gt_cam0": str(pose_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare exact cam0 GT poses for an EuRoC ASL sequence"
    )
    parser.add_argument(
        "--sequence_root",
        required=True,
        help="Path to the EuRoC mav0 directory",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--container_sequence_root",
        default=None,
        help="mav0 path written into the generated image list",
    )
    args = parser.parse_args()

    summary = prepare_sequence(
        args.sequence_root,
        args.output_dir,
        args.container_sequence_root,
    )
    print("EuRoC GT preparation complete:", summary)


if __name__ == "__main__":
    main()
