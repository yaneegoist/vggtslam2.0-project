import tempfile
from pathlib import Path

import numpy as np

from prepare_euroc_gt import prepare_sequence


def write_test_sequence(root):
    cam0 = root / "cam0"
    gt = root / "state_groundtruth_estimate0"
    (cam0 / "data").mkdir(parents=True)
    gt.mkdir(parents=True)

    (cam0 / "sensor.yaml").write_text(
        """T_BS:
  cols: 4
  rows: 4
  data: [1, 0, 0, 1,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1]
""",
        encoding="utf-8",
    )
    (cam0 / "data.csv").write_text(
        """#timestamp [ns],filename
500000000,500000000.png
1000000000,1000000000.png
1500000000,1500000000.png
2000000000,2000000000.png
2500000000,2500000000.png
""",
        encoding="utf-8",
    )
    for timestamp in range(500000000, 2500000001, 500000000):
        (cam0 / "data" / f"{timestamp}.png").touch()

    (gt / "data.csv").write_text(
        """#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z
1000000000,0,0,0,1,0,0,0
2000000000,2,0,0,1,0,0,0
""",
        encoding="utf-8",
    )


def read_poses(path):
    return np.loadtxt(path, comments="#")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="euroc_gt_test_") as directory:
        root = Path(directory) / "mav0"
        output = Path(directory) / "prepared"
        write_test_sequence(root)

        summary = prepare_sequence(
            root,
            output,
            container_sequence_root="/dataset/mav0",
        )
        poses = read_poses(output / "gt_cam0_tum.txt")
        image_paths = (output / "cam0_images.txt").read_text(
            encoding="utf-8"
        ).splitlines()

        assert summary["selected_images"] == 3
        assert summary["discarded_images"] == 2
        assert np.allclose(poses[:, 1], [1.0, 2.0, 3.0])
        assert image_paths[0] == "/dataset/mav0/cam0/data/1000000000.png"
        assert image_paths[-1] == "/dataset/mav0/cam0/data/2000000000.png"
        print("EuRoC GT preparation test: OK")
