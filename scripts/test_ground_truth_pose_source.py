import os
import tempfile

import numpy as np

from vggt_slam.ground_truth_pose_source import GroundTruthPoseSource


POSE_FILE_CONTENT = """\
# timestamp tx ty tz qx qy qz qw
0.000 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1.000 1.0 0.0 0.0 0.0 0.0 0.0 1.0
2.000 2.0 0.0 0.0 0.0 0.0 0.0 1.0
"""


def write_temporary_pose_file():
    file_descriptor, path = tempfile.mkstemp(
        prefix="ground_truth_poses_",
        suffix=".txt",
        text=True,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
        file.write(POSE_FILE_CONTENT)
    return path


def test_exact_relative_pose(pose_file):
    source = GroundTruthPoseSource(pose_file)
    relative_pose = source.get_relative_pose(0.0, 2.0)

    assert np.isclose(relative_pose.x(), 2.0, atol=1e-12)
    assert np.isclose(relative_pose.y(), 0.0, atol=1e-12)
    assert np.isclose(relative_pose.z(), 0.0, atol=1e-12)
    print("exact relative pose test: OK")


def test_overlap_identity_and_association(pose_file):
    source = GroundTruthPoseSource(
        pose_file,
        association_tolerance=0.02,
    )

    overlap_relative = source.get_relative_pose(1.006, 1.006).matrix()
    assert source.match_frame_id(1.006) == 1.0
    assert np.allclose(overlap_relative, np.eye(4), atol=1e-12)
    print("overlap identity and timestamp association test: OK")


if __name__ == "__main__":
    temporary_pose_file = write_temporary_pose_file()
    try:
        test_exact_relative_pose(temporary_pose_file)
        test_overlap_identity_and_association(temporary_pose_file)
    finally:
        os.remove(temporary_pose_file)
