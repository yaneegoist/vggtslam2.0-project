import os
import tempfile

import numpy as np

from vggt_slam.metric_pose_source import MetricPoseSource


POSE_FILE_CONTENT = """\
# frame_id tx ty tz qx qy qz qw
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0
2.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0
"""


def write_temporary_pose_file():
    file_descriptor, path = tempfile.mkstemp(
        prefix="metric_poses_",
        suffix=".txt",
        text=True,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
        file.write(POSE_FILE_CONTENT)
    return path


def test_exact_relative_pose(pose_file):
    source = MetricPoseSource(pose_file)
    relative_pose = source.get_relative_pose(0.0, 2.0)

    assert np.isclose(relative_pose.x(), 2.0, atol=1e-12)
    assert np.isclose(relative_pose.y(), 0.0, atol=1e-12)
    assert np.isclose(relative_pose.z(), 0.0, atol=1e-12)

    print("exact relative pose test: OK")


def test_overlap_identity_and_cache(pose_file):
    source = MetricPoseSource(
        pose_file,
        translation_sigma_m=0.10,
        rotation_sigma_deg=2.0,
        random_seed=7,
    )

    pose_first_query = source.get_pose(1.0).matrix()
    pose_second_query = source.get_pose(1.0).matrix()
    overlap_relative = source.get_relative_pose(1.0, 1.0).matrix()

    assert np.allclose(pose_first_query, pose_second_query)
    assert np.allclose(overlap_relative, np.eye(4), atol=1e-12)

    print("overlap identity and cache test: OK")


def test_seed_reproducibility_and_association(pose_file):
    source_a = MetricPoseSource(
        pose_file,
        translation_sigma_m=0.10,
        rotation_sigma_deg=2.0,
        random_seed=11,
        association_tolerance=1e-3,
    )
    source_b = MetricPoseSource(
        pose_file,
        translation_sigma_m=0.10,
        rotation_sigma_deg=2.0,
        random_seed=11,
        association_tolerance=1e-3,
    )

    pose_a = source_a.get_pose(1.0004).matrix()
    pose_b = source_b.get_pose(1.0).matrix()

    assert np.allclose(pose_a, pose_b)

    print("seed reproducibility and association test: OK")


if __name__ == "__main__":
    temporary_pose_file = write_temporary_pose_file()
    try:
        test_exact_relative_pose(temporary_pose_file)
        test_overlap_identity_and_cache(temporary_pose_file)
        test_seed_reproducibility_and_association(temporary_pose_file)
    finally:
        os.remove(temporary_pose_file)
