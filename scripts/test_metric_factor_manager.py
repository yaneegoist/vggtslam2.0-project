import os
import tempfile

import numpy as np

from vggt_slam.graph import PoseGraph
from vggt_slam.metric_factor_manager import MetricFactorManager
from vggt_slam.metric_pose_source import MetricPoseSource


POSE_FILE_CONTENT = """\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.4 0.4 0.0 0.0 0.0 0.0 0.0 1.0
1.2 1.2 0.0 0.0 0.0 0.0 0.0 1.0
"""


def translation_matrix(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def write_temporary_pose_file():
    file_descriptor, path = tempfile.mkstemp(
        prefix="metric_factor_manager_",
        suffix=".txt",
        text=True,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
        file.write(POSE_FILE_CONTENT)
    return path


def test_overlap_and_baseline_selection(pose_file):
    pose_graph = PoseGraph()
    pose_source = MetricPoseSource(pose_file)
    manager = MetricFactorManager(
        pose_graph=pose_graph,
        pose_source=pose_source,
        min_baseline_m=1.0,
        factor_translation_sigma_m=0.10,
        factor_rotation_sigma_deg=2.0,
    )

    node_data = [
        (0, 0.0, 0.0),
        (1, 0.0, 0.0),
        (2, 0.4, 0.4),
        (3, 1.2, 1.2),
    ]

    for node_id, frame_id, x in node_data:
        pose_graph.add_homography(node_id, translation_matrix(x))
        manager.register_node(node_id, frame_id, np.eye(3))

    summary = manager.get_summary()

    assert summary == {
        "registered_nodes": 4,
        "metric_factors": 2,
        "overlap_factors": 1,
        "baseline_factors": 1,
    }
    assert manager.added_pairs == {(0, 1), (1, 3)}
    assert pose_graph.graph.size() == 2
    assert pose_graph.graph.error(pose_graph.values) < 1e-18

    print("metric factor manager test: OK")
    print(f"  summary: {summary}")
    print(f"  pairs: {sorted(manager.added_pairs)}")


def selected_pairs_with_noise(pose_file, random_seed):
    pose_graph = PoseGraph()
    pose_source = MetricPoseSource(
        pose_file,
        translation_sigma_m=0.75,
        rotation_sigma_deg=5.0,
        random_seed=random_seed,
    )
    manager = MetricFactorManager(
        pose_graph=pose_graph,
        pose_source=pose_source,
        min_baseline_m=1.0,
        factor_translation_sigma_m=0.10,
        factor_rotation_sigma_deg=2.0,
    )

    for node_id, frame_id, x in [
        (0, 0.0, 0.0),
        (1, 0.0, 0.0),
        (2, 0.4, 0.4),
        (3, 1.2, 1.2),
    ]:
        pose_graph.add_homography(node_id, translation_matrix(x))
        manager.register_node(node_id, frame_id, np.eye(3))

    return manager.added_pairs


def test_pair_selection_is_seed_independent(pose_file):
    pairs_seed_1 = selected_pairs_with_noise(pose_file, random_seed=1)
    pairs_seed_2 = selected_pairs_with_noise(pose_file, random_seed=2)

    assert pairs_seed_1 == {(0, 1), (1, 3)}
    assert pairs_seed_2 == pairs_seed_1

    print("seed-independent pair selection test: OK")


if __name__ == "__main__":
    temporary_pose_file = write_temporary_pose_file()
    try:
        test_overlap_and_baseline_selection(temporary_pose_file)
        test_pair_selection_is_seed_independent(temporary_pose_file)
    finally:
        os.remove(temporary_pose_file)
