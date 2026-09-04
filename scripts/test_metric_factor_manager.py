import os
import tempfile

import numpy as np

from vggt_slam.graph import PoseGraph
from vggt_slam.metric_factor_manager import MetricFactorManager
from vggt_slam.ground_truth_pose_source import GroundTruthPoseSource


POSE_FILE_CONTENT = """\
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.4 0.4 0.0 0.0 0.0 0.0 0.0 1.0
1.2 1.2 0.0 0.0 0.0 0.0 0.0 1.0
"""

GAUGE_TRANSFORMED_POSE_FILE_CONTENT = """\
0.0 7.0 -2.0 1.0 0.0 0.0 0.7071067811865475 0.7071067811865476
0.4 7.0 -1.6 1.0 0.0 0.0 0.7071067811865475 0.7071067811865476
1.2 7.0 -0.8 1.0 0.0 0.0 0.7071067811865475 0.7071067811865476
"""


def translation_matrix(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def write_temporary_pose_file(content=POSE_FILE_CONTENT):
    file_descriptor, path = tempfile.mkstemp(
        prefix="metric_factor_manager_",
        suffix=".txt",
        text=True,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
        file.write(content)
    return path


def test_overlap_and_temporal_factors(pose_file):
    pose_graph = PoseGraph()
    pose_source = GroundTruthPoseSource(pose_file)
    manager = MetricFactorManager(
        pose_graph=pose_graph,
        pose_source=pose_source,
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
        "metric_factors": 3,
        "overlap_factors": 1,
        "temporal_factors": 2,
        "metric_initialized_nodes": 0,
    }
    assert manager.added_pairs == {(0, 1), (1, 2), (2, 3)}
    assert pose_graph.graph.size() == 3
    assert pose_graph.graph.error(pose_graph.values) < 1e-18

    print("metric factor manager test: OK")
    print(f"  summary: {summary}")
    print(f"  pairs: {sorted(manager.added_pairs)}")


def initialized_homographies(pose_file):
    pose_graph = PoseGraph()
    pose_source = GroundTruthPoseSource(pose_file)
    manager = MetricFactorManager(
        pose_graph=pose_graph,
        pose_source=pose_source,
        factor_translation_sigma_m=0.10,
        factor_rotation_sigma_deg=2.0,
        initialize_nodes_from_metric_poses=True,
    )

    frame_ids = [0.0, 0.4, 1.2]
    deliberately_wrong_x = [5.0, -20.0, 80.0]
    for node_id, (frame_id, x) in enumerate(
        zip(frame_ids, deliberately_wrong_x)
    ):
        pose_graph.add_homography(node_id, translation_matrix(x))
        manager.register_node(node_id, frame_id, np.eye(3))

    homographies = [
        pose_graph.get_homography(node_id)
        for node_id in range(len(frame_ids))
    ]
    assert manager.get_summary()["metric_initialized_nodes"] == 3
    assert pose_graph.graph.error(pose_graph.values) < 1e-18
    return homographies


def test_metric_initialization_is_gauge_invariant(
    pose_file,
    gauge_transformed_pose_file,
):
    initialized = initialized_homographies(pose_file)
    gauge_transformed = initialized_homographies(
        gauge_transformed_pose_file
    )

    expected_x = [5.0, 5.4, 6.2]
    for node_id, expected in enumerate(expected_x):
        assert np.isclose(initialized[node_id][0, 3], expected)
        assert np.allclose(
            initialized[node_id],
            gauge_transformed[node_id],
            atol=1e-10,
        )

    print("gauge-invariant metric initialization test: OK")


if __name__ == "__main__":
    temporary_pose_file = write_temporary_pose_file()
    gauge_transformed_pose_file = write_temporary_pose_file(
        GAUGE_TRANSFORMED_POSE_FILE_CONTENT
    )
    try:
        test_overlap_and_temporal_factors(temporary_pose_file)
        test_metric_initialization_is_gauge_invariant(
            temporary_pose_file,
            gauge_transformed_pose_file,
        )
    finally:
        os.remove(temporary_pose_file)
        os.remove(gauge_transformed_pose_file)
