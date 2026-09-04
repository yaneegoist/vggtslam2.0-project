import gtsam
import numpy as np

from gtsam.symbol_shorthand import X

from vggt_slam.graph import PoseGraph
from vggt_slam.metric_factor import (
    metric_between_residual,
    sl4_to_pose3,
)
from vggt_slam.slam_utils import decompose_camera


def translation_matrix(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def make_pose3(x):
    return gtsam.Pose3(
        gtsam.Rot3(np.eye(3)),
        np.array([x, 0.0, 0.0]),
    )


def test_parallel_metric_factor(jacobian_scheme):
    pose_graph = PoseGraph()

    initial_i = translation_matrix(0.0)
    initial_j = translation_matrix(1.5)

    pose_graph.add_homography(0, initial_i)
    pose_graph.add_homography(1, initial_j)
    pose_graph.add_prior_factor(0, initial_i)

    visual_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.full(15, 0.20)
    )
    metric_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.full(6, 0.05)
    )

    pose_graph.add_between_factor(
        0,
        1,
        translation_matrix(1.5),
        visual_noise,
    )
    pose_graph.add_metric_between_factor(
        0,
        1,
        np.eye(3),
        np.eye(3),
        make_pose3(1.0),
        metric_noise,
        numerical_derivative_scheme=jacobian_scheme,
    )

    initial_x = sl4_to_pose3(
        pose_graph.values.atSL4(X(1)),
        np.eye(3),
    ).x()

    pose_graph.optimize()

    optimized_x = sl4_to_pose3(
        pose_graph.values.atSL4(X(1)),
        np.eye(3),
    ).x()

    visual_weight = 1.0 / (0.20 ** 2)
    metric_weight = 1.0 / (0.05 ** 2)
    expected_x = (
        visual_weight * 1.5 + metric_weight * 1.0
    ) / (visual_weight + metric_weight)

    assert np.isclose(initial_x, 1.5, atol=1e-9)
    assert np.isclose(optimized_x, expected_x, atol=1e-6)

    print(f"parallel factor test ({jacobian_scheme}): OK")
    print(f"  initial x:   {initial_x:.9f}")
    print(f"  optimized x: {optimized_x:.9f}")
    print(f"  expected x:  {expected_x:.9f}")


def homography_from_camera(pose_camera_to_world, effective_intrinsic):
    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = effective_intrinsic
    world_to_camera = np.linalg.inv(pose_camera_to_world)
    projection_matrix = intrinsic_4x4 @ world_to_camera
    return np.linalg.inv(projection_matrix)


def test_projective_pose_extraction():
    pose_i = translation_matrix(0.0)
    pose_j = translation_matrix(2.0)

    effective_intrinsic_i = np.diag([2.0, 0.5, 1.0])
    effective_intrinsic_j = np.diag([0.5, 2.0, 1.0])

    homography_i = homography_from_camera(
        pose_i,
        effective_intrinsic_i,
    )
    homography_j = homography_from_camera(
        pose_j,
        effective_intrinsic_j,
    )

    sl4_i = gtsam.SL4(homography_i)
    sl4_j = gtsam.SL4(homography_j)

    extracted_i = sl4_to_pose3(sl4_i, np.eye(3))
    extracted_j = sl4_to_pose3(sl4_j, np.eye(3))
    residual = metric_between_residual(
        sl4_i,
        sl4_j,
        np.eye(3),
        np.eye(3),
        make_pose3(2.0),
    )

    assert not np.allclose(homography_i[:3, :3], np.eye(3))
    assert not np.allclose(homography_j[:3, :3], np.eye(3))
    assert np.isclose(extracted_i.x(), 0.0, atol=1e-9)
    assert np.isclose(extracted_j.x(), 2.0, atol=1e-9)
    assert np.linalg.norm(residual) < 1e-9

    print("projective extraction test: OK")
    print(f"  extracted x_i: {extracted_i.x():.9f}")
    print(f"  extracted x_j: {extracted_j.x():.9f}")
    print(f"  residual norm: {np.linalg.norm(residual):.3e}")


def test_projection_sign_invariance():
    angle = np.deg2rad(25.0)
    rotation_camera_to_world = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose_camera_to_world = np.eye(4)
    pose_camera_to_world[:3, :3] = rotation_camera_to_world
    pose_camera_to_world[:3, 3] = np.array([1.0, -2.0, 0.5])

    intrinsic = np.array(
        [
            [450.0, 0.0, 370.0],
            [0.0, 451.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projection = intrinsic @ np.linalg.inv(pose_camera_to_world)[:3, :]

    _, rotation_positive, translation_positive, _ = decompose_camera(
        projection
    )
    _, rotation_negative, translation_negative, _ = decompose_camera(
        -projection
    )

    assert np.linalg.det(rotation_positive) > 0.0
    assert np.linalg.det(rotation_negative) > 0.0
    assert np.allclose(rotation_positive, rotation_negative, atol=1e-10)
    assert np.allclose(translation_positive, translation_negative, atol=1e-10)

    print("projection sign invariance test: OK")


if __name__ == "__main__":
    test_parallel_metric_factor("central")
    test_parallel_metric_factor("forward")
    test_projective_pose_extraction()
    test_projection_sign_invariance()
