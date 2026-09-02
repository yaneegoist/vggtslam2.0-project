import gtsam
import numpy as np

from vggt_slam.slam_utils import decompose_camera


SL4_TANGENT_DIM = 15
METRIC_RESIDUAL_DIM = 6


def _intrinsic_to_4x4(intrinsic):
    """Return a homogeneous 4x4 intrinsic matrix."""
    intrinsic = np.asarray(intrinsic, dtype=float)

    if intrinsic.shape == (4, 4):
        return intrinsic.copy()

    if intrinsic.shape != (3, 3):
        raise ValueError(
            "Camera intrinsic must have shape (3, 3) or (4, 4), "
            f"got {intrinsic.shape}."
        )

    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = intrinsic
    return intrinsic_4x4


def _project_to_so3(rotation):
    """Project a nearly rotational 3x3 matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(rotation)
    rotation_so3 = u @ vt

    if np.linalg.det(rotation_so3) < 0:
        u[:, -1] *= -1
        rotation_so3 = u @ vt

    return rotation_so3


def sl4_to_pose3(sl4_value, intrinsic):
    """Extract the physical camera-to-world pose from an SL(4) node."""
    intrinsic_4x4 = _intrinsic_to_4x4(intrinsic)
    homography = sl4_value.matrix()
    projection_matrix = intrinsic_4x4 @ np.linalg.inv(homography)

    _, rotation, translation, _ = decompose_camera(projection_matrix)
    rotation = _project_to_so3(rotation)

    return gtsam.Pose3(
        gtsam.Rot3(rotation),
        np.asarray(translation, dtype=float),
    )


def metric_between_residual(
    sl4_i,
    sl4_j,
    intrinsic_i,
    intrinsic_j,
    measured_relative_pose,
):
    """Compute a six-dimensional SE(3) residual for two SL(4) nodes."""
    pose_i = sl4_to_pose3(sl4_i, intrinsic_i)
    pose_j = sl4_to_pose3(sl4_j, intrinsic_j)
    estimated_relative_pose = pose_i.between(pose_j)

    pose_error = measured_relative_pose.inverse().compose(
        estimated_relative_pose
    )
    return gtsam.Pose3.Logmap(pose_error)


def _numerical_jacobian(
    sl4_i,
    sl4_j,
    intrinsic_i,
    intrinsic_j,
    measured_relative_pose,
    variable_index,
    epsilon,
):
    jacobian = np.zeros((METRIC_RESIDUAL_DIM, SL4_TANGENT_DIM))

    for column in range(SL4_TANGENT_DIM):
        delta = np.zeros(SL4_TANGENT_DIM)
        delta[column] = epsilon

        if variable_index == 0:
            residual_plus = metric_between_residual(
                sl4_i.retract(delta),
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
            )
            residual_minus = metric_between_residual(
                sl4_i.retract(-delta),
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
            )
        else:
            residual_plus = metric_between_residual(
                sl4_i,
                sl4_j.retract(delta),
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
            )
            residual_minus = metric_between_residual(
                sl4_i,
                sl4_j.retract(-delta),
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
            )

        jacobian[:, column] = (
            residual_plus - residual_minus
        ) / (2.0 * epsilon)

    return np.asfortranarray(jacobian)


def make_metric_between_factor(
    key_i,
    key_j,
    intrinsic_i,
    intrinsic_j,
    measured_relative_pose,
    noise_model,
    numerical_derivative_epsilon=1e-6,
):
    """Create a 6D metric CustomFactor between two SL(4) variables."""
    intrinsic_i = np.asarray(intrinsic_i, dtype=float).copy()
    intrinsic_j = np.asarray(intrinsic_j, dtype=float).copy()

    def error_function(_, values, jacobians):
        sl4_i = values.atSL4(key_i)
        sl4_j = values.atSL4(key_j)

        residual = metric_between_residual(
            sl4_i,
            sl4_j,
            intrinsic_i,
            intrinsic_j,
            measured_relative_pose,
        )

        if jacobians is not None:
            jacobians[0] = _numerical_jacobian(
                sl4_i,
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
                variable_index=0,
                epsilon=numerical_derivative_epsilon,
            )
            jacobians[1] = _numerical_jacobian(
                sl4_i,
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
                variable_index=1,
                epsilon=numerical_derivative_epsilon,
            )

        return residual

    return gtsam.CustomFactor(
        noise_model,
        [key_i, key_j],
        error_function,
    )
