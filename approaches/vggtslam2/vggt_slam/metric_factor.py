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
    rotation = np.asarray(rotation, dtype=float)
    orthogonality_error = np.linalg.norm(
        rotation.T @ rotation - np.eye(3),
        ord="fro",
    )
    if (
        orthogonality_error < 1e-10
        and np.linalg.det(rotation) > 0.0
    ):
        return rotation

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
    return _metric_pose_residual(
        pose_i,
        pose_j,
        measured_relative_pose,
    )


def _metric_pose_residual(
    pose_i,
    pose_j,
    measured_relative_pose,
):
    """Compute the metric residual from already extracted Pose3 values."""
    estimated_relative_pose = pose_i.between(pose_j)

    pose_error = measured_relative_pose.inverse().compose(
        estimated_relative_pose
    )
    return gtsam.Pose3.Logmap(pose_error)


def _pose_extraction_jacobian(
    sl4_value,
    intrinsic,
    base_pose,
    epsilon,
    scheme,
):
    """Differentiate SL4-to-Pose3 once for one graph node."""
    jacobian = np.zeros((METRIC_RESIDUAL_DIM, SL4_TANGENT_DIM))

    for column in range(SL4_TANGENT_DIM):
        delta = np.zeros(SL4_TANGENT_DIM)
        delta[column] = epsilon

        pose_plus = sl4_to_pose3(
            sl4_value.retract(delta),
            intrinsic,
        )
        local_plus = np.asarray(
            base_pose.localCoordinates(pose_plus),
            dtype=float,
        )

        if scheme == "forward":
            jacobian[:, column] = local_plus / epsilon
        else:
            pose_minus = sl4_to_pose3(
                sl4_value.retract(-delta),
                intrinsic,
            )
            local_minus = np.asarray(
                base_pose.localCoordinates(pose_minus),
                dtype=float,
            )
            jacobian[:, column] = (
                local_plus - local_minus
            ) / (2.0 * epsilon)

    return np.asfortranarray(jacobian)


def _pose_residual_jacobian(
    pose_i,
    pose_j,
    measured_relative_pose,
    variable_index,
    epsilon,
    scheme,
    base_residual,
):
    """Differentiate the cheap Pose3 residual with respect to one pose."""
    jacobian = np.zeros((METRIC_RESIDUAL_DIM, METRIC_RESIDUAL_DIM))

    def evaluate(signed_delta):
        if variable_index == 0:
            return _metric_pose_residual(
                pose_i.retract(signed_delta),
                pose_j,
                measured_relative_pose,
            )
        return _metric_pose_residual(
            pose_i,
            pose_j.retract(signed_delta),
            measured_relative_pose,
        )

    for column in range(METRIC_RESIDUAL_DIM):
        delta = np.zeros(METRIC_RESIDUAL_DIM)
        delta[column] = epsilon

        residual_plus = evaluate(delta)
        if scheme == "forward":
            jacobian[:, column] = (
                residual_plus - base_residual
            ) / epsilon
        else:
            residual_minus = evaluate(-delta)
            jacobian[:, column] = (
                residual_plus - residual_minus
            ) / (2.0 * epsilon)

    return np.asfortranarray(jacobian)


class MetricFactorLinearizationCache:
    """Share expensive SL4-to-Pose3 derivatives between adjacent factors."""

    def __init__(self, epsilon=1e-6, scheme="forward"):
        if scheme not in {"central", "forward"}:
            raise ValueError("scheme must be 'central' or 'forward'.")

        self.epsilon = float(epsilon)
        self.scheme = scheme
        self._entries = {}

    @staticmethod
    def _matches(entry, matrix, intrinsic):
        return (
            entry is not None
            and np.array_equal(entry[0], matrix)
            and np.array_equal(entry[1], intrinsic)
        )

    def get_pose(self, key, sl4_value, intrinsic):
        """Return a cached pose without unnecessarily forming its Jacobian."""
        matrix = np.asarray(sl4_value.matrix(), dtype=float)
        intrinsic = np.asarray(intrinsic, dtype=float)
        cache_key = int(key)
        entry = self._entries.get(cache_key)

        if self._matches(entry, matrix, intrinsic):
            return entry[2]

        pose = sl4_to_pose3(sl4_value, intrinsic)
        self._entries[cache_key] = (
            matrix.copy(),
            intrinsic.copy(),
            pose,
            None,
        )
        return pose

    def get_pose_and_jacobian(
        self,
        key,
        sl4_value,
        intrinsic,
    ):
        matrix = np.asarray(sl4_value.matrix(), dtype=float)
        intrinsic = np.asarray(intrinsic, dtype=float)
        cache_key = int(key)
        entry = self._entries.get(cache_key)

        if self._matches(entry, matrix, intrinsic) and entry[3] is not None:
            return entry[2], entry[3]

        if self._matches(entry, matrix, intrinsic):
            pose = entry[2]
        else:
            pose = sl4_to_pose3(sl4_value, intrinsic)
        jacobian = _pose_extraction_jacobian(
            sl4_value,
            intrinsic,
            pose,
            self.epsilon,
            self.scheme,
        )
        self._entries[cache_key] = (
            matrix.copy(),
            intrinsic.copy(),
            pose,
            jacobian,
        )
        return pose, jacobian


def _numerical_jacobian(
    sl4_i,
    sl4_j,
    intrinsic_i,
    intrinsic_j,
    measured_relative_pose,
    variable_index,
    epsilon,
    scheme,
    base_residual,
):
    jacobian = np.zeros((METRIC_RESIDUAL_DIM, SL4_TANGENT_DIM))

    def evaluate(signed_delta):
        if variable_index == 0:
            return metric_between_residual(
                sl4_i.retract(signed_delta),
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
            )
        return metric_between_residual(
            sl4_i,
            sl4_j.retract(signed_delta),
            intrinsic_i,
            intrinsic_j,
            measured_relative_pose,
        )

    for column in range(SL4_TANGENT_DIM):
        delta = np.zeros(SL4_TANGENT_DIM)
        delta[column] = epsilon

        residual_plus = evaluate(delta)
        if scheme == "forward":
            jacobian[:, column] = (
                residual_plus - base_residual
            ) / epsilon
        else:
            residual_minus = evaluate(-delta)
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
    numerical_derivative_scheme="central",
    linearization_cache=None,
):
    """Create a 6D metric CustomFactor between two SL(4) variables."""
    if numerical_derivative_scheme not in {
        "central",
        "forward",
        "cached_forward",
    }:
        raise ValueError(
            "numerical_derivative_scheme must be 'central', 'forward', "
            "or 'cached_forward'."
        )
    if (
        numerical_derivative_scheme == "cached_forward"
        and linearization_cache is None
    ):
        raise ValueError(
            "cached_forward requires a shared linearization cache."
        )
    intrinsic_i = np.asarray(intrinsic_i, dtype=float).copy()
    intrinsic_j = np.asarray(intrinsic_j, dtype=float).copy()

    def error_function(_, values, jacobians):
        sl4_i = values.atSL4(key_i)
        sl4_j = values.atSL4(key_j)

        if numerical_derivative_scheme == "cached_forward":
            pose_i = linearization_cache.get_pose(
                key_i,
                sl4_i,
                intrinsic_i,
            )
            pose_j = linearization_cache.get_pose(
                key_j,
                sl4_j,
                intrinsic_j,
            )
            residual = _metric_pose_residual(
                pose_i,
                pose_j,
                measured_relative_pose,
            )

            if jacobians is None:
                return residual

            pose_i, pose_extraction_jacobian_i = (
                linearization_cache.get_pose_and_jacobian(
                    key_i,
                    sl4_i,
                    intrinsic_i,
                )
            )
            pose_j, pose_extraction_jacobian_j = (
                linearization_cache.get_pose_and_jacobian(
                    key_j,
                    sl4_j,
                    intrinsic_j,
                )
            )

            residual_jacobian_i = _pose_residual_jacobian(
                pose_i,
                pose_j,
                measured_relative_pose,
                variable_index=0,
                epsilon=numerical_derivative_epsilon,
                scheme="forward",
                base_residual=residual,
            )
            residual_jacobian_j = _pose_residual_jacobian(
                pose_i,
                pose_j,
                measured_relative_pose,
                variable_index=1,
                epsilon=numerical_derivative_epsilon,
                scheme="forward",
                base_residual=residual,
            )

            jacobians[0] = np.asfortranarray(
                residual_jacobian_i @ pose_extraction_jacobian_i
            )
            jacobians[1] = np.asfortranarray(
                residual_jacobian_j @ pose_extraction_jacobian_j
            )
            return residual

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
                scheme=numerical_derivative_scheme,
                base_residual=residual,
            )
            jacobians[1] = _numerical_jacobian(
                sl4_i,
                sl4_j,
                intrinsic_i,
                intrinsic_j,
                measured_relative_pose,
                variable_index=1,
                epsilon=numerical_derivative_epsilon,
                scheme=numerical_derivative_scheme,
                base_residual=residual,
            )

        return residual

    return gtsam.CustomFactor(
        noise_model,
        [key_i, key_j],
        error_function,
    )
