import gtsam
import numpy as np
from scipy.spatial.transform import Rotation


class GroundTruthPoseSource:
    """Load exact camera-to-world poses from a TUM trajectory file."""

    def __init__(self, pose_file, association_tolerance=0.02):
        if association_tolerance < 0.0:
            raise ValueError("association_tolerance must be non-negative.")

        self.association_tolerance = association_tolerance
        self.poses = self._load_pose_file(pose_file)
        self.frame_ids = np.array(sorted(self.poses), dtype=float)

    @staticmethod
    def _load_pose_file(pose_file):
        poses = {}

        with open(pose_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                values = line.split()
                if len(values) != 8:
                    raise ValueError(
                        f"Expected 8 values at {pose_file}:{line_number}, "
                        f"got {len(values)}."
                    )

                frame_id, tx, ty, tz, qx, qy, qz, qw = map(float, values)
                if frame_id in poses:
                    raise ValueError(
                        f"Duplicate frame_id {frame_id} in {pose_file}."
                    )

                quaternion_xyzw = np.array([qx, qy, qz, qw])
                quaternion_norm = np.linalg.norm(quaternion_xyzw)
                if quaternion_norm < 1e-12:
                    raise ValueError(
                        f"Zero quaternion at {pose_file}:{line_number}."
                    )

                rotation = Rotation.from_quat(
                    quaternion_xyzw / quaternion_norm
                ).as_matrix()
                poses[frame_id] = gtsam.Pose3(
                    gtsam.Rot3(rotation),
                    np.array([tx, ty, tz]),
                )

        if not poses:
            raise ValueError(f"No poses found in {pose_file}.")

        return poses

    def _match_frame_id(self, requested_frame_id):
        requested_frame_id = float(requested_frame_id)
        insertion_index = np.searchsorted(
            self.frame_ids,
            requested_frame_id,
        )

        candidate_indices = []
        if insertion_index < len(self.frame_ids):
            candidate_indices.append(insertion_index)
        if insertion_index > 0:
            candidate_indices.append(insertion_index - 1)

        matched_index = min(
            candidate_indices,
            key=lambda index: abs(
                self.frame_ids[index] - requested_frame_id
            ),
        )
        matched_frame_id = self.frame_ids[matched_index]
        association_error = abs(matched_frame_id - requested_frame_id)

        if association_error > self.association_tolerance:
            raise KeyError(
                f"No GT pose for frame_id {requested_frame_id}. "
                f"Nearest id is {matched_frame_id} "
                f"(difference {association_error}, tolerance "
                f"{self.association_tolerance})."
            )

        return matched_frame_id

    def get_pose(self, frame_id):
        """Return the exact camera-to-world pose associated with a frame."""
        matched_frame_id = self._match_frame_id(frame_id)
        return self.poses[matched_frame_id]

    def match_frame_id(self, frame_id):
        """Return the GT timestamp associated with an input frame."""
        return self._match_frame_id(frame_id)

    def get_relative_pose(self, frame_id_i, frame_id_j):
        """Return the exact relative transform T_Ci_Cj."""
        pose_i = self.get_pose(frame_id_i)
        pose_j = self.get_pose(frame_id_j)
        return pose_i.between(pose_j)
