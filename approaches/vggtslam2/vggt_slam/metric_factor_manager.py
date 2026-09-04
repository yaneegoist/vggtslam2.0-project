from dataclasses import dataclass

import gtsam
import numpy as np


@dataclass
class _MetricNode:
    node_id: int
    frame_id: float
    matched_frame_id: float
    intrinsic: np.ndarray


class MetricFactorManager:
    """Select node pairs and add metric factors to a PoseGraph."""

    def __init__(
        self,
        pose_graph,
        pose_source,
        factor_translation_sigma_m,
        factor_rotation_sigma_deg,
        numerical_derivative_scheme="central",
    ):
        if factor_translation_sigma_m <= 0.0:
            raise ValueError(
                "factor_translation_sigma_m must be positive."
            )
        if factor_rotation_sigma_deg <= 0.0:
            raise ValueError(
                "factor_rotation_sigma_deg must be positive."
            )

        self.pose_graph = pose_graph
        self.pose_source = pose_source
        self.numerical_derivative_scheme = numerical_derivative_scheme

        rotation_sigma_rad = np.deg2rad(
            factor_rotation_sigma_deg
        )
        self.noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [rotation_sigma_rad] * 3
                + [factor_translation_sigma_m] * 3
            )
        )

        self.nodes = {}
        self.previous_node = None
        self.added_pairs = set()
        self.num_overlap_factors = 0
        self.num_temporal_factors = 0

    def _add_factor(self, node_i, node_j, factor_type):
        pair = (node_i.node_id, node_j.node_id)
        if pair in self.added_pairs:
            return False

        measured_relative_pose = self.pose_source.get_relative_pose(
            node_i.frame_id,
            node_j.frame_id,
        )
        self.pose_graph.add_metric_between_factor(
            node_i.node_id,
            node_j.node_id,
            node_i.intrinsic,
            node_j.intrinsic,
            measured_relative_pose,
            self.noise_model,
            numerical_derivative_scheme=self.numerical_derivative_scheme,
        )

        self.added_pairs.add(pair)
        if factor_type == "overlap":
            self.num_overlap_factors += 1
        else:
            self.num_temporal_factors += 1
        return True

    def register_node(self, node_id, frame_id, intrinsic):
        """Register a graph node and add any newly observable factor."""
        node_id = int(node_id)
        if node_id in self.nodes:
            raise ValueError(f"Metric node {node_id} is already registered.")

        node = _MetricNode(
            node_id=node_id,
            frame_id=float(frame_id),
            matched_frame_id=self.pose_source.match_frame_id(frame_id),
            intrinsic=np.asarray(intrinsic, dtype=float).copy(),
        )
        self.nodes[node_id] = node

        if self.previous_node is not None:
            is_overlap_duplicate = (
                node.matched_frame_id
                == self.previous_node.matched_frame_id
            )

            if is_overlap_duplicate:
                self._add_factor(
                    self.previous_node,
                    node,
                    factor_type="overlap",
                )
            else:
                self._add_factor(
                    self.previous_node,
                    node,
                    factor_type="temporal",
                )

        self.previous_node = node

    def get_summary(self):
        """Return counts useful for experiment logging."""
        return {
            "registered_nodes": len(self.nodes),
            "metric_factors": len(self.added_pairs),
            "overlap_factors": self.num_overlap_factors,
            "temporal_factors": self.num_temporal_factors,
        }
