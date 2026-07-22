#!/usr/bin/env python3

import collections
import math
from time import perf_counter

import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header


_FIELDS_XYZI = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


class ScanAccumulator(Node):
    """Maintains a local submap of distance-gated keyframe scans from
    FAST-LIO and publishes it, plus the current scan, as one dense
    "virtual scan" for the scan matcher.

    Each incoming scan is transformed into the odom frame using the
    odom -> scan-frame tf at the scan's own timestamp. FAST-LIO's tf is
    the exact state pose it registered the scan with, so this reproduces
    its own world registration with no extra drift. A scan is *retained*
    as a keyframe only when the robot has translated keyframe_dist_m or
    rotated keyframe_rot_deg since the last keyframe -- consecutive scans
    are nearly identical, so the ring buffer instead spans a trail of
    roughly n_scans * keyframe_dist_m meters behind the robot, and it
    freezes (rather than flushing the trail) while the robot idles.

    On every scan, keyframes + the current scan are concatenated,
    voxel-downsampled to collapse cross-keyframe overlap (_output_filter),
    re-expressed in base_frame at the current scan's timestamp, and
    published -- so the scan matcher always aligns the robot's current
    view, with the submap trail providing extra constraint.

    Future work:
      - per-scan voxel downsampling (_per_scan_filter) to bound memory/CPU
        further, if a denser lidar ever makes individual scans themselves
        heavy (see docs/2026-07-scan-accumulator-submap.md)
      - decoupled (lower) publish rate if a denser lidar makes per-scan
        publishing expensive
    """

    def __init__(self):
        super().__init__("scan_accumulator")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("n_scans", 10),
                ("keyframe_dist_m", 1.0),
                ("keyframe_rot_deg", 30.0),
                ("odom_frame", "odom"),
                ("base_frame", "base_link"),
                ("tf_timeout", 0.1),
                ("output_voxel_size", 0.1),
                ("profile_timing", False),
            ],
        )

        # (stamp, Nx4 float32 array of xyz+intensity, already in odom frame)
        self.scans = collections.deque(maxlen=self.get_parameter("n_scans").value)
        # 4x4 odom pose of the last retained keyframe's scan frame.
        self.T_last_keyframe = None

        # Diagnostic-only per-stage timing, gated behind profile_timing so it
        # is zero-overhead when disabled. Read once here (not a runtime knob;
        # requires a relaunch to toggle) to avoid a parameter lookup on every
        # callback in the non-profiling path.
        self.profile = self.get_parameter("profile_timing").value
        self._prof = {}  # stage -> [total_s, count, max_s]
        self._prof_last_log = perf_counter()
        self._prof_last_in_points = 0
        self._prof_last_out_points = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_accumulated = self.create_publisher(PointCloud2, "/accumulated_scan", 10)
        # The scan callback blocks up to tf_timeout waiting for tf, so it
        # lives in its own callback group: with the multi-threaded executor,
        # the tf listener's subscriptions (default group) keep filling the
        # buffer during that wait instead of deadlocking behind it.
        self.scan_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            PointCloud2, "/cloud_registered_body", self.cb_scan, 10, callback_group=self.scan_cb_group
        )

    def lookup_mat(self, target_frame, source_frame, stamp):
        """4x4 target <- source transform at stamp, or None if unavailable."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.get_parameter("tf_timeout").value),
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f"No {target_frame} <- {source_frame} tf at scan stamp; dropping: {e}",
                throttle_duration_sec=2.0,
            )
            return None
        trans = tf_msg.transform.translation
        rot = tf_msg.transform.rotation
        T = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T[:3, 3] = [trans.x, trans.y, trans.z]
        return T

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        trans_inverse[:3, :3] = trans[:3, :3].T
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def transform_points(self, transform, points):
        """Applies a 4x4 transform to the xyz columns of an Nx4 array,
        passing intensity through untouched."""
        xyz = np.column_stack([points[:, :3], np.ones(len(points))])
        xyz = np.matmul(transform, xyz.T).T[:, :3]
        return np.column_stack([xyz, points[:, 3]]).astype(np.float32)

    def _prof_add(self, stage, dt):
        entry = self._prof.setdefault(stage, [0.0, 0, 0.0])
        entry[0] += dt
        entry[1] += 1
        entry[2] = max(entry[2], dt)

    def _prof_maybe_log(self):
        """Emits a throttled (~5s) summary of accumulated per-stage timing.
        Only called when profile_timing is enabled -- see cb_scan."""
        now = perf_counter()
        elapsed = now - self._prof_last_log
        if elapsed < 5.0:
            return
        lines = [
            f"  {stage}: mean={1000 * total / count:.2f}ms max={1000 * mx:.2f}ms n={count}"
            for stage, (total, count, mx) in sorted(self._prof.items())
        ]
        self.get_logger().info(
            "scan_accumulator profile (last {:.1f}s, points_in={}, points_out={}):\n{}".format(
                elapsed, self._prof_last_in_points, self._prof_last_out_points, "\n".join(lines)
            )
        )
        self._prof = {}
        self._prof_last_log = now

    def _per_scan_filter(self, points):
        # Seam for future per-scan voxel downsampling.
        return points

    def _output_filter(self, points):
        """Voxel-downsamples the concatenated submap so overlapping
        keyframes collapse to one point per cell, instead of each
        contributing near-duplicate points. Picks one representative point
        per occupied voxel (first occurrence) rather than a true centroid
        average -- cheap, vectorized, and sufficient for ICP correspondence,
        which already tolerates offsets within a voxel_size.

        Voxel indices are packed into a single int64 key (21 bits/axis, safe
        for submap extents up to ~200km at this voxel_size) and deduplicated
        with a 1-D np.unique -- np.unique(..., axis=0) on the raw Nx3 index
        array is a well-known ~4-5x slower path in numpy (generic lexsort)
        for the same result."""
        if len(points) == 0:
            return points
        voxel_size = self.get_parameter("output_voxel_size").value
        if voxel_size <= 0.0:
            return points
        voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)
        offset = voxel_idx - voxel_idx.min(axis=0)
        keys = (offset[:, 0] << 42) | (offset[:, 1] << 21) | offset[:, 2]
        _, unique_indices = np.unique(keys, return_index=True)
        return points[unique_indices]

    def is_keyframe(self, T_odom_to_scan):
        """True when the robot has moved far enough from the last keyframe
        that this scan adds meaningfully new geometry to the submap."""
        if self.T_last_keyframe is None:
            return True
        T_rel = np.matmul(self.inverse_se3(self.T_last_keyframe), T_odom_to_scan)
        if np.linalg.norm(T_rel[:3, 3]) >= self.get_parameter("keyframe_dist_m").value:
            return True
        # Relative rotation angle from the trace of the rotation block.
        cos_angle = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle)) >= self.get_parameter("keyframe_rot_deg").value

    def cb_scan(self, msg):
        if self.profile:
            t_cb_start = perf_counter()

        stamp = rclpy.time.Time.from_msg(msg.header.stamp)

        # A backward time jump (bag loop/restart) invalidates the buffer.
        if self.scans:
            newest = rclpy.time.Time.from_msg(self.scans[-1][0])
            if stamp < newest - Duration(seconds=1.0):
                self.get_logger().warn("Scan stamp jumped backwards; clearing accumulated scans.")
                self.scans.clear()
                self.T_last_keyframe = None

        if self.profile:
            t0 = perf_counter()
        T_odom_to_scan = self.lookup_mat(
            self.get_parameter("odom_frame").value, msg.header.frame_id, msg.header.stamp
        )
        if self.profile:
            self._prof_add("lookup_odom_scan", perf_counter() - t0)
        if T_odom_to_scan is None:
            return

        if self.profile:
            t0 = perf_counter()
        points = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"))
        if self.profile:
            self._prof_add("read_points", perf_counter() - t0)
            self._prof_last_in_points = len(points)
        if len(points) == 0:
            return

        if self.profile:
            t0 = perf_counter()
        points = self.transform_points(T_odom_to_scan, points)
        if self.profile:
            self._prof_add("transform_scan", perf_counter() - t0)
        points = self._per_scan_filter(points)

        if self.profile:
            t0 = perf_counter()
        is_kf = self.is_keyframe(T_odom_to_scan)
        if self.profile:
            self._prof_add("keyframe_check", perf_counter() - t0)

        if is_kf:
            self.scans.append((msg.header.stamp, points))
            self.T_last_keyframe = T_odom_to_scan
            self.get_logger().debug(f"Keyframe added ({len(self.scans)} in buffer).")
            self.publish_accumulated(msg.header.stamp)
        else:
            # Not retained, but the current view always heads the output so
            # the scan matcher aligns what the robot sees right now.
            self.publish_accumulated(msg.header.stamp, current_points=points)

        if self.profile:
            self._prof_add("callback_total", perf_counter() - t_cb_start)
            self._prof_maybe_log()

    def publish_accumulated(self, stamp, current_points=None):
        base_frame = self.get_parameter("base_frame").value

        if self.profile:
            t0 = perf_counter()
        T_odom_to_base = self.lookup_mat(self.get_parameter("odom_frame").value, base_frame, stamp)
        if self.profile:
            self._prof_add("lookup_odom_base", perf_counter() - t0)
        if T_odom_to_base is None:
            return

        clouds = [points for _, points in self.scans]
        if current_points is not None:
            clouds.append(current_points)

        if self.profile:
            t0 = perf_counter()
        cloud = np.concatenate(clouds)
        if self.profile:
            self._prof_add("concatenate", perf_counter() - t0)

        if self.profile:
            t0 = perf_counter()
        cloud = self.transform_points(self.inverse_se3(T_odom_to_base), cloud)
        if self.profile:
            self._prof_add("transform_full", perf_counter() - t0)

        if self.profile:
            t0 = perf_counter()
        cloud = self._output_filter(cloud)
        if self.profile:
            self._prof_add("voxel_filter", perf_counter() - t0)
            self._prof_last_out_points = len(cloud)

        header = Header()
        header.stamp = stamp
        header.frame_id = base_frame

        if self.profile:
            t0 = perf_counter()
        cloud_msg = pc2.create_cloud(header, _FIELDS_XYZI, cloud)
        if self.profile:
            self._prof_add("create_cloud", perf_counter() - t0)
            t0 = perf_counter()
        self.pub_accumulated.publish(cloud_msg)
        if self.profile:
            self._prof_add("publish", perf_counter() - t0)


def main(args=None):
    rclpy.init(args=args)
    node = ScanAccumulator()
    rclpy.spin(node, executor=MultiThreadedExecutor())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
