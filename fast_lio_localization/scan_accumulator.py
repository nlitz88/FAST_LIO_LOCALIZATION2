#!/usr/bin/env python3

import collections

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
    """Accumulates the last N deskewed scans from FAST-LIO into a denser
    "virtual scan" for the scan matcher.

    Each incoming scan is transformed into the odom frame using the
    odom -> scan-frame tf at the scan's own timestamp. FAST-LIO's tf is
    the exact state pose it registered the scan with, so this reproduces
    its own world registration with no extra drift. On every scan, the
    ring buffer's contents are concatenated, re-expressed in base_frame
    at the newest scan's timestamp, and published -- so the scan matcher
    is effectively aligning that latest scan, just with a denser cloud.

    Future work:
      - per-scan voxel downsampling (_per_scan_filter) to bound memory/CPU
      - output voxel/range filtering (_output_filter) to shrink messages
      - distance/rotation-traveled gating instead of a count-based buffer
      - decoupled (lower) publish rate if a denser lidar makes per-scan
        publishing expensive
    """

    def __init__(self):
        super().__init__("scan_accumulator")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("n_scans", 10),
                ("odom_frame", "odom"),
                ("base_frame", "base_link"),
                ("tf_timeout", 0.1),
            ],
        )

        # (stamp, Nx4 float32 array of xyz+intensity, already in odom frame)
        self.scans = collections.deque(maxlen=self.get_parameter("n_scans").value)

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

    def _per_scan_filter(self, points):
        # Seam for future per-scan voxel downsampling.
        return points

    def _output_filter(self, points):
        # Seam for future output voxel/range filtering.
        return points

    def cb_scan(self, msg):
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)

        # A backward time jump (bag loop/restart) invalidates the buffer.
        if self.scans:
            newest = rclpy.time.Time.from_msg(self.scans[-1][0])
            if stamp < newest - Duration(seconds=1.0):
                self.get_logger().warn("Scan stamp jumped backwards; clearing accumulated scans.")
                self.scans.clear()

        T_odom_to_scan = self.lookup_mat(
            self.get_parameter("odom_frame").value, msg.header.frame_id, msg.header.stamp
        )
        if T_odom_to_scan is None:
            return

        points = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"))
        if len(points) == 0:
            return
        points = self.transform_points(T_odom_to_scan, points)
        points = self._per_scan_filter(points)
        self.scans.append((msg.header.stamp, points))

        self.publish_accumulated(msg.header.stamp)

    def publish_accumulated(self, stamp):
        base_frame = self.get_parameter("base_frame").value
        T_odom_to_base = self.lookup_mat(self.get_parameter("odom_frame").value, base_frame, stamp)
        if T_odom_to_base is None:
            return

        cloud = np.concatenate([points for _, points in self.scans])
        cloud = self.transform_points(self.inverse_se3(T_odom_to_base), cloud)
        cloud = self._output_filter(cloud)

        header = Header()
        header.stamp = stamp
        header.frame_id = base_frame
        self.pub_accumulated.publish(pc2.create_cloud(header, _FIELDS_XYZI, cloud))


def main(args=None):
    rclpy.init(args=args)
    node = ScanAccumulator()
    rclpy.spin(node, executor=MultiThreadedExecutor())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
