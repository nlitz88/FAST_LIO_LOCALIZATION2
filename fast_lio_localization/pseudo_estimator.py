#!/usr/bin/env python3

import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time


class PseudoEstimator(Node):
    """Maintains map->odom by directly composing FAST-LIO odometry with
    scan-matcher fixes -- no Gaussian state estimation, just transform
    composition. Also publishes the map->base_link pose estimate that the
    scan matcher uses as its ICP prior.

    map->odom is computed the same way for both a scan-matcher fix and an
    /initialpose bootstrap: T_map_odom = T_map_base * inv(T_odom_base). For
    a fix, T_odom_base is looked up from tf at the fix's own timestamp
    (handles the ~250-300ms staleness between when FAST-LIO processed the
    scan and when the fix arrives here). For /initialpose, T_odom_base is
    taken from the next live /odom message instead of a tf lookup, avoiding
    any startup timing race with a message that has no "own timestamp"
    tying it to a past robot pose.

    Deliberately mirrors map_odom_broadcaster.py's composition (see that
    file's docstring for why this avoids the lever-arm effect an EKF's
    independently-filtered attitude produced), plus /initialpose bootstrap
    and the map->base_link prior publisher that node didn't need.
    """

    def __init__(self):
        super().__init__("pseudo_estimator")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tf_lookup_timeout", 0.1)
        self.declare_parameter("rebroadcast_rate", 30.0)
        self.declare_parameter("pose_estimate_topic", "/localization/pose_estimate")

        self.tf_buffer = tf2_ros.Buffer()
        # spin_thread=True: the tf listener must keep servicing /tf on its
        # own thread, otherwise lookup_transform's timeout below deadlocks
        # against this node's single-threaded executor.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.T_map_odom = None
        # Pending map->base_link pose from /initialpose, consumed by the
        # next /odom message.
        self.pending_bootstrap = None

        pose_estimate_topic = self.get_parameter("pose_estimate_topic").value
        self.pub_pose_estimate = self.create_publisher(Odometry, pose_estimate_topic, 10)

        self.create_subscription(Odometry, "/odom", self.cb_odom, 10)
        self.create_subscription(Odometry, "/scan_matcher/fix", self.cb_fix, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.cb_initialpose, 10
        )

        rate = self.get_parameter("rebroadcast_rate").value
        self.create_timer(1.0 / rate, self.rebroadcast)

    def cb_initialpose(self, msg):
        self.pending_bootstrap = self.pose_to_mat(msg.pose.pose)
        self.get_logger().info("Received /initialpose; will bootstrap map->odom on next /odom message.")

    def cb_odom(self, msg):
        T_odom_base = self.pose_to_mat(msg.pose.pose)

        if self.pending_bootstrap is not None:
            self.T_map_odom = self.pending_bootstrap @ np.linalg.inv(T_odom_base)
            self.pending_bootstrap = None
            self.get_logger().info("Bootstrapped map->odom from /initialpose.")

        if self.T_map_odom is None:
            return

        T_map_base = self.T_map_odom @ T_odom_base
        self.publish_pose_estimate(T_map_base, msg.header.stamp)

    def cb_fix(self, msg):
        odom_frame = self.get_parameter("odom_frame").value
        base_frame = self.get_parameter("base_frame").value
        timeout = self.get_parameter("tf_lookup_timeout").value

        stamp = Time.from_msg(msg.header.stamp)
        try:
            odom_to_base = self.tf_buffer.lookup_transform(
                odom_frame, base_frame, stamp, timeout=rclpy.duration.Duration(seconds=timeout)
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f"No time-matched {odom_frame}->{base_frame} tf at fix stamp; skipping this fix: {e}",
                throttle_duration_sec=2.0,
            )
            return

        T_map_base = self.pose_to_mat(msg.pose.pose)
        T_odom_base = self.transform_to_mat(odom_to_base.transform)
        self.T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)

    def publish_pose_estimate(self, T_map_base, stamp):
        base_frame = self.get_parameter("base_frame").value
        map_frame = self.get_parameter("map_frame").value

        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = map_frame
        out.child_frame_id = base_frame

        xyz = T_map_base[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(T_map_base)
        out.pose.pose.position.x = xyz[0]
        out.pose.pose.position.y = xyz[1]
        out.pose.pose.position.z = xyz[2]
        out.pose.pose.orientation.x = quat[0]
        out.pose.pose.orientation.y = quat[1]
        out.pose.pose.orientation.z = quat[2]
        out.pose.pose.orientation.w = quat[3]

        self.pub_pose_estimate.publish(out)

    def rebroadcast(self):
        if self.T_map_odom is None:
            return
        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.get_parameter("map_frame").value
        out.child_frame_id = self.get_parameter("odom_frame").value
        xyz = self.T_map_odom[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.T_map_odom)
        out.transform.translation.x = xyz[0]
        out.transform.translation.y = xyz[1]
        out.transform.translation.z = xyz[2]
        out.transform.rotation.x = quat[0]
        out.transform.rotation.y = quat[1]
        out.transform.rotation.z = quat[2]
        out.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(out)

    def pose_to_mat(self, pose):
        T = tf_transformations.quaternion_matrix(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def transform_to_mat(self, transform):
        t = transform.translation
        r = transform.rotation
        T = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T


def main(args=None):
    rclpy.init(args=args)
    node = PseudoEstimator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
