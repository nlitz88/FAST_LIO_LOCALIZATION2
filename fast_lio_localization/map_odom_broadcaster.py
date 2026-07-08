#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf_transformations
import numpy as np


class MapOdomBroadcaster(Node):
    """Broadcasts map->odom, holding it constant between scan-matcher fixes.

    map->odom is computed by composing scan_matcher's raw map->base_link fix
    (already anchored at the scan's own timestamp) with a time-matched
    odom->base_link tf lookup at that same instant, then re-broadcast at a
    steady rate with a fresh timestamp so tf doesn't go stale between fixes.

    This intentionally does NOT source from /odometry/global (the fused
    estimate). Composing THAT against a time-matched odom->base_link tf was
    the first design here, and the timing was correct (verified: >99% exact
    tf matches) -- but /odometry/global's own attitude (roll/pitch/yaw) is a
    Kalman-smoothed blend that disagrees with FAST-LIO's raw attitude by a
    few tenths of a degree at any given instant, simply because they're
    independently filtered. That's normally negligible, but map->odom's
    translation is that attitude difference applied through a lever arm equal
    to the robot's full distance traveled from the origin -- at 20m out, a
    0.5 degree disagreement alone is ~18cm of spurious translation, matching
    exactly what was observed. Since scan_matcher's raw fix and FAST-LIO's
    tf come from the same "instant, unfiltered" family (no independent
    smoothing to disagree with each other), this avoids the effect entirely,
    at the cost of only updating map->odom at scan_matcher's fix rate rather
    than continuously. See docs/2026-07-global-ekf-integration.md.
    """

    def __init__(self):
        super().__init__("map_odom_broadcaster")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tf_lookup_timeout", 0.1)
        self.declare_parameter("rebroadcast_rate", 30.0)

        self.tf_buffer = tf2_ros.Buffer()
        # spin_thread=True: the tf listener must keep servicing /tf on its own
        # thread, otherwise lookup_transform's timeout below deadlocks against
        # this node's single-threaded executor -- the /tf callback that would
        # populate the buffer can't run while we're blocked waiting on it.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.T_map_odom = None
        self.create_subscription(Odometry, "/scan_matcher/fix", self.cb_fix, 10)
        rate = self.get_parameter("rebroadcast_rate").value
        self.create_timer(1.0 / rate, self.rebroadcast)

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
            self.get_logger().warn(f"No time-matched {odom_frame}->{base_frame} tf at fix stamp; skipping this fix: {e}", throttle_duration_sec=2.0)
            return

        T_map_base = self.pose_to_mat(msg.pose.pose)
        T_odom_base = self.transform_to_mat(odom_to_base.transform)
        self.T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)

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
    node = MapOdomBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
