#!/usr/bin/env python3

import copy
import time

import open3d as o3d
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
import tf2_ros
import tf_transformations


_FIELDS_XYZ = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_FIELDS_XYZI = _FIELDS_XYZ + [
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


class ScanMatcher(Node):
    """Registers body-frame lidar scans against a prior map with ICP.

    Consumes the global estimator's latest map->base_link estimate as the
    registration prior and publishes the refined map->base_link pose as a
    "fix" for the global estimator to fuse. Deliberately does nothing else:
    no odometry consumption, no tf broadcasting, no initialization logic.
    """

    def __init__(self):
        super().__init__("scan_matcher")
        self.global_map = None
        # Latest map->base_link estimate from the global estimator, used as
        # the ICP prior. None until the first estimate arrives.
        self.T_map_to_base_prior = None
        self.cur_scan = None
        self.cur_scan_stamp = None
        # Static base_link <- scan frame extrinsic, cached after first lookup.
        self.T_base_to_scan_frame = None

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_voxel_size", 0.4),
                ("scan_voxel_size", 0.1),
                ("freq_localization", 0.5),
                ("localization_threshold", 0.8),
                ("fov", 6.28319),
                ("fov_far", 300),
                ("pcd_map_path", ""),
                ("base_frame", "base_link"),
                ("fix_position_stddev", 0.1),
                ("fix_orientation_stddev", 0.05),
            ],
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pc_in_map = self.create_publisher(PointCloud2, "/cur_scan_in_map", 10)
        self.pub_submap = self.create_publisher(PointCloud2, "/submap", 10)
        self.pub_fix = self.create_publisher(Odometry, "/scan_matcher/fix", 10)

        self.get_logger().info("Loading global map...")
        self.initialize_global_map()

        self.create_subscription(PointCloud2, "/cloud_registered_body", self.cb_save_cur_scan, 10)
        self.create_subscription(Odometry, "/odometry/global", self.cb_save_prior, 10)

        self.timer_localisation = self.create_timer(1.0 / self.get_parameter("freq_localization").value, self.localisation_timer_callback)

    def pose_to_mat(self, pose):
        trans = np.eye(4)
        trans[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return trans

    def msg_to_array(self, pc_msg):
        return pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"))

    def registration_at_scale(self, scan, map, initial, scale):
        result_icp = o3d.pipelines.registration.registration_icp(
            self.voxel_down_sample(scan, self.get_parameter("scan_voxel_size").value * scale),
            self.voxel_down_sample(map, self.get_parameter("map_voxel_size").value * scale),
            1.0 * scale,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        return result_icp.transformation, result_icp.fitness

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        # R
        trans_inverse[:3, :3] = trans[:3, :3].T
        # t
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def publish_point_cloud(self, publisher, header, pc):
        pc = np.asarray(pc, dtype=np.float32)
        fields = _FIELDS_XYZI if pc.shape[1] == 4 else _FIELDS_XYZ
        msg = pc2.create_cloud(header, fields, pc[:, : len(fields)])
        publisher.publish(msg)

    def transform_points(self, transform, points):
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return np.matmul(transform, homogeneous.T).T[:, :3]

    def crop_global_map_in_FOV(self, T_map_to_base, scan_stamp):
        T_base_to_map = self.inverse_se3(T_map_to_base)

        global_map_in_map = np.array(self.global_map.points)
        global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
        global_map_in_base_link = np.matmul(T_base_to_map, global_map_in_map.T).T

        if self.get_parameter("fov").value > 3.14:
            indices = np.where(
                (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        else:
            indices = np.where(
                (global_map_in_base_link[:, 0] > 0)
                & (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        global_map_in_FOV = o3d.geometry.PointCloud()
        global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

        header = Header()
        header.stamp = scan_stamp
        header.frame_id = "map"
        self.publish_point_cloud(self.pub_submap, header, np.array(global_map_in_FOV.points)[::10])

        return global_map_in_FOV

    def global_localization(self):
        t_start = self.get_clock().now()
        scan_tobe_mapped = copy.copy(self.cur_scan)
        scan_stamp = self.cur_scan_stamp
        T_prior = self.T_map_to_base_prior
        n_scan_pts = len(scan_tobe_mapped.points)

        t0 = time.perf_counter()
        global_map_in_FOV = self.crop_global_map_in_FOV(T_prior, scan_stamp)
        crop_ms = (time.perf_counter() - t0) * 1e3

        # Coarse-to-fine: a first registration at a large scale to pull in a
        # rough prior, then a second at full resolution to refine it.
        t0 = time.perf_counter()
        transformation, _ = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=T_prior, scale=5)
        icp5_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        transformation, fitness = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1)
        icp1_ms = (time.perf_counter() - t0) * 1e3

        if fitness > self.get_parameter("localization_threshold").value:
            self.publish_fix(transformation, scan_stamp)

            now = self.get_clock().now()
            icp_ms = int((now - t_start).nanoseconds) / 1e6
            scan_age_ms = (now - rclpy.time.Time.from_msg(scan_stamp)).nanoseconds / 1e6
            self.get_logger().info(
                f"fix published: fitness={fitness:.3f} total={icp_ms:.0f}ms "
                f"(crop={crop_ms:.0f}ms icp_scale5={icp5_ms:.0f}ms icp_scale1={icp1_ms:.0f}ms) "
                f"scan_pts={n_scan_pts} scan_age={scan_age_ms:.0f}ms"
            )

            # Debug: current scan transformed into the map frame with the
            # refined pose; should visually align with the map in RViz.
            header = Header()
            header.stamp = scan_stamp
            header.frame_id = "map"
            scan_in_map = self.transform_points(transformation, np.asarray(scan_tobe_mapped.points))
            self.publish_point_cloud(self.pub_pc_in_map, header, scan_in_map)
        else:
            self.get_logger().warn(f"Fitness score {fitness} less than localization threshold {self.get_parameter('localization_threshold').value}")

    def voxel_down_sample(self, pcd, voxel_size):
        return pcd.voxel_down_sample(voxel_size)

    def cb_save_prior(self, msg):
        self.T_map_to_base_prior = self.pose_to_mat(msg.pose.pose)

    def lookup_scan_extrinsic(self, scan_frame_id):
        # base_link <- scan frame is a static mount extrinsic; look it up once.
        if self.T_base_to_scan_frame is None:
            base_frame = self.get_parameter("base_frame").value
            try:
                tf_msg = self.tf_buffer.lookup_transform(base_frame, scan_frame_id, rclpy.time.Time())
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"No {base_frame} <- {scan_frame_id} tf yet; dropping scan: {e}")
                return None
            trans = tf_msg.transform.translation
            rot = tf_msg.transform.rotation
            T = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
            T[:3, 3] = [trans.x, trans.y, trans.z]
            self.T_base_to_scan_frame = T
        return self.T_base_to_scan_frame

    def cb_save_cur_scan(self, msg):
        T_base_to_scan = self.lookup_scan_extrinsic(msg.header.frame_id)
        if T_base_to_scan is None:
            return
        t0 = time.perf_counter()
        pc = self.transform_points(T_base_to_scan, self.msg_to_array(msg))
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(pc)
        dt_ms = (time.perf_counter() - t0) * 1e3
        # This callback runs once per incoming scan (subscription rate), but
        # only the most recent conversion is ever used by the ~0.5Hz
        # localization timer -- so its steady-state CPU cost is this
        # per-call time times the *subscription* rate, not the timer rate.
        # Throttled log surfaces that steady-state cost.
        self.get_logger().debug(
            f"cb_save_cur_scan: {len(pc)} pts, convert={dt_ms:.1f}ms",
            throttle_duration_sec=2.0,
        )
        self.cur_scan = scan
        self.cur_scan_stamp = msg.header.stamp

    def initialize_global_map(self):
        map_path = self.get_parameter("pcd_map_path").value
        self.global_map = o3d.io.read_point_cloud(map_path)
        if len(self.global_map.points) == 0:
            raise RuntimeError(f"Global map is empty or could not be read: '{map_path}'")
        self.global_map = self.voxel_down_sample(self.global_map, self.get_parameter("map_voxel_size").value)
        self.get_logger().info(f"Global map loaded: {len(self.global_map.points)} points after downsampling.")

    def publish_fix(self, transform, stamp):
        fix = Odometry()
        xyz = transform[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(transform)
        fix.pose.pose.position.x = xyz[0]
        fix.pose.pose.position.y = xyz[1]
        fix.pose.pose.position.z = xyz[2]
        fix.pose.pose.orientation.x = quat[0]
        fix.pose.pose.orientation.y = quat[1]
        fix.pose.pose.orientation.z = quat[2]
        fix.pose.pose.orientation.w = quat[3]

        pos_var = self.get_parameter("fix_position_stddev").value ** 2
        ori_var = self.get_parameter("fix_orientation_stddev").value ** 2
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = np.eye(3) * pos_var
        covariance[3:, 3:] = np.eye(3) * ori_var
        fix.pose.covariance = covariance.flatten()

        fix.header.stamp = stamp
        fix.header.frame_id = "map"
        fix.child_frame_id = self.get_parameter("base_frame").value
        self.pub_fix.publish(fix)

    def localisation_timer_callback(self):
        if self.T_map_to_base_prior is None:
            self.get_logger().info("Waiting for prior from global estimator...", throttle_duration_sec=5.0)
            return
        if self.cur_scan is None:
            self.get_logger().info("Waiting for first scan...", throttle_duration_sec=5.0)
            return
        self.global_localization()


def main(args=None):
    rclpy.init(args=args)
    node = ScanMatcher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
