#ifndef FAST_LIO_LOCALIZATION__SCAN_ACCUMULATOR_COMPONENT_HPP_
#define FAST_LIO_LOCALIZATION__SCAN_ACCUMULATOR_COMPONENT_HPP_

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace fast_lio_localization
{

// C++ reimplementation of scan_accumulator.py (see
// docs/2026-07-scan-accumulator-submap.md and issue #64). Behavior is a
// direct port: same keyframe-gating logic (only publish/parse on a new
// keyframe), no submap voxelization.
//
// Executor note: unlike the rclpy version, this does NOT need a
// MultiThreadedExecutor + dedicated callback group for the scan
// subscription. tf2_ros::TransformListener's spin_thread=true actually
// works as documented in C++ (its own internal executor thread services
// /tf and /tf_static independently of this node's executor) -- the rclpy
// binding's spin_thread=True does not have this property (see the
// scan_accumulator.py docstring / design doc), which is why the Python
// version needed the callback-group workaround. A plain
// SingleThreadedExecutor is sufficient here.
class ScanAccumulatorComponent : public rclcpp::Node
{
public:
  explicit ScanAccumulatorComponent(const rclcpp::NodeOptions & options);

private:
  struct StageStat
  {
    double total_s = 0.0;
    double max_s = 0.0;
    int count = 0;
  };

  void cbScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void publishAccumulated(const rclcpp::Time & stamp);

  // target <- source at stamp, or false if unavailable.
  bool lookupMat(
    const std::string & target_frame, const std::string & source_frame,
    const rclcpp::Time & stamp, Eigen::Isometry3d & out) const;

  bool isKeyframe(const Eigen::Isometry3d & T_odom_to_scan) const;

  void profAdd(const std::string & stage, double dt_s);
  void profMaybeLog(int in_points, int out_points);

  // Parameters
  int n_scans_;
  double keyframe_dist_m_;
  double keyframe_rot_deg_;
  std::string odom_frame_;
  std::string base_frame_;
  double tf_timeout_s_;
  bool profile_timing_;

  // (stamp, keyframe cloud in odom frame)
  std::deque<std::pair<rclcpp::Time, pcl::PointCloud<pcl::PointXYZI>::Ptr>> scans_;
  Eigen::Isometry3d T_last_keyframe_;
  bool have_last_keyframe_ = false;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_accumulated_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_;

  std::map<std::string, StageStat> prof_stats_;
  std::chrono::steady_clock::time_point prof_last_log_;
  int prof_last_in_points_ = 0;
  int prof_last_out_points_ = 0;
};

}  // namespace fast_lio_localization

#endif  // FAST_LIO_LOCALIZATION__SCAN_ACCUMULATOR_COMPONENT_HPP_
