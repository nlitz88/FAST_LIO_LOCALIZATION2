#include "fast_lio_localization/scan_accumulator_component.hpp"

#include <algorithm>
#include <cmath>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/exceptions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace fast_lio_localization
{

ScanAccumulatorComponent::ScanAccumulatorComponent(const rclcpp::NodeOptions & options)
: rclcpp::Node("scan_accumulator", options)
{
  n_scans_ = this->declare_parameter<int>("n_scans", 10);
  keyframe_dist_m_ = this->declare_parameter<double>("keyframe_dist_m", 1.0);
  keyframe_rot_deg_ = this->declare_parameter<double>("keyframe_rot_deg", 30.0);
  odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
  tf_timeout_s_ = this->declare_parameter<double>("tf_timeout", 0.1);
  profile_timing_ = this->declare_parameter<bool>("profile_timing", false);

  prof_last_log_ = std::chrono::steady_clock::now();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // spin_thread=true: the listener services /tf and /tf_static on its own
  // internal executor thread, independent of this node's executor -- see
  // the header comment. This is what lets cbScan block on lookupTransform
  // without needing a MultiThreadedExecutor.
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, true);

  pub_accumulated_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/accumulated_scan", 10);
  sub_scan_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/cloud_registered_body", 10,
    std::bind(&ScanAccumulatorComponent::cbScan, this, std::placeholders::_1));
}

bool ScanAccumulatorComponent::lookupMat(
  const std::string & target_frame, const std::string & source_frame,
  const rclcpp::Time & stamp, Eigen::Isometry3d & out) const
{
  try {
    const auto tf_msg = tf_buffer_->lookupTransform(
      target_frame, source_frame, stamp,
      rclcpp::Duration::from_seconds(tf_timeout_s_));
    out = tf2::transformToEigen(tf_msg);
    return true;
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "No %s <- %s tf at scan stamp; dropping: %s", target_frame.c_str(),
      source_frame.c_str(), e.what());
    return false;
  }
}

bool ScanAccumulatorComponent::isKeyframe(const Eigen::Isometry3d & T_odom_to_scan) const
{
  if (!have_last_keyframe_) {
    return true;
  }
  const Eigen::Isometry3d T_rel = T_last_keyframe_.inverse() * T_odom_to_scan;
  if (T_rel.translation().norm() >= keyframe_dist_m_) {
    return true;
  }
  const double angle_deg = Eigen::AngleAxisd(T_rel.rotation()).angle() * 180.0 / M_PI;
  return angle_deg >= keyframe_rot_deg_;
}

void ScanAccumulatorComponent::profAdd(const std::string & stage, double dt_s)
{
  auto & stat = prof_stats_[stage];
  stat.total_s += dt_s;
  stat.max_s = std::max(stat.max_s, dt_s);
  stat.count += 1;
}

void ScanAccumulatorComponent::profMaybeLog(int in_points, int out_points)
{
  prof_last_in_points_ = in_points;
  prof_last_out_points_ = out_points;
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - prof_last_log_).count();
  if (elapsed < 5.0) {
    return;
  }
  std::string lines;
  for (const auto & [stage, stat] : prof_stats_) {
    char buf[256];
    std::snprintf(
      buf, sizeof(buf), "  %s: mean=%.2fms max=%.2fms n=%d\n", stage.c_str(),
      1000.0 * stat.total_s / stat.count, 1000.0 * stat.max_s, stat.count);
    lines += buf;
  }
  RCLCPP_INFO(
    this->get_logger(),
    "scan_accumulator profile (last %.1fs, points_in=%d, points_out=%d):\n%s", elapsed,
    prof_last_in_points_, prof_last_out_points_, lines.c_str());
  prof_stats_.clear();
  prof_last_log_ = now;
}

void ScanAccumulatorComponent::cbScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  const auto t_cb_start = std::chrono::steady_clock::now();
  const rclcpp::Time stamp(msg->header.stamp);

  // A backward time jump (bag loop/restart) invalidates the buffer.
  if (!scans_.empty()) {
    if (stamp < scans_.back().first - rclcpp::Duration::from_seconds(1.0)) {
      RCLCPP_WARN(this->get_logger(), "Scan stamp jumped backwards; clearing accumulated scans.");
      scans_.clear();
      have_last_keyframe_ = false;
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  Eigen::Isometry3d T_odom_to_scan;
  const bool have_tf = lookupMat(odom_frame_, msg->header.frame_id, stamp, T_odom_to_scan);
  if (profile_timing_) {
    profAdd("lookup_odom_scan", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (!have_tf) {
    return;
  }

  // Keyframe check only depends on T_odom_to_scan, so it runs before the
  // expensive parse/transform below -- a non-keyframe scan is dropped here
  // with none of that work done, and no publish.
  t0 = std::chrono::steady_clock::now();
  const bool is_kf = isKeyframe(T_odom_to_scan);
  if (profile_timing_) {
    profAdd("keyframe_check", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (!is_kf) {
    if (profile_timing_) {
      profAdd("callback_total", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cb_start).count());
      profMaybeLog(prof_last_in_points_, prof_last_out_points_);
    }
    return;
  }

  t0 = std::chrono::steady_clock::now();
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  pcl::fromROSMsg(*msg, *cloud);
  if (profile_timing_) {
    profAdd("read_points", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (cloud->empty()) {
    return;
  }

  t0 = std::chrono::steady_clock::now();
  auto transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  pcl::transformPointCloud(*cloud, *transformed, T_odom_to_scan.matrix().cast<float>());
  if (profile_timing_) {
    profAdd("transform_scan", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  scans_.emplace_back(stamp, transformed);
  while (static_cast<int>(scans_.size()) > n_scans_) {
    scans_.pop_front();
  }
  T_last_keyframe_ = T_odom_to_scan;
  have_last_keyframe_ = true;
  RCLCPP_DEBUG(this->get_logger(), "Keyframe added (%zu in buffer).", scans_.size());
  publishAccumulated(stamp);

  if (profile_timing_) {
    profAdd("callback_total", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cb_start).count());
    profMaybeLog(prof_last_in_points_, prof_last_out_points_);
  }
}

void ScanAccumulatorComponent::publishAccumulated(const rclcpp::Time & stamp)
{
  auto t0 = std::chrono::steady_clock::now();
  Eigen::Isometry3d T_odom_to_base;
  const bool have_tf = lookupMat(odom_frame_, base_frame_, stamp, T_odom_to_base);
  if (profile_timing_) {
    profAdd("lookup_odom_base", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (!have_tf) {
    return;
  }

  t0 = std::chrono::steady_clock::now();
  pcl::PointCloud<pcl::PointXYZI> cloud;
  for (const auto & [kf_stamp, kf_cloud] : scans_) {
    (void)kf_stamp;
    cloud += *kf_cloud;
  }
  if (profile_timing_) {
    profAdd("concatenate", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  t0 = std::chrono::steady_clock::now();
  pcl::PointCloud<pcl::PointXYZI> cloud_in_base;
  pcl::transformPointCloud(cloud, cloud_in_base, T_odom_to_base.inverse().matrix().cast<float>());
  if (profile_timing_) {
    profAdd("transform_full", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  t0 = std::chrono::steady_clock::now();
  sensor_msgs::msg::PointCloud2 out_msg;
  pcl::toROSMsg(cloud_in_base, out_msg);
  out_msg.header.stamp = stamp;
  out_msg.header.frame_id = base_frame_;
  if (profile_timing_) {
    profAdd("create_cloud", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    t0 = std::chrono::steady_clock::now();
  }
  pub_accumulated_->publish(out_msg);
  if (profile_timing_) {
    profAdd("publish", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    prof_last_out_points_ = static_cast<int>(cloud_in_base.size());
  }
}

}  // namespace fast_lio_localization

RCLCPP_COMPONENTS_REGISTER_NODE(fast_lio_localization::ScanAccumulatorComponent)
