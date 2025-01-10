#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class LaserScanFusion : public rclcpp::Node {
public:
    LaserScanFusion() : Node("laser_scan_fusion") {
        // Subscriptions for both LaserScan topics
        depth_scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan_depth", 10, std::bind(&LaserScanFusion::depthScanCallback, this, std::placeholders::_1));
        lidar_scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&LaserScanFusion::lidarScanCallback, this, std::placeholders::_1));

        // Publisher for the fused LaserScan
        fused_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/fused_scan", 10);

        // Initialize TF buffer and listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    // Callback for the depth camera's projected scan
    void depthScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        depth_scan_ = msg;
        tryFusion();
    }

    // Callback for the LiDAR's scan
    void lidarScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        lidar_scan_ = msg;
        tryFusion();
    }

    // Attempt to fuse the scans if both are available
    void tryFusion() {
        if (!depth_scan_ || !lidar_scan_) {
            return;  // Wait until both scans are received
        }

        // Check for TF alignment
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                lidar_scan_->header.frame_id, depth_scan_->header.frame_id, lidar_scan_->header.stamp,
                rclcpp::Duration::from_seconds(0.1));
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
            return;
        }

        // Start fusion process
        auto fused_scan = std::make_shared<sensor_msgs::msg::LaserScan>();
        fused_scan->header.frame_id = "laser";  // Target frame
        fused_scan->header.stamp = lidar_scan_->header.stamp;  // Use the latest LiDAR scan timestamp
        fused_scan->angle_min = lidar_scan_->angle_min;
        fused_scan->angle_max = lidar_scan_->angle_max;
        fused_scan->angle_increment = lidar_scan_->angle_increment;
        fused_scan->range_min = lidar_scan_->range_min;
        fused_scan->range_max = lidar_scan_->range_max;

        // Initialize ranges with LiDAR scan data
        fused_scan->ranges = lidar_scan_->ranges;

        // Integrate depth scan into fused ranges
        for (size_t i = 0; i < depth_scan_->ranges.size(); ++i) {
            float depth_range = depth_scan_->ranges[i];
            float lidar_range = fused_scan->ranges[i];

            // Merge logic: use the minimum range if both are valid
            if (std::isfinite(depth_range) && depth_range >= fused_scan->range_min && depth_range <= fused_scan->range_max) {
                if (!std::isfinite(lidar_range) || depth_range < lidar_range) {
                    fused_scan->ranges[i] = depth_range;
                }
            }
        }

        // Publish the fused LaserScan
        fused_scan_pub_->publish(*fused_scan);

        // Clear local references
        depth_scan_.reset();
        lidar_scan_.reset();
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr depth_scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr fused_scan_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    sensor_msgs::msg::LaserScan::SharedPtr depth_scan_;
    sensor_msgs::msg::LaserScan::SharedPtr lidar_scan_;
    std::mutex data_mutex_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaserScanFusion>());
    rclcpp::shutdown();
    return 0;
}

