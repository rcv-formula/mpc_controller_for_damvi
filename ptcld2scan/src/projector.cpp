#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

class PointCloudToLaserScan : public rclcpp::Node {
public:
    PointCloudToLaserScan()
        : Node("pointcloud_to_laserscan"),
          tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
          tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_)) {

        // Declare and retrieve parameters from the config file
        this->declare_parameter("azimuth_min", -M_PI);
        this->declare_parameter("azimuth_max", M_PI);
        this->declare_parameter("elevation_min", -M_PI / 4.0);
        this->declare_parameter("elevation_max", M_PI / 4.0);
        this->declare_parameter("range_min", 0.1);
        this->declare_parameter("range_max", 5.0);
        this->declare_parameter("input_topic", "/camera/depth/depth/color/points");
        this->declare_parameter("output_frame", "laser");

        azimuth_min_ = this->get_parameter("azimuth_min").as_double();
        azimuth_max_ = this->get_parameter("azimuth_max").as_double();
        elevation_min_ = this->get_parameter("elevation_min").as_double();
        elevation_max_ = this->get_parameter("elevation_max").as_double();
        range_min_ = this->get_parameter("range_min").as_double();
        range_max_ = this->get_parameter("range_max").as_double();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_frame_ = this->get_parameter("output_frame").as_string();

        // Log ROI and other configuration information at startup
        RCLCPP_INFO(this->get_logger(), "ROI Angles - Azimuth: [%f, %f], Elevation: [%f, %f]",
                    azimuth_min_, azimuth_max_, elevation_min_, elevation_max_);
        RCLCPP_INFO(this->get_logger(), "Range Limits - Min: %f, Max: %f", range_min_, range_max_);
        RCLCPP_INFO(this->get_logger(), "Input Topic: %s, Output Frame: %s", input_topic_.c_str(), output_frame_.c_str());

        // Subscribe to the input PointCloud2 topic
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, 10,
            std::bind(&PointCloudToLaserScan::pointCloudCallback, this, std::placeholders::_1));

        // Publisher for the LaserScan messages
        scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan_depth", 10);

        RCLCPP_INFO(this->get_logger(), "Node initialized and waiting for PointCloud2 data...");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Error handling if no data is received
        if (!msg) {
            RCLCPP_ERROR(this->get_logger(), "No PointCloud2 data received on topic: %s", input_topic_.c_str());
            return;
        }

        // Transform the PointCloud2 into the desired frame (e.g., LiDAR frame)
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(output_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
            return;
        }

        // Convert PointCloud2 to PCL format
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // Filter points based on Angular ROI
        pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
        for (const auto &point : pcl_cloud.points) {
            float azimuth = std::atan2(point.y, point.x);  // Horizontal angle
            float elevation = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));  // Vertical angle

            if (azimuth >= azimuth_min_ && azimuth <= azimuth_max_ &&
                elevation >= elevation_min_ && elevation <= elevation_max_) {
                filtered_cloud.points.push_back(point);
            }
        }

        // Project filtered points to 2D LaserScan format
        auto scan_msg = std::make_shared<sensor_msgs::msg::LaserScan>();
        projectToLaserScan(filtered_cloud, transform, scan_msg);

        // Publish the LaserScan message
        scan_pub_->publish(*scan_msg);
    }

    void projectToLaserScan(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                            const geometry_msgs::msg::TransformStamped &transform,
                            std::shared_ptr<sensor_msgs::msg::LaserScan> scan_msg) {
        // LaserScan parameters
        scan_msg->header.stamp = this->now();
        scan_msg->header.frame_id = output_frame_;  // Use the configured output frame
        scan_msg->angle_min = -M_PI;
        scan_msg->angle_max = M_PI;
        scan_msg->angle_increment = 0.01;  // Adjust based on resolution
        scan_msg->range_min = range_min_;
        scan_msg->range_max = range_max_;
        size_t num_ranges = std::ceil((scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);
        scan_msg->ranges.assign(num_ranges, scan_msg->range_max);

        // Process each point in the filtered cloud
        for (const auto &point : cloud.points) {
            // Transform the point to the desired frame
            geometry_msgs::msg::PointStamped point_in, point_out;
            point_in.point.x = point.x;
            point_in.point.y = point.y;
            point_in.point.z = point.z;
            point_in.header.frame_id = output_frame_;

            try {
                tf2::doTransform(point_in, point_out, transform);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Point transformation failed: %s", ex.what());
                continue;
            }

            // Convert to polar coordinates (angle, range)
            float angle = std::atan2(point_out.point.y, point_out.point.x);
            float range = std::hypot(point_out.point.x, point_out.point.y);

            // Map the angle to the correct index in the LaserScan ranges
            int index = static_cast<int>((angle - scan_msg->angle_min) / scan_msg->angle_increment);
            if (index >= 0 && index < static_cast<int>(num_ranges)) {
                // Update the range if the new range is closer
                if (range >= scan_msg->range_min && range <= scan_msg->range_max && range < scan_msg->ranges[index]) {
                    scan_msg->ranges[index] = range;
                }
            }
        }
    }

    // ROS 2 entities
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Configurable parameters
    double azimuth_min_, azimuth_max_;
    double elevation_min_, elevation_max_;
    double range_min_, range_max_;
    std::string input_topic_;
    std::string output_frame_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToLaserScan>());
    rclcpp::shutdown();
    return 0;
}

