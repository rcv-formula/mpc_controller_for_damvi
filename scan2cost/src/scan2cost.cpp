#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <cmath>
#include <geometry_msgs/msg/transform_stamped.hpp>

class Scan2CostNode : public rclcpp::Node
{
public:
    Scan2CostNode()
        : Node("scan2cost")
    {
        // Declare parameters
        input_topic_ = this->declare_parameter<std::string>("input_topic", "/scan");
        output_topic_ = this->declare_parameter<std::string>("output_topic", "/costmap");
        frame_id_ = this->declare_parameter<std::string>("frame_id", "base_link");
        resolution_ = this->declare_parameter<double>("resolution", 0.1);
        width_ = this->declare_parameter<int>("width", 100);
        height_ = this->declare_parameter<int>("height", 100);
        origin_x_ = this->declare_parameter<double>("origin_x", 0.0);
        origin_y_ = this->declare_parameter<double>("origin_y", 0.0);

        // Load static transform parameters
        static_transform_.header.frame_id = this->declare_parameter<std::string>("static_transform.frame_id", "base_link");
        static_transform_.child_frame_id = this->declare_parameter<std::string>("static_transform.child_frame_id", "laser");
        static_transform_.transform.translation.x = this->declare_parameter<double>("static_transform.translation.x", 0.2);
        static_transform_.transform.translation.y = this->declare_parameter<double>("static_transform.translation.y", 0.0);
        static_transform_.transform.translation.z = this->declare_parameter<double>("static_transform.translation.z", 0.0);
        static_transform_.transform.rotation.x = this->declare_parameter<double>("static_transform.rotation.x", 0.0);
        static_transform_.transform.rotation.y = this->declare_parameter<double>("static_transform.rotation.y", 0.0);
        static_transform_.transform.rotation.z = this->declare_parameter<double>("static_transform.rotation.z", 0.0);
        static_transform_.transform.rotation.w = this->declare_parameter<double>("static_transform.rotation.w", 1.0);

        // Topic subscription and publisher setup
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            input_topic_, 10, std::bind(&Scan2CostNode::scanCallback, this, std::placeholders::_1));

        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            output_topic_, 10);

        initializeCostmap();

        // Topic status checker (0.5s cycle)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), std::bind(&Scan2CostNode::checkScanTopic, this));

        RCLCPP_INFO(this->get_logger(), "Scan2CostNode initialized.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    nav_msgs::msg::OccupancyGrid costmap_;
    int width_, height_;
    double resolution_, origin_x_, origin_y_;
    std::string input_topic_, output_topic_, frame_id_;
    bool scan_received_ = false;

    geometry_msgs::msg::TransformStamped static_transform_;

    void initializeCostmap()
    {
        costmap_.header.frame_id = frame_id_;
        costmap_.info.resolution = resolution_;
        costmap_.info.width = width_;
        costmap_.info.height = height_;
        costmap_.info.origin.position.x = origin_x_;
        costmap_.info.origin.position.y = origin_y_;
        costmap_.info.origin.position.z = 0.0;
        costmap_.info.origin.orientation.w = 1.0;

        costmap_.data.resize(width_ * height_, -1);
    }

    // /scan topic status check (timer)
    void checkScanTopic()
    {
        if (!scan_received_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for input topic: %s", input_topic_.c_str());
        }
        scan_received_ = false; // Reset for next check
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        scan_received_ = true; // Topic received check flag
        auto map = costmap_;
        std::fill(map.data.begin(), map.data.end(), 0);

        double angle = scan->angle_min;
        for (size_t i = 0; i < scan->ranges.size(); ++i)
        {
            double range = scan->ranges[i];
            if (range >= scan->range_min && range <= scan->range_max)
            {
                // Transform point using the static transform
                double lx = range * cos(angle);
                double ly = range * sin(angle);

                // Apply the static offset from laser_frame to base_link
                double bx = lx + static_transform_.transform.translation.x;
                double by = ly + static_transform_.transform.translation.y;

                int x = static_cast<int>((bx - origin_x_) / resolution_);
                int y = static_cast<int>((by - origin_y_) / resolution_);

                if (x >= 0 && x < width_ && y >= 0 && y < height_)
                {
                    map.data[y * width_ + x] = 100; // Map obstacles
                }
            }
            angle += scan->angle_increment;
        }

        map.header.stamp = this->get_clock()->now();
        map.header.frame_id = frame_id_;
        map_pub_->publish(map);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Scan2CostNode>());
    rclcpp::shutdown();
    return 0;
}
