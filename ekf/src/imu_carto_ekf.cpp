#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("imu_cartographer_ekf") {
        // 초기 상태 벡터 X = [x, y, v_x, v_y, yaw, yaw_rate]
        X.setZero();

        // 초기 상태 공분산 P (초기 불확실성 낮게 설정)
        P.setZero();
        P.diagonal() << 0.01, 0.01, 0.001, 0.001, 0.0005, 0.0001;

        // 상태 전이 행렬 F (고정값 초기화)
        F.setIdentity();

        // 프로세스 노이즈 공분산 Q (고정값 초기화)
        Q.setZero();
        Q(0, 0) = 0.01;      // X (source : Cartographer)
        Q(1, 1) = 0.01;      // Y (source : Cartographer)
        Q(2, 2) = 0.001;     // V_x (source : IMU)
        Q(3, 3) = 0.001;     // V_y (source : IMU)
        Q(4, 4) = 0.0005;    // yaw (source : Cartographer and IMU)
        Q(5, 5) = 0.0001;    // yaw_rate (source : IMU)

        last_imu_time = this->now();
        last_cartographer_time = this->now();

        // ROS 2 Subscribers
        imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10, std::bind(&EKFNode::imu_callback, this, std::placeholders::_1));
        
        cartographer_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&EKFNode::cartographer_callback, this, std::placeholders::_1));

        // ROS 2 Publisher
        fused_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("/ekf_odom", 10);
    }

private:
    Vector<double, 6> X;  // [x, y, v_x, v_y, yaw, yaw_rate]
    Matrix<double, 6, 6> P;  // 상태 공분산
    Matrix<double, 6, 6> F;  // 상태 전이 행렬 (고정값 초기화)
    Matrix<double, 6, 6> Q;  // 프로세스 노이즈 공분산 (고정값 초기화)

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr cartographer_sub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub;

    rclcpp::Time last_imu_time;

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        rclcpp::Time current_time = this->now();
        double dt = (last_imu_time.seconds() > 0) ? (current_time - last_imu_time).seconds() : 0.002;
        last_imu_time = current_time;

        predict(dt, msg->angular_velocity.z, msg->linear_acceleration.x, msg->linear_acceleration.y);
        publish_fused_odom();
    }

    void cartographer_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Vector<double, 3> Z;
        Z << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.orientation.z;

        update(Z);
        publish_fused_odom();
    }

    void predict(double dt, double yaw_rate, double a_x, double a_y) {
        // dt 값만 업데이트 (변수 부분만 수정)
        F(0, 2) = dt;  // x = x + v_x * dt
        F(1, 3) = dt;  // y = y + v_y * dt
        F(4, 5) = dt;  // yaw = yaw + yaw_rate * dt

        // 예측 단계 적용
        X(0) += X(2) * dt;
        X(1) += X(3) * dt;
        X(2) += a_x * dt;
        X(3) += a_y * dt;
        X(4) += yaw_rate * dt;
        X(5) = yaw_rate;

        // 공분산 업데이트: P = F * P * F^T + Q
        P = F * P * F.transpose() + Q;
    }

    void update(const Vector<double, 3>& Z) {
        // 측정 행렬 H (Cartographer의 측정값과 상태 벡터의 관계)
        Matrix<double, 3, 6> H = Matrix<double, 3, 6>::Zero();
        H(0, 0) = 1; // x 측정
        H(1, 1) = 1; // y 측정
        H(2, 4) = 1; // yaw 측정

        // 측정 오차: Y = Z - H * X
        Vector<double, 3> Y = Z - H * X;

        // 측정 노이즈 공분산 R
        Matrix<double, 3, 3> R;
        R.diagonal() << 0.05, 0.05, 0.02;

        // 칼만 이득 K = P * H^T * (H * P * H^T + R)^(-1)
        Matrix<double, 3, 3> S = H * P * H.transpose() + R;
        Matrix<double, 6, 3> K = P * H.transpose() * S.inverse();

        // 상태 업데이트: X = X + K * Y
        X = X + K * Y;

        // 공분산 업데이트: P = (I - K H) P
        P = (Matrix<double, 6, 6>::Identity() - K * H) * P;
    }

    void publish_fused_odom() {
        auto fused_msg = nav_msgs::msg::Odometry();
        fused_msg.header.stamp = this->now();
        fused_msg.header.frame_id = "odom";

        fused_msg.pose.pose.position.x = X(0);
        fused_msg.pose.pose.position.y = X(1);
        fused_msg.pose.pose.orientation.z = X(4);
        fused_msg.twist.twist.linear.x = X(2);
        fused_msg.twist.twist.linear.y = X(3);
        fused_msg.twist.twist.angular.z = X(5);

        fused_odom_pub->publish(fused_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}
