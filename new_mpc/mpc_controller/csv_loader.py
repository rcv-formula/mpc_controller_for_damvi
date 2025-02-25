from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class csv_path_loader:
    def __init__(self, node :Node):
        self.node = node

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,               # 신뢰성: Reliable
            durability=DurabilityPolicy.TRANSIENT_LOCAL,          # 지속성: Transient Local
            history=HistoryPolicy.KEEP_LAST,                      # 히스토리: Keep Last
            depth=1                                              # Depth 설정
        )
        self.path_recived = False
        self.global_path = list()


        self.subscription = self.node.create_subscription(Path, 'global_path', self.path_callback, qos_profile)
        
        timeout = 5.0  # 최대 5초 대기
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        while not self.path_recived and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            current_time = self.node.get_clock().now().nanoseconds / 1e9
            if current_time - start_time > timeout:
                self.node.get_logger().warn("Timeout waiting for global path!")
                break
        self.node.get_logger().info(f"path has recived, total {len(self.global_path)} points")

    def get_global_path(self):
        return np.array(self.global_path)

    def path_callback(self, msg : Path):
        self.global_path = list()
        for pose in msg.poses:
            self.global_path.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        self.path_recived = True

def main():
    rclpy.init()
    node = Node("cordinate_converter_test_node")
    converter = csv_path_loader(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
