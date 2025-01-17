import yaml, math, os
import torch
import numpy as np
import rclpy, tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_matrix
from ament_index_python.packages import get_package_share_directory


class mapParser:
    def __init__(self, path_file_dir, node : Node):
        self.path_file_dir = path_file_dir
        self.node = node

        if(torch.xpu.is_available()):
            self.accelerrator = "xpu"
        elif (torch.cuda.is_available()):
            self.accelerrator = 'cuda'
        else:
            self.accelerrator = "cpu"
        print("using " + self.accelerrator +" to compute")
        self.__private_load_path_file()

    def set_path_file_dir(self, path_file_dir):
        self.path_file_dir = path_file_dir

    def get_path(self, transform):
        transform_matrix = self.__private_parse_tf_to_transform(transform)
        self.__private_apply_transform_path_cordinate(transform_matrix)


    def reload_path_file(self, new_path_dir):
        self.path_file_dir = new_path_dir
        self.__private_load_path_file()

    def __private_apply_transform_path_cordinate(self, transform_matrix):
        path_point_count = self.raw_waypoints.shape[0]
        homogeneous_path_points = torch.hstack((self.raw_waypoints[:, :2], torch.zeros((path_point_count, 1)).to(self.accelerrator), torch.ones((path_point_count, 1)).to(self.accelerrator) )).to(self.accelerrator)

        #print(homogeneous_path_points)
        path_for_baselink = torch.matmul(homogeneous_path_points, transform_matrix.T)
        #print(path_for_baselink)


    def __private_load_path_file(self):
        waypoints = np.loadtxt(self.path_file_dir,  delimiter=',')
        
        if waypoints.shape[1] != 3:
            self.node.get_logger().warn("Error, path file(CSV) should have 3 cloumns(x, y, v)")
            return[]
        self.raw_waypoints = torch.tensor(waypoints, dtype=torch.float32).to(self.accelerrator)
        self.node.get_logger().info(f"Loaded {len(waypoints)} way points from {self.path_file_dir}")
        return waypoints
    
    def __private_parse_tf_to_transform(self, tf : tf2_ros.TransformStamped):
        translation = tf.transform.translation
        rotation = tf.transform.rotation

        transform_matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32).to(self.accelerrator)
        #print(transform_matrix)
        return transform_matrix

def main(args=None):#
    rclpy.init(args=args)
    node = rclpy.create_node("testNode")
    package_dir = get_package_share_directory('mpc_controller')
    config_path = os.path.join(package_dir, 'map', 'Spielberg_map.csv')
    test_content = mapParser(config_path, node)
    tf = tf2_ros.TransformStamped()
    tf.transform.translation.x = 1.0
    tf.transform.translation.y = 2.0
    tf.transform.translation.z = 5.0

    tf.transform.rotation._w =1.0
    tf.transform.rotation._x =0.0
    tf.transform.rotation._y =0.0
    tf.transform.rotation._z =0.0

    for i in range (6000) :
        test_content.get_path(tf)
    print("done")

    # 노드를 spin하여 실행
    rclpy.spin(node)

    # 노드 종료 시 클린업
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()