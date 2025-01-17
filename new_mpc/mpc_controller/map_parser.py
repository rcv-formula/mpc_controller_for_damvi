import yaml, math, os
import time
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
            self.accelerrator = "xpu:0"
        elif (torch.cuda.is_available()):
            self.accelerrator = 'cuda'
        else:
            self.accelerrator = "cpu"
        print("using " + self.accelerrator +" to compute")
        self.__private_load_path_file()

    def set_path_file_dir(self, path_file_dir):
        self.path_file_dir = path_file_dir


    def get_homogeneous_path(self, transform):
        transform_matrix = self.__private_parse_tf_to_transform(transform)
        return self.__private_apply_transform_path_cordinate(self.waypoints, transform_matrix)
    
    def get_xyvy_path(self, transform):
        transform_matrix = self.__private_parse_tf_to_transform(transform)
        self.__private_apply_transform_path_cordinate(self.waypoints, transform_matrix)



    def xyvy_path_maker(self):
        point_count = self.waypoints.shape[0]
        xyvy_path = torch.hstack(self.waypoints[:, :2], self.raw_waypoints[:,3], )


    def reload_path_file(self, new_path_dir):
        self.path_file_dir = new_path_dir
        self.__private_load_path_file()


    def __private_apply_transform_path_cordinate(self, waypoints, transform_matrix):        #transform_matrix = self.__private_parse_tf_to_transform(transform)
        path_for_baselink = torch.matmul(waypoints, transform_matrix.T)
        return path_for_baselink


    def __private_prepare_homogeneous_path(self, raw_waypoints):
        # 이 함수를 초기화나 로딩 이후 단 한 번만 호출
        path_point_count = raw_waypoints.shape[0]
        zeros_col = torch.zeros((path_point_count, 1), device=raw_waypoints.device)
        ones_col  = torch.ones((path_point_count, 1), device=raw_waypoints.device)

        # x, y만 추출하여 z=0, w=1을 합치기
        # (N, 4) shape
        homogeneous_path_points = torch.hstack((
            raw_waypoints[:, :2],
            zeros_col,
            ones_col
        ))
        return homogeneous_path_points


    def __private_load_path_file(self):
        waypoints = np.loadtxt(self.path_file_dir,  delimiter=',')
        if waypoints.shape[1] != 3:
            self.node.get_logger().warn("Error, path file(CSV) should have 3 cloumns(x, y, v)")
            return []
        self.node.get_logger().info(f"Loaded {len(waypoints)} way points from {self.path_file_dir}")

        self.raw_waypoints = torch.tensor(waypoints, dtype=torch.float32, device=torch.device(self.accelerrator))
        
        self.waypoints = self.__private_prepare_homogeneous_path(self.raw_waypoints)
        
        return self.waypoints
    
    def __private_parse_tf_to_transform(self, tf : tf2_ros.TransformStamped):
        translation = tf.transform.translation
        rotation = tf.transform.rotation

        transform_matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32).to(self.accelerrator)
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


    start = time.time()
    test_content = mapParser("/home/shin/Desktop/mpc_ws/src/mpc_controller_for_damvi/new_mpc/map/Spielberg_map.csv", node, tf)
    for i in range (500) :
        test_content.get_homogeneous_path(tf)
    end = time.time()
    print("done, "+ str(end-start) +" sec")

    # 노드를 spin하여 실행
    rclpy.spin(node)

    # 노드 종료 시 클린업
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()