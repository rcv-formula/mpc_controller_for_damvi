import yaml, math, os
import torch
import numpy as np
import rclpy
from rclpy.node import Node

class mapParser:
    def __init__(self, path_file_dir, node):
        self.path_file_dir = path_file_dir
        self.node = Node(node)

    def set_path_file_dir(self, path_file_dir):
        self.path_file_dir = path_file_dir

    def get_path(self):
        waypoints = self.__private_load_path_file()



    def __private_load_path_file(self):
        waypoints = np.loadtxt(self.path_file_dir)
        if waypoints.shape[1] != 3:
            self.node.get_logger().warn("Error, path file(CSV) should have 3 cloumns(x, y, v)")
            return[]
        self.node.get_logger().info(f"Loaded {len(waypoints)} way points from {self.path_file_dir}")
        return waypoints

