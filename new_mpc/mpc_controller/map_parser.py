import yaml, math, os
import numpy as np

class mapParser:
    def __init__(self, path_file_dir):
        self.path_file_dir = path_file_dir

    def set_path_file_dir(self, path_file_dir):
        self.path_file_dir = path_file_dir
