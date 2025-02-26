import numpy as np
import math

class PathProcessor:
    def __init__(self, mpc_controller):
        params = mpc_controller.get_parameters()  # ✅ 메인 클래스에서 변수 가져오기
        self.NX = params["NX"]
        self.N_IND_SEARCH = params["N_IND_SEARCH"]
        self.laps = params["laps"]
        self.point_count = None       # Number of points in original Global Path
        self.global_path_np = None    # [x, y, v, sin(yaw), cos(yaw), yaw]
        self.local_path_np = None     # [x, y, v, sin(yaw), cos(yaw), yaw]

    def calc_global_path(self, way_points):   # input [x, y, v] output [x, y, v, sin(yaw), cos(yaw)]
        
        self.point_count = len(way_points)
        self.global_path_np = np.tile(way_points[:-1], (self.laps, 1))

        num_points = self.global_path_np.shape[0]
        path_points = np.zeros((num_points, self.NX+1)) # [x, y, v, sin(yaw), cos(yaw), yaw]

        for i in range(num_points-1):
            # Current and next points
            current_point = self.global_path_np[i]
            next_point = self.global_path_np[i+1]

            # Store x and y
            path_points[i,0] = current_point[0]
            path_points[i,1] = current_point[1]
            path_points[i,2] = current_point[2] # reference velocity

            # Calculate reference yaw (yaw)
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            path_points[i,5] = math.atan2(dy, dx)
            path_points[i,3] = math.sin(path_points[i,5])
            path_points[i,4] = math.cos(path_points[i,5])

        # Last point, dx dy can not be calculated
        path_points[-1,0] = self.global_path_np[-1,0]
        path_points[-1,1] = self.global_path_np[-1,1]
        path_points[-1,2] = self.global_path_np[-1,2] # reference velocity
        path_points[-1,3] = path_points[-2,3]  # reference sin(yaw)
        path_points[-1,4] = path_points[-2,4]  # reference cos(yaw)
        path_points[-1,5] = path_points[-2,5] if num_points > 1 else 0.0 # reference yaw (optional)
        # Path Point
        self.global_path_np = path_points
        # 가독성을 위해 분산
        self.cx, self.cy, self.sp, self.sin_yaw, self.cos_yaw, self.cyaw = path_points.T

    def calc_local_path(self):  # input [x, y, v, yaw] output [x, y, v, sin(yaw), cos(yaw)]
        pass

    def calc_nearest_index(self, state, p_ind, flag): # input self.state from main node, previous index, local_path_flag
        
        if p_ind is None: # which means it is on init phase -> needs to do global search
            dx = state[0] - self.global_path_np[:self.point_count,0]
            dy = state[1] - self.global_path_np[:self.point_count,1]
            p_ind = 0
        else :
            dx = [state[0] - icx for icx in self.cx[p_ind:(p_ind + self.N_IND_SEARCH)]]
            dy = [state[1] - icy for icy in self.cy[p_ind:(p_ind + self.N_IND_SEARCH)]]
        
        d = np.sqrt(dx**2 + dy**2)

        ind = np.argmin(d) + p_ind
        mind = math.sqrt(d[ind])

        dxl = self.cx[ind] - state[0]
        dyl = self.cy[ind] - state[1]

        d_yaw = self.cyaw[ind] - math.atan2(dyl, dxl)
        angle = math.atan2(math.sin(d_yaw), math.cos(d_yaw))
        if angle < 0:
            mind *= -1

        return ind, mind