import numpy as np
import math

class PathProcessor:
    def __init__(self, mpc_controller):
        params = mpc_controller.get_parameters() # Main Class에서 변수 받기 (상속 X)
        self.NX = params["NX"]
        self.T = params["T"]
        self.DT = params["DT"]
        self.N_IND_SEARCH = params["N_IND_SEARCH"]
        self.laps = params["laps"]
        
        self.g_num_points = None       # Number of points in original Global Path
        self.l_num_points = None       # Number of points in local path
        self.global_path_np = None     # [x, y, v, sin(yaw), cos(yaw), yaw]
        self.local_path_np = None      # [x, y, v, sin(yaw), cos(yaw), yaw]

    def process_global_path(self, way_points):   # input [x, y, v] output [x, y, v, sin(yaw), cos(yaw), yaw]
        
        self.g_num_points = len(way_points)
        self.global_path_np = np.tile(way_points[:-1], (self.laps, 1))

        num_points = self.global_path_np.shape[0]
        path_points = np.zeros((num_points, self.NX+1)) # [x, y, v, sin(yaw), cos(yaw), yaw]

        path_points[:, :3] = self.global_path_np[:, :3] # [x, y, v]
        dx = np.diff(self.global_path_np[:, 0], append=self.global_path_np[-1, 0])
        dy = np.diff(self.global_path_np[:, 1], append=self.global_path_np[-1, 1])
        path_points[:, 5] = np.arctan2(dy, dx)         # yaw
        path_points[:, 3] = np.sin(path_points[:, 5])  # sin(yaw)
        path_points[:, 4] = np.cos(path_points[:, 5])  # cos(yaw)
        path_points[-1, 3:] = path_points[-2, 3:] if num_points > 1 else 0.0  # 마지막 점 보정
        
        self.global_path_np = path_points

    def process_local_path(self, way_points):  # input [x, y, v, yaw] output [x, y, v, sin(yaw), cos(yaw), yaw]

        self.l_num_points = len(way_points)
        path_points = np.zeros((self.l_num_points, self.NX+1)) # [x, y, v, sin(yaw), cos(yaw), yaw]

        path_points[:, :3] = way_points[:, :3]        # [x, y, v]
        dx = np.diff(self.global_path_np[:, 0], append=self.global_path_np[-1, 0])
        dy = np.diff(self.global_path_np[:, 1], append=self.global_path_np[-1, 1])
        path_points[:, 3] = np.sin(way_points[:, 3])  # [sin(yaw)]
        path_points[:, 4] = np.cos(way_points[:, 3])  # [cos(yaw)]
        path_points[:, 5] = way_points[:, 3]          # [yaw]

        self.local_path_np = path_points

    def calc_nearest_index(self, state, p_g_ind, p_l_ind): # input self.state, previous index(global), previous index(local)
        
        if p_g_ind is None: # which mean it is on init phase -> needs to do global search
            dx = state[0] - self.global_path_np[:self.g_num_points,0]
            dy = state[1] - self.global_path_np[:self.g_num_points,1]
            p_g_ind = 0
        else :            # find nearest index within next N_IND_SEARCH, starting from previous index
            dx = state[0] - self.global_path_np[p_g_ind:(p_g_ind + self.N_IND_SEARCH),0]
            dy = state[1] - self.global_path_np[p_g_ind:(p_g_ind + self.N_IND_SEARCH),1]
        
        d = np.sqrt(dx**2 + dy**2)
        g_ind = np.argmin(d)
        mind = math.sqrt(d[g_ind])
        g_ind = np.argmin(d) + p_g_ind

        dxl = self.global_path_np[g_ind,0] - state[0]
        dyl = self.global_path_np[g_ind,1] - state[1]
        d_yaw = self.global_path_np[g_ind,5] - math.atan2(dyl, dxl)
        angle = math.atan2(math.sin(d_yaw), math.cos(d_yaw))
        if angle < 0:
            mind *= -1

        if p_l_ind is not None: # which mean, it is in local path mode
            dx = state[0] - self.local_path_np[:,0]
            dy = state[1] - self.local_path_np[:,1]

            d = np.sqrt(dx**2 + dy**2)
            l_ind = np.argmin(d)
        else :
            l_ind = p_l_ind
            
        return g_ind, mind, l_ind   # g_ind 는 메인 노드에서 : target_index, l_ind 는 메인 노드에서 self.local_flag, mind는 아직 사용처 없음.
    
    def calc_ref_trajectory(self, state, p_g_ind, p_l_ind): # input self.state, prev index(global), prev index(local)이자 main에선 self.p_l_ind
        
        xref = np.zeros((self.NX, self.T + 1))
        g_ind, _, l_ind = self.calc_nearest_index(state, p_g_ind, p_l_ind)

        def xref_model(xref, path, ind):
            ncourse = len(path)
            indices = np.arange(ind, ind + xref.shape[1])  # [ind, ind+1, ..., ind+T]
            indices = np.clip(indices, 0, ncourse - 1)  # 최대 인덱스 초과 시 마지막 값 유지
            xref[:, :] = path[indices, :5].T  # (T+1, 5).T → (5, T+1) 로 변환하여 할당

            return xref

        if p_g_ind >= g_ind:
            g_ind = p_g_ind

        if l_ind is not None:     # local path mode
            xref = xref_model(xref, self.local_path_np, l_ind)
        else :                    # global path mode
            xref = xref_model(xref, self.global_path_np, g_ind)
        
        return xref, g_ind, l_ind  # output xref for nonlinear MPC, prev_index(global)=target_index, prev index(local)
    
    # 메인노드에 local path를 구독 받는 subscriber 1개, local path flag를 구독 받는 subscriber 1개 총 2개가 필요하다.
    # 메인노드에서 flag는 self.p_l_ind 이다. 
    # local path flag callback에서 flag(p_l_ind) 를 None으로 바꿔주어야한다.
    # local path flag를 True(실수)로 바꾸는건, local path callback에서 하자, True(실수)인데 local path가 없으면 문제가 생길 여지가 있다.)