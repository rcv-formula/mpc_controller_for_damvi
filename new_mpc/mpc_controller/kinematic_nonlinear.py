import yaml, time, os, math, threading, rclpy, rclpy.duration
import numpy as np
import casadi as ca
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from mpc_controller.path_processor import PathProcessor

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        #load parameters from YAML file
        self.load_config()

        # State and Control Variables
        self.state = np.zeros(5)     # [x, y, v, sin(yaw), cos(yaw)]
        self.control = np.zeros(2)   # [acceleration, steering_angle]
        self.p_g_ind = None          # init value for calc_nearest_index(), if it is None, it will do global search
        self.p_l_ind = None          # flag and local_path_index, if it is None, global path will be used

        # 불러오기 - 객체 생성
        self.path_processor = PathProcessor(self)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,              # 신뢰성: Reliable
            durability=DurabilityPolicy.TRANSIENT_LOCAL,         # 지속성: Transient Local
            history=HistoryPolicy.KEEP_LAST,                     # 히스토리: Keep Last
            depth=1                                              # Depth 설정
        )
        # Subscriber
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(Float64, '/commands/motor/speed', self.speed_callback, 10)
        self.create_subscription(Path, '/global_path', self.global_path_callback, qos_profile)     # /global_path 입력이 있어야 전체 MPC 시작 가능.

        # Publisher
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.ackm_drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def load_config(self):
        # Load Parameters from YAML file
        package_dir = get_package_share_directory('mpc_controller')
        config_path = os.path.join(package_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.validate_config(config)

        # Vehicle parameters
        car_config = config['car']
        self.WB = car_config['WB']  # [m]        
        self.MAX_STEER = np.deg2rad(car_config['MAX_STEER'])  # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(car_config['MAX_DSTEER'])  # maximum steering speed [rad/s]
        self.MAX_SPEED = car_config['MAX_SPEED']  # maximum speed [m/s]
        self.MIN_SPEED = car_config['MIN_SPEED']  # minimum speed [m/s]
        self.MAX_ACCEL = car_config['MAX_ACCEL']  # maximum accel [m/ss]
        self.ERPM_GAIN = car_config['ERPM_GAIN']

        # MPC parameters
        mpc_config = config['mpc']
        self.NX = mpc_config['NX'] # X = [x, y, v, yaw]
        self.NU = mpc_config['NU'] # a = [accel, steer]
        self.T = mpc_config['T'] # Horizon Lengthspeed'] # Search Index Number
        self.DT = mpc_config['DT']  # [s] time tick
        self.N_IND_SEARCH = mpc_config['N_IND_SEARCH']

        # MPC Weights
        weights_config = config['weights']
        self.Q = np.diag(weights_config['Q'])   # state cost matrix
        self.Qf = np.diag(weights_config['Qf'])
        self.R = np.diag(weights_config['R'])  # input cost matrix
        self.Rd = np.diag(weights_config['Rd'])   # input difference cost matrix

        # Map
        map_config = config['map']
        self.out_of_bounds_penalty = map_config['out_of_bounds_penalty']
        self.global_path_dir = os.path.join(package_dir, 'map', f"{map_config['global_path']}.csv")
        self.laps = map_config['laps']
        
        # Control
        control_config = config['control']
        self.CONTROL_RATE = control_config['CONTROL_RATE']
        self.SIM_MODE = control_config['SIM_MODE']

        # Warm Start
        self.prev_sol_x = None
        self.prev_sol_u = None

    def validate_config(self, config):
        required_keys = ['car', 'mpc', 'weights', 'map']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

    def get_parameters(self): # parameter to send it to path_processor
        return {"NX": self.NX, "T" : self.T, "DT" : self.DT, "N_IND_SEARCH" : self.N_IND_SEARCH, "laps" : self.laps}

    def odom_callback(self, msg):
        # Update robot's state from Odometry, however since we are using Cartographer speed value is not given.
        # Speed value can be obtained by differentiating positions, or it can also be obtained from VESC (wheel encoder)
        # However on Simulator, speed value is provided via odometry message
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _,_, yaw = euler_from_quaternion(orientation_list)

        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        self.state[3] = np.sin(yaw)
        self.state[4] = np.cos(yaw)
        
        if self.SIM_MODE :
            self.state[2] = msg.twist.twist.linear.x

    def speed_callback(self, msg:Float64):
        # Speed value from VESC # Real World
        if not self.SIM_MODE :
            self.state[2] = msg.data / self.ERPM_GAIN

    def global_path_callback(self, msg:Path):

        try:
            path_list = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in msg.poses]
            waypoints = np.array(path_list)

            if waypoints.shape[1] != 3:
                raise ValueError("Global path CSV must have exactly 3 columns: [x, y, velocity]")

            self.get_logger().info(f"Loaded {len(waypoints)} way points")

        except Exception as e:
            self.get_logger().warn(f"Failed to load waypoints from {self.global_path_dir}: {e}")

        # populate self.cx self.cy self.sp self.cyaw
        self.path_processor.process_global_path(waypoints)

        # All parameters are initialized
        self.mpc_thread = threading.Thread(target=self.run_mpc, daemon=True)
        self.mpc_thread.start()

    def nonlinear_mpc_control(self, xref):

        T, DT, NX, NU = self.T, self.DT, self.NX, self.NU      # X = [x, y, v, sin(yaw), cos(yaw)], U = [a, delta]

        # 1) CasADi Opti 환경 생성
        opti = ca.Opti()

        # 2) 결정변수: X (5 x (T+1)), U(2 x T)
        X = opti.variable(NX, T+1)  # X[:, k]
        U = opti.variable(NU, T)    # U[:, k]

        # 3) 비용(cost) 초기화
        cost_expr = 0.0

        # 4) 동역학 방정식 (자전거 모델)
        def bike_model(xk, uk):
            # xk = [x, y, v, sin(yaw), cos(yaw)], uk = [a, delta]
            
            X_next = ca.vertcat(
                xk[0] + xk[2]*xk[4]*DT,   # x_{k+1} = x_k + v_k cos(yaw_k) dt
                xk[1] + xk[2]*xk[3]*DT,   # y_{k+1} = y_k + v_k sin(yaw_k) dt
                xk[2] + uk[0]*DT,                 # v_{k+1} = v_k + a_k dt
                xk[3] + xk[4]*(xk[2] / self.WB) * ca.tan(uk[1]) * DT,
                xk[4] - xk[3]*(xk[2] / self.WB) * ca.tan(uk[1]) * DT
            )
            return X_next

        # 5) 비용 + 제약 설정
        for k in range(T):
            # 5-1) Cost
            pos_err = X[0:2,k] - xref[0:2,k]  # shape (2,)
            v_err = X[2,k] - xref[2,k]  # (v_k - v_ref)
            yaw_err = X[3:5,k] - xref[3:5,k]

            cost_expr += ca.mtimes([pos_err.T, self.Q[0:2, 0:2], pos_err])
            cost_expr += (v_err**2) * self.Q[2,2]
            cost_expr += ca.mtimes([yaw_err.T, self.Q[3:5,3:5],yaw_err])
            
            # 5-2) input Cost
            a_k     = U[0,k]
            delta_k = U[1,k]
            u_vec   = ca.vertcat(a_k, delta_k)
            cost_expr += ca.mtimes([u_vec.T, self.R, u_vec])

            # 5-3) Dynamic Constaint
            x_next = bike_model(X[:,k], U[:,k])
            opti.subject_to( X[:,k+1] == x_next )

            # 5-4) input difference Cost & Constraint
            if k < T-1:
                # Rd 부분
                du = U[:,k+1] - U[:,k]
                cost_expr += ca.mtimes([du.T, self.Rd, du])
                # ex: 조향속도 제한
                opti.subject_to( ( U[1,k+1] - U[1,k] ) <= self.MAX_DSTEER * DT )
                opti.subject_to( ( U[1,k+1] - U[1,k] ) >= -self.MAX_DSTEER * DT )

        # 6) Terminal Cost
        pos_err_final = X[0:2, T] - xref[0:2, T]
        v_err_final = X[2, T] - xref[2, T] 
        yaw_err_final = X[3:5, T] - xref[3:5, T]

        cost_expr += ca.mtimes([pos_err_final.T, self.Qf[0:2, 0:2], pos_err_final])
        cost_expr += ( v_err_final**2 ) * self.Qf[2,2]
        cost_expr += ca.mtimes([yaw_err_final.T, self.Qf[3:5, 3:5], yaw_err_final])

        # 7) Initial Constraint (현재 state)
        x_init = np.array([self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]])  # [x, y, v, sin(yaw), cos(yaw)]
        opti.subject_to( X[:,0] == x_init )

        # 8) Hard Constraint
        opti.subject_to( (U[0,:]) <= self.MAX_ACCEL )
        opti.subject_to( (U[0,:]) >= -self.MAX_ACCEL )
        opti.subject_to( (U[1,:]) <= self.MAX_STEER )
        opti.subject_to( (U[1,:]) >= -self.MAX_STEER )
        opti.subject_to( X[2,:] >= self.MIN_SPEED )
        opti.subject_to( X[2,:] <= self.MAX_SPEED )

        # 9) Set Cost Function
        opti.minimize( cost_expr )

        # 10) Solver 옵션 (IPOPT)
        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 500,
                "tol": 1e-2
            }
        }
        opti.solver("ipopt", opts)

        # Warm Start - optional
        if self.prev_sol_x is not None:
            opti.set_initial(X, self.prev_sol_x)
            opti.set_initial(U, self.prev_sol_u)

        # 11) Solve
        try:
            sol = opti.solve()
            x_opt = sol.value(X)  # shape (4, T+1)
            u_opt = sol.value(U)  # shape (2, T)
            obj_val = sol.value(cost_expr)
            status = sol.stats()['return_status']

            print(f"[NonlinearMPC] status={status}, cost={obj_val:.3f}")

            # 필요 시 numpy 변환
            ox = x_opt[0,:]  # x
            oy = x_opt[1,:]  # y
            ov = x_opt[2,:]  # velocity
            oyaw = np.arctan2(x_opt[3,:], x_opt[4,:]) # yaw
            
            oa = u_opt[0,:]
            odelta = u_opt[1,:]

            # 첫 입력
            a_cmd = oa[0]
            delta_cmd = odelta[0]
            
            # # warm start
            self.prev_sol_x = x_opt
            self.prev_sol_u = u_opt

            return a_cmd, delta_cmd, ox, oy, oa, oyaw

        except RuntimeError as e:
            print("[NonlinearMPC] Solve failed:", e)
            return None, None, None, None, None, None
    
    def run_mpc(self):

        target_ind, mind, self.p_l_ind = self.path_processor.calc_nearest_index(self.state, self.p_g_ind, self.p_l_ind)
        self.p_g_ind = target_ind

        prev_time = 0.0

        while rclpy.ok():
            time_now = time.time()

            xref, target_ind, self.p_l_ind = self.path_processor.calc_ref_trajectory(self.state, target_ind, self.p_l_ind)
            self.local_path_visualizer(xref)
            a_cmd, delta_cmd, ox, oy, oa, oyaw = self.nonlinear_mpc_control(xref)
            print(f"index : {target_ind}, xref : {xref[:,0]} state : {self.state}]")

            if a_cmd is None or delta_cmd is None:
                self.get_logger().warn("MPC solver failed. Using fallback controls.")
                a_cmd, delta_cmd = 0.1, 0.0  # Default inputs
            else:
                print(f"velocity : {self.state[2]}, steering angle : {delta_cmd}")

            time_elasped = time_now - prev_time
            prev_time = time_now

            self.publish_control(a_cmd, delta_cmd)
            time.sleep(0.1)
            print(f"MPC Loop : {time_elasped} s")
    
    def publish_control(self, a_cmd, delta_cmd):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "/ego_racecar/base_link"

        drive_msg.drive.steering_angle = delta_cmd
        v_cmd = self.state[2] + a_cmd * self.DT
        # if v_cmd >= 2.4 :
        #     v_cmd = 2.4
        drive_msg.drive.speed = v_cmd # Current speed

        self.ackm_drive_publisher.publish(drive_msg)    

    def local_path_visualizer(self, xref):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for i in range(xref.shape[1]):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            
            # xref 데이터를 position에 매핑
            pose.pose.position.x = xref[0,i]  # x
            pose.pose.position.y = xref[1,i]  # y
            pose.pose.position.z = 0.0         # z (평면이라면 0)

            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(xref[3, i] / 2.0)
            pose.pose.orientation.w = math.cos(xref[3, i] / 2.0)
            path_msg.poses.append(pose)
        
        self.local_path_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)  # Keeps the node responsive to callbacks
    except KeyboardInterrupt:
        node.get_logger().info("MPC control loop interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    


# To-Do:
# 1. mind in calc_nearest_index() could be used to determine "the real closest index" in sharp turns like U-turns or somewhat close.
# 2. implement local path input
# 3. make subsriber that changes self.p_l_ind to None (local path flag)