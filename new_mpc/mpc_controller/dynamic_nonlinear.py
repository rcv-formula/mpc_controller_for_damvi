import yaml, time, os, math, threading
import rclpy.duration
import rclpy
import numpy as np
import casadi as ca
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory

from mpc_controller.csv_loader import csv_path_loader

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        #load parameters from YAML file
        self.load_config()

        # State and Control Variables
        self.state = np.zeros(7)     # [x, y, vx, vy, sin(yaw), cos(yaw), r]
        self.control = np.zeros(2)   # [acceleration, steering_angle]
        self.raw_yaw = 0.0 # 각도 값 비정규화

        # Subscriber
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float64, '/commands/motor/speed', self.speed_callback, 10)

        # Publisher
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.ackm_drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.load_path_csv()
        self.mpc_thread = threading.Thread(target=self.run_mpc, daemon=True)
        self.mpc_thread.start()

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

    def odom_callback(self, msg):
        # Update robot's state from Odometry, however since we are using Cartographer speed value is not given.
        # Speed value can be obtained by differentiating positions, or it can also be obtained from VESC (wheel encoder)
        # However on Simulator, speed value is provided via odometry message
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _,_, yaw = euler_from_quaternion(orientation_list)

        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        self.state[3] = yaw
        
        if self.SIM_MODE :
            self.state[2] = msg.twist.twist.linear.x

    def speed_callback(self, msg:Float64):
        # Speed value from VESC # Real World
        if not self.SIM_MODE :
            self.state[2] = msg.data / self.ERPM_GAIN

    def local_costmap_callback(self, msg):
        pass

    def load_path_csv(self):
        # load csv to numpy array, then trasnform it to odometry frame
        try:
            waypoints = np.loadtxt(self.global_path_dir, delimiter=',')

            if waypoints.shape[1] != 3:
                raise ValueError("Global path CSV must have exactly 3 columns: [x, y, velocity]")

            self.global_path_np = waypoints
            self.init_point = waypoints
            self.get_logger().info(f"Loaded {len(self.global_path_np)} way points from {self.global_path_dir}")

        except Exception as e:
            self.get_logger().warn(f"Failed to load waypoints from {self.global_path_dir}: {e}")
            self.global_path_np = None

        # # Load CSV only once
        # self.timer.cancel()

        # populate self.cx self.cy self.sp self.cyaw
        self.calc_global_path()

        # # All parameters are initialized
        # self.mpc_thread = threading.Thread(target=self.run_mpc, daemon=True)
        # self.mpc_thread.start()


    def calc_global_path(self):

        self.global_path_np = np.tile(self.global_path_np[:-1], (self.laps, 1))

        num_points = self.global_path_np.shape[0]
        path_points = np.zeros((num_points, self.NX)) # [x, y, vx, yaw, r]

        for i in range(num_points-1):
            # Current and next points
            current_point = self.global_path_np[i]
            next_point = self.global_path_np[i+1]

            # Store x and y
            path_points[i,0] = current_point[0]
            path_points[i,1] = current_point[1]
            path_points[i,2] = current_point[2] # reference V_x

            # Calculate reference yaw (yaw)
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            path_points[i,3] = math.atan2(dy, dx)

        # Last point, dx dy can not be calculated
        path_points[-1,0] = self.global_path_np[-1,0]
        path_points[-1,1] = self.global_path_np[-1,1]
        path_points[-1,2] = self.global_path_np[-1,2] # reference velocity
        path_points[-1,3] = path_points[-2,3] if num_points > 1 else 0.0
        
        # Path Point
        self.cx = path_points[:,0]
        self.cy = path_points[:,1]
        self.sp = path_points[:,2] # slow down
        self.cyaw = self.angle_mod(path_points[:,3]) # rad
        
        # # path debugging
        # a = np.asanyarray([self.cx, self.cy, self.sp, self.cyaw])
        # np.savetxt("foo1.csv", a.T, delimiter=",")

    def calc_global_nearest_index(self):

        if self.state is None:
            self.get_logger().warn("State is None")

        dx = self.init_point[:,0] - self.state[0]
        dy = self.init_point[:,1] - self.state[1]
        d = np.sqrt(dx**2 + dy**2)

        ind = np.argmin(d)
        mind = math.sqrt(d[ind])

        return ind, mind

    def calc_nearest_index(self, pind):
        # find nearest index
        dx = [self.state[0] - icx for icx in self.cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [self.state[1] - icy for icy in self.cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind
        mind = math.sqrt(mind)

        dxl = self.cx[ind] - self.state[0]
        dyl = self.cy[ind] - self.state[1]

        d_yaw = self.cyaw[ind] - math.atan2(dyl, dxl)
        angle = self.angle_mod(d_yaw)
        if angle < 0:
            mind *= -1

        return ind, mind

    def calc_ref_trajectory(self, pind):
        # calculates xref, dref, and call calc_nearest_index
        
        xref = np.zeros((self.NX+1, self.T + 1))
        ncourse = len(self.cx)

        ind, _ = self.calc_nearest_index(pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]

        travel = 0.0
        SPEED_FACTOR = 0.5
        FIXED_INCREMENT = 1

        for i in range(1, self.T + 1):
            travel += abs(self.state[2]) * self.DT
            # dind = int(travel / self.DL)
            # dind = int(round(travel + self.state[2] * SPEED_FACTOR) / self.DL)
            dind = i * FIXED_INCREMENT

            if (ind + dind) <= ncourse -1:
                xref[0, i] = self.cx[ind + dind]
                xref[1, i] = self.cy[ind + dind]
                xref[2, i] = self.sp[ind + dind]
                xref[3, i] = self.cyaw[ind + dind]
            # 마지막 index 초과했을 때
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]

        return xref, ind

    def nonlinear_mpc_control(self, xref):
        """
        Nonlinear MPC using CasADi + IPOPT
        * State: [x, y, v, yaw]
        * Input: [a, delta]
        * Model: simple bicycle (no acceleration state)
        * T-step horizon
        * xref: shape (3, T+1) = desired [x, y, yaw] reference
        """

        T = self.T         # 예: 예측 시간 스텝
        DT = self.DT       # 예: 시뮬레이션 / MPC step
        NX = self.NX             # [x, y, v, yaw]
        NU = self.NU             # [v, delta]

        # 1) CasADi Opti 환경 생성
        opti = ca.Opti()

        # 2) 결정변수: X ( x (T+1)), U(2 x T)
        X = opti.variable(NX, T+1)  # X[:, k]
        U = opti.variable(NU, T)    # U[:, k]

        # 3) 비용(cost) 초기화
        cost_expr = 0.0

        # 4) 동역학 방정식 (자전거 모델)
        def bike_model(xk, uk):
            # xk = [x, y, yaw], uk = [v, delta]
            X_next = ca.vertcat(
                xk[0] + xk[2]*ca.cos(xk[3])*DT,   # x_{k+1} = x_k + v_k cos(yaw_k) dt
                xk[1] + xk[2]*ca.sin(xk[3])*DT,   # y_{k+1} = y_k + v_k sin(yaw_k) dt
                xk[2] + uk[0]*DT,                 # v_{k+1} = v_k + a_k dt
                xk[3] + (xk[2]/self.WB)*ca.tan(uk[1])*DT  # yaw_{k+1} = yaw_k + v_k/WB * tan(delta_k) dt
            )
            return X_next

        Qpos  = self.Q[0:2, 0:2]  # 2x2
        Qv    = self.Q[2,2]       # scalar
        Qyaw  = self.Q[3,3]       # scalar

        # 5) 비용 + 제약 설정
        for k in range(T):
            # (a) position error cost (x, y)
            pos_err = X[0:2,k] - xref[0:2,k]  # shape (2,)
            cost_expr += ca.mtimes([pos_err.T, Qpos, pos_err])

            # (b) speed error cost (v)
            v_err = X[2,k] - xref[2,k]  # (v_k - v_ref)
            cost_expr += (v_err**2) * Qv

            # (c) yaw error cost, (yaw, normalized)
            # yaw_ref = xref[2,k], yaw_robot = X[2,k]
            # delta_yaw = yaw_ref - yaw_robot
            yaw_err = xref[3,k] - X[3,k]
            # 주기성 처리: theta_mod = atan2(sin(delta_yaw), cos(delta_yaw))
            yaw_mod = ca.atan2(ca.sin(yaw_err), ca.cos(yaw_err))
            # yaw cost: theta_mod^2 * Qyaw
            cost_expr += ( yaw_mod**2 ) * Qyaw

            # (d) input cost
            a_k     = U[0,k]
            delta_k = U[1,k]
            u_vec   = ca.vertcat(a_k, delta_k)
            cost_expr += ca.mtimes([u_vec.T, self.R, u_vec])  # ex. R is 2x2

            # (e) 동역학 제약: X[:,k+1] == bike_model(X[:,k], U[:,k])
            x_next = bike_model(X[:,k], U[:,k])
            opti.subject_to( X[:,k+1] == x_next )

            # (f) 입력 변화 비용/제약
            if k < T-1:
                # Rd 부분
                du = U[:,k+1] - U[:,k]
                cost_expr += ca.mtimes([du.T, self.Rd, du])
                # ex: 조향속도 제한
                opti.subject_to( ( U[1,k+1] - U[1,k] ) <= self.MAX_DSTEER * DT )
                opti.subject_to( ( U[1,k+1] - U[1,k] ) >= -self.MAX_DSTEER * DT )

        # terminal cost (position)
        pos_err_final = X[0:2, T] - xref[0:2, T]
        cost_expr += ca.mtimes([pos_err_final.T, self.Qf[0:2, 0:2], pos_err_final])

        # terminal cost (speed)
        v_err_final = X[2, T] - xref[2, T] # last input vs last speed ref
        cost_expr += ( v_err_final**2 ) * self.Qf[2,2]

        # terminal cost (yaw)
        delta_yaw_final   = xref[3, T] - X[3, T]
        theta_mod_final   = ca.atan2(ca.sin(delta_yaw_final), ca.cos(delta_yaw_final))
        cost_expr += ( theta_mod_final**2 ) * self.Qf[3,3]

        # 6) 초기 상태 제약: X[:,0] = [self.state.x, self.state.y, self.state.yaw]
        #    => 현재 로봇 상태가 [x0, y0, yaw0], shape (3,)
        x_init = np.array([self.state[0], self.state[1], self.state[2], self.state[3]])  # [x, y, yaw] 정규화된 yaw
        opti.subject_to( X[:,0] == x_init )

        # 7) 상태/입력 범위 제약
        opti.subject_to( (U[0,:]) <= self.MAX_ACCEL )
        opti.subject_to( (U[0,:]) >= -self.MAX_ACCEL )
        opti.subject_to( (U[1,:]) <= self.MAX_STEER )
        opti.subject_to( (U[1,:]) >= -self.MAX_STEER )
        opti.subject_to( X[2,:] >= self.MIN_SPEED )
        opti.subject_to( X[2,:] <= self.MAX_SPEED )

        # 8) 목적함수 설정
        opti.minimize( cost_expr )

        # 9) Solver 옵션 (IPOPT)
        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 500,
                "tol": 1e-2
            }
        }
        opti.solver("ipopt", opts)

        # # (warm start) - OPTIONAL
        # if self.prev_sol_x and self.prev_sol_u is not None:
        #     opti.set_initial(X, self.prev_sol_x)
        #     opti.set_initial(U, self.prev_sol_u)

        # 10) Solve
        try:
            sol = opti.solve()
            x_opt = sol.value(X)  # shape (4, T+1)
            u_opt = sol.value(U)  # shape (2, T)
            obj_val = sol.value(cost_expr)
            status = sol.stats()['return_status']

            # 필요 시 numpy 변환
            ox = x_opt[0,:]  # x
            oy = x_opt[1,:]  # y
            ov = x_opt[2,:]  # velocity
            oyaw = x_opt[3,:] # yaw
            
            oa = u_opt[0,:]
            odelta = u_opt[1,:]

            # 첫 입력
            a_cmd = oa[0]
            delta_cmd = odelta[0]

            print(f"[NonlinearMPC] status={status}, cost={obj_val:.3f}")
            # # 저장해서 warm start에 쓰거나
            # self.prev_sol_x = x_opt
            # self.prev_sol_u = u_opt

            return a_cmd, delta_cmd, ox, oy, oa ,oyaw

        except RuntimeError as e:
            print("[NonlinearMPC] Solve failed:", e)
            return None, None, None, None, None, None
    
    def run_mpc(self):
        # csv 파일 읽고 변환을 못했으면 여기서 스탑.
        while self.global_path_np is None or self.global_path_np.size == 0: 
            self.get_logger().info("No Global Path, loading CSV might not have worked properly")
            break

        target_ind, min_d_ = self.calc_global_nearest_index()
        self.smooth_yaw()
        prev_time = 0.0

        while rclpy.ok():
            time_now = time.time()

            xref, target_ind = self.calc_ref_trajectory(target_ind)
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
            print(f"MPC Loop : {time_elasped} s")
    
    def publish_control(self, a_cmd, delta_cmd):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "/base_link"

        drive_msg.drive.steering_angle = delta_cmd
        v_cmd = self.state[2] + a_cmd * self.DT
        if v_cmd >= 2.4 :
            v_cmd = 2.4
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

    def smooth_yaw(self):

        for i in range(len(self.cyaw) - 1):
            dyaw = self.cyaw[i + 1] - self.cyaw[i]

            while dyaw >= math.pi :
            # if dyaw >= math.pi:
                self.cyaw[i + 1] -= 2.0 * math.pi
                dyaw = self.cyaw[i + 1] - self.cyaw[i]

            while dyaw < -math.pi :
            # elif dyaw < -math.pi:
                self.cyaw[i + 1] += 2.0 * math.pi
                dyaw = self.cyaw[i + 1] - self.cyaw[i]

    def angle_mod(self, x):

        if isinstance(x, float):
            is_float = True
        else:
            is_float = False

        x = np.asarray(x).flatten()

        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

        if is_float:
            return mod_angle.item()
        else:
            return mod_angle

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
    
