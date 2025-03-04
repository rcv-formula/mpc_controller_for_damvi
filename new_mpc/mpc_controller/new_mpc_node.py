import cvxpy, yaml, time, os, math, threading
import rclpy.duration
import rclpy, tf2_ros
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_matrix
from ament_index_python.packages import get_package_share_directory

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        #load parameters from YAML file
        self.load_config()
        
        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State and Control Variables
        self.state = np.zeros(4)     # [x, y, v, yaw]
        self.control = np.zeros(2)   # [acceleration, steering_angle]

        # Subscriber
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap', self.local_costmap_callback, 10)
        self.create_subscription(VescStateStamped, '/commands/motor/speed', self.speed_callback, 10)

        # Publisher
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.ackm_drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.segment_publisher = self.create_publisher(Path, '/visualized_path_segment', 10)

        # load_path_csv once
        self.timer = self.create_timer(2.0, self.load_path_csv)


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
        self.MAX_ITER = 3  # Max iteration
        self.DU_TH = 0.1  # iteration finish param
        self.dl = 1 # [m] dl 값을 더 작게 설정하면 참조 궤적의 지점 간 간격이 줄어들어 더 세밀한 경로를 생성
        # 예시로, Horizon length가 10인데 dl이 1이면 10m 앞의 궤적까지 예측
        # dl이 0.5m 면 5미터 앞의 궤적 예측.

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
        
        # Control
        control_config = config['control']
        self.CONTROL_RATE = control_config['CONTROL_RATE']
        self.SIM_MODE = control_config['SIM_MODE']

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

    def speed_callback(self, msg):
        # Speed value from VESC # Real World
        if not self.SIM_MODE :
            self.state[2] = msg.state.speed / self.ERPM_GAIN

    def local_costmap_callback(self, msg):
        pass

    def load_path_csv(self):
        # load csv to numpy array, then trasnform it to odometry frame
        try:
            waypoints = np.loadtxt(self.global_path_dir, delimiter=',')

            if waypoints.shape[1] != 3:
                raise ValueError("Global path CSV must have exactly 3 columns: [x, y, velocity]")

            self.global_path_np = waypoints
            # self.global_path_np = np.flipud(waypoints)
            self.get_logger().info(f"Loaded {len(self.global_path_np)} way points from {self.global_path_dir}")

        except Exception as e:
            self.get_logger().warn(f"Failed to load waypoints from {self.global_path_dir}: {e}")
            self.global_path_np = None

        if self.global_path_np is None:
            self.get_logger().warn("Global path is not loaded. Cannot transform.")
            return None

        # Load CSV only once
        self.timer.cancel()

        self.transformer()

        # All parameters are initialized
        self.mpc_thread = threading.Thread(target=self.run_mpc, daemon=True)
        self.mpc_thread.start()

    def transformer(self):

        for _ in range(10):
            try:
                transform = self.tf_buffer.lookup_transform(
                    "ego_racecar/base_link",            # Target frame (Odometry, but for simulation, base_link)
                    "map",                              # Source frame
                    rclpy.time.Time(),
                )
                
                break
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"TF lookup failed: {e}. Retrying...")
        else:
            self.get_logger().error("Failed to lookup TF after multiple attempts.")
            return None
        # Extract translation and rotation
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert rotation to transformation matrix
        transform_matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]

        # Apply transformation to the global path
        num_points = self.global_path_np.shape[0]
        homogeneous_points = np.hstack((self.global_path_np[:, :2], np.zeros((num_points, 1)), np.ones((num_points, 1))))
        transformed_points = (transform_matrix @ homogeneous_points.T).T

        # Preserve velocities and return transformed path
        self.transformed_path = np.hstack((transformed_points[:, :2], self.global_path_np[:, 2:3]))
        self.get_logger().info("Successfully transformed global path to odometry frame.")
        
        # populate self.cx self.cy self.sp self.cyaw
        self.calc_path_yaw()

    def calc_path_yaw(self):
        # calculate cyaw [rad]
        
        # Number of Points in transformed_path
        num_points = self.transformed_path.shape[0]
        path_points = np.zeros((num_points,4)) # [x, y, v, yaw]

        for i in range(num_points-1):
            # Current and next points
            current_point = self.transformed_path[i]
            next_point = self.transformed_path[i+1]

            # Store x and y
            path_points[i,0] = current_point[0]
            path_points[i,1] = current_point[1]
            path_points[i,2] = current_point[2] # reference velocity

            # Calculate reference yaw (yaw)
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            path_points[i,3] = np.arctan2(dy,dx)

        # Last point, dx dy can not be calculated
        path_points[-1,0] = self.transformed_path[-1,0]
        path_points[-1,1] = self.transformed_path[-1,1]
        path_points[-1,2] = self.transformed_path[-1,2] # reference velocity
        path_points[-1,3] = path_points[-2,3] if num_points > 1 else 0.0

        # Path Point
        self.cx = path_points[:,0]
        self.cy = path_points[:,1]
        self.sp = path_points[:,2]
        self.cyaw = path_points[:,3] # rad

    def calc_global_nearest_index(self):
        # find nearest index globally
        dx = self.cx - self.state[0]
        dy = self.cy - self.state[1]
        d = dx**2 + dy**2

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
        angle = (d_yaw + math.pi) % (2 * math.pi) - math.pi  # 각도를 -pi ~ pi로 정규화
        if angle < 0:
            mind *= -1

        return ind, mind

    def calc_ref_trajectory(self, pind):
        # calculates xref, dref, and call calc_nearest_index
        
        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        ncourse = len(self.cx)

        ind, _ = self.calc_nearest_index(pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(self.state[2]) * self.DT
            dind = int(round(travel / self.dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self.cx[ind + dind]
                xref[1, i] = self.cy[ind + dind]
                xref[2, i] = self.sp[ind + dind]
                xref[3, i] = self.cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref
    
    def predict_motion(self, xref):
        # calcualtes xbar
        xbar = xref * 0.0
        for i, _ in enumerate(self.state):
            xbar[i, 0] = self.state[i]

        for i in range(1, self.T + 1):
            xbar[0, i] = self.state[0]
            xbar[1, i] = self.state[1]
            xbar[2, i] = self.state[2]
            xbar[3, i] = self.state[3]

        return xbar

    def get_linear_model_matrix(self, v, phi, delta):
        # get A B C
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta) / self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = - self.DT * v * math.cos(phi) * phi
        C[3] = - self.DT * v * delta / (self.WB * math.cos(delta) ** 2)

        return A, B, C

    def linear_mpc_control(self, xref, dref, xbar):
        # oa odelta as output, use ox oy oyaw osp to compare with xref for accuracy
        # calls get_linear_model_matrix
        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                self.MAX_DSTEER * self.DT]

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)
        
        x0 = self.state
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = self.get_nparray_from_matrix(x.value[0, :])
            oy = self.get_nparray_from_matrix(x.value[1, :])
            ov = self.get_nparray_from_matrix(x.value[2, :])
            oyaw = self.get_nparray_from_matrix(x.value[3, :])
            oa = self.get_nparray_from_matrix(u.value[0, :])
            odelta = self.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov
    
    def iterative_linear_mpc_control(self, oa, od, xref, dref):
        # calls predict_motion and feed it to linear_mpc_motion
        """
        MPC control with updating operational point iteratively
        """
        ox, oy, oyaw, ov = None, None, None, None

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, dref, xbar)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov
    
    def run_mpc(self):
        # odometry topic susbcription check
        # speed topic subscription check
        
        # 모든 state 가 0인 경우 여기서 스탑, csv 파일 읽고 변환을 못했으면 여기서 스탑.
        # while self.transformed_path is None or self.transformed_path.size == 0 or np.all(self.state == 0):
        while self.transformed_path is None or self.transformed_path.size == 0: 
            self.get_logger().info("Waiting for path ...")
            time.sleep(1.0)

        target_ind, min_d_ = self.calc_global_nearest_index()

        odelta, oa = None, None
        
        while rclpy.ok():
            self.transformer()
            self.smooth_yaw()
            xref, target_ind, dref = self.calc_ref_trajectory(target_ind)
            self.local_path_visualizer(xref)
            print(target_ind)

            oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(oa, odelta, xref, dref)

            if oa is None or odelta is None:
                self.get_logger().warn("MPC solver failed. Using fallback controls.")
                oa, odelta = [0.1] * self.T, [0.0] * self.T  # Default inputs
                accel, delta = 0.1, 0.0  # Default inputs
                break
            else:
                accel, delta = oa[0], odelta[0]

            self.publish_control(accel, delta)
            

            # Wait for the next control cycle
            time.sleep(self.CONTROL_RATE)
    
    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()
    
    def publish_control(self, accel, delta):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "ego_racecar/base_link"

        drive_msg.drive.steering_angle = delta
        drive_msg.drive.speed = self.state[2] + accel*self.DT  # Current speed

        self.ackm_drive_publisher.publish(drive_msg)

    def local_path_visualizer(self, xref):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "ego_racecar/base_link"

        for i in range(xref.shape[1]):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "ego_racecar/base_link"
            
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

            while dyaw >= math.pi / 2.0:
                self.cyaw[i + 1] -= math.pi * 2.0
                dyaw = self.cyaw[i + 1] - self.cyaw[i]

            while dyaw <= -math.pi / 2.0:
                self.cyaw[i + 1] += math.pi * 2.0
                dyaw = self.cyaw[i + 1] - self.cyaw[i]

      

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
