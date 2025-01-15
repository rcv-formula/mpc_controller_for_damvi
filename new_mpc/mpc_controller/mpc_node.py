import cvxpy, yaml, math, time, os, threading
import rclpy, tf2_ros
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_matrix
from ament_index_python.packages import get_package_share_directory


class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Load parameters from YAML file
        self.load_config()
        self.load_global_path(self.global_path_dir)

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("Waiting for TF frames...")
        time.sleep(2)  # Delay to allow TF frames to populate
        self.debug_tf()

        # State and control Variables
        self.state = np.zeros(4) # [x, y, v, yaw]
        self.control = np.zeros(2) # [acceleration, steering_angle]

        # Placeholders
        self.reference_path = None
        self.cost_map = None
        self.transformed_segment = None
        self.cx = self.cy = self.sp = self.cyaw = np.empty((0,))
        

        # Subscriber
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap', self.local_costmap_callback, 10)

        # Publisher
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.ackm_drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.segment_publisher = self.create_publisher(Path, '/visualized_path_segment', 10)

        # Start the MPC control loop in a separate thread
        self.mpc_thread = threading.Thread(target=self.run_mpc, daemon=True)
        self.mpc_thread.start()
        self.get_logger().info("MPC node started")

        # Updates at 10 Hz
        self.update_reference_path()
        self.reference_update_timer = self.create_timer(0.1, self.update_reference_path)  
        # Timer to publish path segments at 10 Hz
        self.create_timer(0.1, self.publish_reference_path)


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
        
        # MPC parameters
        mpc_config = config['mpc']
        self.NX = mpc_config['NX'] # X = [x, y, v, yaw]
        self.NU = mpc_config['NU'] # a = [accel, steer]
        self.T = mpc_config['T'] # Horizon Length
        self.DT = mpc_config['DT']  # [s] time tick

        self.N_IND_SEARCH = mpc_config['N_IND_SEARCH'] # Search Index Number
        self.MAX_ITER = 3  # Max iteration
        self.DU_TH = 0.1  # iteration finish param
        
        # MPC Weights
        weights_config = config['weights']
        self.Q = np.diag(weights_config['Q'])   # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.R = np.diag(weights_config['R'])  # input cost matrix
        self.Rd = np.diag(weights_config['Rd'])   # input difference cost matrix

        # Map
        map_config = config['map']
        self.out_of_bounds_penalty = map_config['out_of_bounds_penalty']
        self.global_path_dir = os.path.join(package_dir, 'map', f"{map_config['global_path']}.csv")
        
        # Control
        control_config = config['control']
        self.CONTROL_RATE = control_config['CONTROL_RATE']

    def validate_config(self, config):
        required_keys = ['car', 'mpc', 'weights', 'map']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
        
    def load_global_path(self, csv_file_path):
        try:
            waypoints = np.loadtxt(csv_file_path, delimiter=',')

            if waypoints.shape[1] != 3:
                self.get_logger().warn("Error: CSV File should have 3 columns")
                return []

            self.global_path_np = waypoints
            self.get_logger().info(f"Loaded {len(self.global_path_np)} way points from {csv_file_path}")

        except Exception as e:
            self.get_logger().warn(f"Failed to load waypoints from {csv_file_path}: {e}")
            return []

    def odom_callback(self, msg):
        # Update robot's state from Odometry
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _,_, yaw = euler_from_quaternion(orientation_list)

        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        self.state[2] = msg.twist.twist.linear.x
        self.state[3] = yaw

        # self.get_logger().info("Odom Callback Called")



    def update_reference_path(self):

        # 이거 시작할때 맵이랑 base_link랑 정렬되어있어야함. 시뮬에서는 정렬되어있으니 일단 보류
        # 실제로할땐, 처음에 transform_global_path_to_base_link()해줘야함. (odom이랑 base_link랑 처음엔 같다는 가정)
        # Ensure global path is available for initialization

        print(self.global_path_np)
        if self.cx.size == 0 or self.cy.size == 0:
            # self.get_logger().info("Populating reference arrays from the global path for initialization.")
            
            init_global_path = self.transform_global_path_to_base_link(self.global_path_np)
            init_reference_path = self.calculate_reference_points(init_global_path)
            self.cx = init_reference_path[:, 0]
            self.cy = init_reference_path[:, 1]
            self.sp = init_reference_path[:, 2]
            self.cyaw = init_reference_path[:, 3]

        # Calculate nearest index
        nearest_idx, _ = self.calculate_nearest_index(0)

        # Define segment range
        start_idx = max(0, nearest_idx - 10)
        end_idx = min(len(self.global_path_np), nearest_idx + 50)

        # Transform segment to base_link
        segment = self.global_path_np[start_idx:end_idx]
        transformed_segment = self.transform_global_path_to_base_link(segment)

        if transformed_segment is not None:
            self.reference_path = self.calculate_reference_points(transformed_segment)
            self.cx = self.reference_path[:, 0]
            self.cy = self.reference_path[:, 1]
            self.sp = self.reference_path[:, 2]
            self.cyaw = self.reference_path[:, 3]
            self.get_logger().info(f"Updated reference path: Start={start_idx}, End={end_idx}")
        else:
            self.get_logger().warn("Failed to transform global path segment.")

    def local_costmap_callback(self, msg):
        # Obtain local cost map
        pass    

    def transform_global_path_to_base_link(self, segments):
        # Check for valid transform

        for _ in range(10):  # Retry up to 10 times
            try:
                transform = self.tf_buffer.lookup_transform(
                    "ego_racecar/base_link", 
                    "map", 
                    rclpy.time.Time()
                )
                break
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                time.sleep(0.1)
                return None
        else:
            self.get_logger().error("Failed to lookup transform after multiple attempts")
            return None

        # Extract transloation and rotation from the transform
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert rotation to a transformation matrix
        transform_matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]

        # Transform each pose and store as NumPy array
        num_points = segments.shape[0]
        homogeneous_points = np.hstack((segments[:, :2], np.zeros((num_points, 1)), np.ones((num_points, 1))))
        transformed_points = (transform_matrix @ homogeneous_points.T).T

        # Preserve velocity and return trasnformed path
        transformed_path = np.hstack((transformed_points[:, :2], segments[:, 2:3]))

        return transformed_path

    def calculate_reference_points(self, transformed_path_segment):
        # Number of Points in transformed_path
        num_points = transformed_path_segment.shape[0]
        reference_points = np.zeros((num_points,4)) # [x, y, v, yaw]

        for i in range(num_points-1):
            # Current and next points
            current_point = transformed_path_segment[i]
            next_point = transformed_path_segment[i+1]

            # Store x and y
            reference_points[i,0] = current_point[0]
            reference_points[i,1] = current_point[1]
            reference_points[i,2] = current_point[2] # reference velocity

            # Calculate reference yaw (yaw)
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            reference_points[i,3] = np.arctan2(dy,dx)

        # Last point, dx dy can not be calculated
        reference_points[-1,0] = transformed_path_segment[-1,0]
        reference_points[-1,1] = transformed_path_segment[-1,1]
        reference_points[-1,2] = transformed_path_segment[-1,2] # reference velocity
        reference_points[-1,3] = reference_points[-2,3] if num_points > 1 else 0.0

        return reference_points

    def calculate_nearest_index(self, pind):
        if self.cx.size == 0 or self.cy.size == 0:
            self.get_logger().warn("Reference path is not populated. Cannot calculate nearest index.")
            return 0, float('inf')  # Return a default index and infinite distance

        # Limit search to a window near the previous index
        # search_start = max(0, pind - self.N_IND_SEARCH)
        # search_end = min(len(self.cx), pind + self.N_IND_SEARCH)

        # if search_start == search_end:
        #     search_start = 0
        #     search_end = len(self.cx)

        dx = [self.state[0]-icx for icx in self.cx]
        dy = [self.state[1]-icy for icy in self.cy]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]  # calculate distance
        
        min_d = min(d)    # (minimum distance)^2
        nearest_idx = d.index(min_d)
        print(nearest_idx)
        min_d = math.sqrt(min_d)

        # Check if the nearest point is in front of the car
        dxl = self.cx[nearest_idx] - self.state[0]
        dyl = self.cy[nearest_idx] - self.state[1]

        d_yaw = self.cyaw[nearest_idx] - math.atan2(dyl, dxl)
        angle = (d_yaw + math.pi) % (2 * math.pi) - math.pi # 각도를 -pi ~ pi로 정규화
        if angle < 0:
            min_d *= -1

        return nearest_idx, min_d

    def calculate_ref_trajectory(self, pind):
        xref = np.zeros((self.NX,self.T+1))
        dref = np.zeros((1, self.T+1))
        ncourse = len(self.cx)

        ind, _ = self.calculate_nearest_index(pind)

        if pind >= ind:
            ind = pind
        
        xref[0,0] = self.cx[ind]
        xref[1,0] = self.cy[ind]
        xref[2,0] = self.sp[ind]
        xref[3,0] = self.cyaw[ind]
        dref[0,0] = 0.0 # steer operational point should be 0
        
        travel = 0.0

        for i in range(self.T+1):
            travel += abs(self.state[2]) * self.DT
            dind = int(round(travel))

            if (ind + dind) < ncourse:
                xref[0,i] = self.cx[ind + dind]
                xref[1,i] = self.cy[ind + dind]
                xref[2,i] = self.sp[ind + dind]
                xref[3,i] = self.cyaw[ind + dind]
                dref[0,i] = 0.0
            else:
                xref[0,i] = self.cx[ncourse -1]
                xref[1,i] = self.cy[ncourse -1]
                xref[2,i] = self.sp[ncourse -1]
                xref[3,i] = self.cyaw[ncourse -1]
                dref[0,i] = 0.0

        return xref, ind, dref
    
    def get_linear_model_matrix(self, v, phi, delta):
        # simple bicycle model 
        # phi = yaw
        # delta = steering angle
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

    def predict_motion(self, x0, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        xbar[:, 0] = x0  # Initial state

        for t in range(1, self.T + 1):
            A, B, C = self.get_linear_model_matrix(xbar[2, t - 1], xbar[3, t - 1], 0.0)
            xbar[:, t] = A @ xbar[:, t - 1] + B @ np.zeros(self.NU) + C

        return xbar

    def linear_mpc_control(self, xref, xbar, x0, dref):
        """
        Solves the linear MPC optimization problem.

        Args:
            xref: Reference trajectory (shape: [NX, T + 1])
            xbar: Predicted trajectory (initial guess, shape: [NX, T + 1])
            dref: Reference control inputs (shape: [NU, T])

        Returns:
            oa: Optimized acceleration inputs (shape: [T])
            od: Optimized steering angle inputs (shape: [T])
            ox, oy, oyaw, ov: Predicted state trajectory
        """
        # self.get_logger().info(f"xref: {xref}")
        # self.get_logger().info(f"xbar: {xbar}") 
        # self.get_logger().info(f"x0: {x0}")
        # self.get_logger().info(f"dref: {dref}")

        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        # Build cost function and constraints
        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)  # Input cost
            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)  # State cost

            A, B, C = self.get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)  # Smooth input
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MAX_DSTEER * self.DT]

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)  # Terminal cost
        constraints += [x[:, 0] == x0]  # Initial state constraint
        constraints += [x[2, :] <= self.MAX_SPEED + 0.1]  # Relaxed velocity constraint
        constraints += [x[2, :] >= self.MIN_SPEED - 0.1]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL + 0.1]  # Relaxed acceleration limit
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER + 0.1]  # Relaxed steering limit

        # Solve the problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        # Handle solver results
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = self.get_nparray_from_matrix(x.value[0, :])
            oy = self.get_nparray_from_matrix(x.value[1, :])
            ov = self.get_nparray_from_matrix(x.value[2, :])
            oyaw = self.get_nparray_from_matrix(x.value[3, :])
            oa = self.get_nparray_from_matrix(u.value[0, :])
            od = self.get_nparray_from_matrix(u.value[1, :])
        else:
            self.get_logger().warn("MPC solver failed. Returning fallback controls.")
            return None, None, None, None, None, None

        return oa, od, ox, oy, oyaw, ov
    
    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC control with updating operational point iteratively
        """

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        ox, oy, oyaw, ov = None, None, None, None

        for i in range(self.MAX_ITER):

            xbar = self.predict_motion(x0, xref)
            poa, pod = oa[:], od[:]
            # Solve MPC
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            #self.get_logger().info(f"Iteration {i}: Control change (du) = {du}")

            if du <= self.DU_TH:
                #self.get_logger().info(f"Converged after {i} iterations.")
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov
    
    def get_nparray_from_matrix(self,x):
        return np.array(x).flatten()
    
    def run_mpc(self):
        """
        Runs the MPC control loop for a looping global path.
        """
        self.get_logger().info("Starting MPC control loop...")

        oa, od = None, None  # Previous acceleration and steering
        pind = 0  # Path index

        # Continuous control loop
        while rclpy.ok():
            if self.reference_path is None or len(self.reference_path) == 0 :
                self.get_logger().info("Waiting for reference path or state initialization...")
                time.sleep(0.1)
                continue
        # while rclpy.ok():
        #     if self.reference_path is None or len(self.reference_path) == 0 or np.all(self.state == 0):
        #         self.get_logger().info("Waiting for reference path or state initialization...")
        #         time.sleep(0.1)
        #         continue

            # Wrap path index for looping
            pind %= len(self.cx)
            pind = max(0, pind)   # Ensure it's within bounds

            # Compute reference trajectory
            xref, ind, dref = self.calculate_ref_trajectory(pind)
            pind = ind
            # self.get_logger().info(f"Current state: {self.state}")
            # self.get_logger().info(f"First reference point: {xref[:, 0]}")

            # Solve MPC
            oa, od, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(xref, self.state, dref, oa, od)

            if oa is None or od is None:
                self.get_logger().warn("MPC solver failed. Using fallback controls.")
                oa, od = [0.1] * self.T, [0.0] * self.T  # Default inputs
                accel, delta = 0.1, 0.0  # Default inputs
                break
            else:
                accel, delta = oa[0], od[0]

            self.publish_control(accel, delta)

            # Wait for the next control cycle
            time.sleep(self.CONTROL_RATE)

        self.get_logger().info("MPC control loop stopped.")

    def publish_control(self, accel, delta):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "ego_racecar/base_link"

        drive_msg.drive.speed = self.state[2]  # Current speed
        drive_msg.drive.acceleration = accel
        drive_msg.drive.steering_angle = delta

        self.ackm_drive_publisher.publish(drive_msg)

    def debug_tf(self):
        try:
            available_frames = self.tf_buffer.all_frames_as_string()
            self.get_logger().info(f"Available TF frames:\n{available_frames}")
        except Exception as e:
            self.get_logger().error(f"Failed to get TF frames: {e}")

    def debug_function(self):
        """
        Publishes the reference path and transformed segment for debugging.
        """
        path_msg = Path()
        path_msg.header.frame_id = "ego_racecar/base_link"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Visualize reference path
        if self.reference_path is not None:
            for i in range(len(self.reference_path)):
                pose = PoseStamped()
                pose.header.frame_id = "ego_racecar/base_link"
                pose.pose.position.x = self.reference_path[i, 0]
                pose.pose.position.y = self.reference_path[i, 1]
                path_msg.poses.append(pose)

        # Visualize transformed segment
        if self.transformed_segment is not None:
            for i in range(len(self.transformed_segment)):
                pose = PoseStamped()
                pose.header.frame_id = "ego_racecar/base_link"
                pose.pose.position.x = self.transformed_segment[i, 0]
                pose.pose.position.y = self.transformed_segment[i, 1]
                path_msg.poses.append(pose)

        self.local_path_publisher.publish(path_msg)

    def publish_reference_path(self):
        """
        Publishes the reference path as a nav_msgs/Path message.
        """
        if self.reference_path is None or len(self.reference_path) == 0:
            self.get_logger().warn("Reference path is not available. Cannot publish.")
            return

        # Create the Path message
        path_msg = Path()
        path_msg.header.frame_id = "map"  # Assumes reference path is in "map" frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in self.reference_path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = np.sin(point[3] / 2.0)  # Convert yaw to quaternion
            pose.pose.orientation.w = np.cos(point[3] / 2.0)
            path_msg.poses.append(pose)

        self.segment_publisher.publish(path_msg)
        #self.get_logger().info(f"Published reference path with {len(self.reference_path)} points.")



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



# To Do

# run_mpc()
    # # Wait until the reference path and odometry are available
    # while self.reference_path is None or self.state is None:
    #     self.get_logger().info("Waiting for global path and odometry...")
    #     time.sleep(1.0)
# 이 부분 나중에 local cost map 들어오면 바꾸기
# obstacle avoidance cost function 작성하기