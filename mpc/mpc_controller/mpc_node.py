import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
import numpy as np
import yaml
import os
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory
import time
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
import casadi
from casadi import SX, vertcat, nlpsol
from scipy.interpolate import interp1d

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Load parameters from YAML file
        self.load_config()
        self.global_path = None

        # State and control variables
        self.state = np.zeros(4)  # [x, y, theta, v]
        self.control = [0.0, 0.0]  # [acceleration, steering_rate]
        self.cost_map = None  # Placeholder for the 2.5D cost map

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap', self.cost_map_callback, 10)
        self.create_subscription(Path, '/global_path', self.global_path_callback, 10)

        # Publishers
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.cmd_drive_publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)

        # Timer for MPC loop
        self.create_timer(self.dt, self.run_mpc)
        # Flag to indicate if /odom has been received and updated.
        self.state_updated = False 

        self.get_logger().info("MPC Controller Initialized")

    def load_config(self):
        """Load parameters from a YAML file."""
        package_dir = get_package_share_directory('mpc_controller')
        config_path = os.path.join(package_dir, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.validate_config(config)

        # Load car specifications
        car_config = config['car']
        self.wheel_base = car_config['wheel_base']
        self.wheel_radius = car_config['wheel_radius']
        self.track_width = car_config['track_width']
        self.max_speed = car_config['max_speed']
        self.max_steering_angle = np.deg2rad(car_config['max_steering_angle'])  # Convert to radians

        # Load MPC parameters
        mpc_config = config['mpc']
        self.horizon = mpc_config['horizon']
        self.dt = mpc_config['dt']
        self.max_acceleration = mpc_config['max_acceleration']
        self.max_steering_rate = mpc_config['max_steering_rate']

        # Load MPC weights
        weights_config = config['weights']
        self.Q = np.diag(weights_config['Q'])
        self.R = np.diag(weights_config['R'])

        # Load Map Resolution
        map_config = config['map']
        self.out_of_bounds_penalty = map_config['out_of_bounds_penalty']

        self.get_logger().info(f"Loaded configuration from {config_path}")

    def validate_config(self, config):
        required_keys = ['car', 'mpc', 'weights', 'map']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

    def odom_callback(self, msg):
        """Updates the robot's state from odometry."""
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        self.state[2] = yaw
        self.state[3] = msg.twist.twist.linear.x

        self.state_updated = True  # Set the flag to True
        self.get_logger().debug(f"Odometry updated: {self.state}")

    def cost_map_callback(self, msg):
        """Updates the cost map."""
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        map_data = np.array(msg.data).reshape((height, width))
        self.cost_map = np.where(map_data == -1, self.out_of_bounds_penalty, map_data)

        self.map_resolution = resolution
        self.map_origin = origin

    def global_path_callback(self, msg):
        """Updates the global path."""
        self.global_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

    def interpolate_path(self, global_path, target_length):
        """Interpolates the global path to ensure smooth local trajectories."""
        global_x, global_y = zip(*global_path)
        t = np.linspace(0, 1, len(global_x))
        t_interp = np.linspace(0, 1, target_length)
        interp_x = interp1d(t, global_x, kind="linear")(t_interp)
        interp_y = interp1d(t, global_y, kind="linear")(t_interp)
        return list(zip(interp_x, interp_y))

    def find_closest_point(self):
        """Finds the closest point on the global path to the current state."""
        if self.global_path is None or len(self.global_path) == 0:
            self.get_logger().warn("Global path is unavailable.")
            return None

        current_position = np.array([self.state[0], self.state[1]])
        distances = [np.linalg.norm(current_position - np.array(point)) for point in self.global_path]
        return np.argmin(distances)

    def generate_local_trajectory(self, closest_idx):
        """Generates a local trajectory from the global path."""
        if self.global_path is None or len(self.global_path) == 0:
            return None

        end_idx = min(closest_idx + self.horizon + 1, len(self.global_path))
        local_path = self.global_path[closest_idx:end_idx]

        if len(local_path) < self.horizon + 1:
            local_path = self.interpolate_path(self.global_path, self.horizon + 1)

        return local_path

    def run_mpc(self):
        if self.cost_map is None:
            self.get_logger().warn("Skipping MPC as cost map is unavailable.")
            return

        if self.global_path is None or len(self.global_path) == 0:
            self.get_logger().warn("Global path is unavailable. Skipping MPC computation.")
            return

        if not self.state_updated:
            self.get_logger().warn("Odometry data unavailable. Skipping MPC computation.")
            return

        # Symbolic variables for states and controls
        x = SX.sym("x", 4, self.horizon + 1)  # States [x, y, theta, v]
        u = SX.sym("u", 2, self.horizon)     # Controls [acceleration, steering_rate]
        slack = SX.sym("slack", self.horizon)  # Slack variables for path-following relaxation

        constraints = []  # Constraints list
        cost = 0.0        # Cost accumulator

        # Penalty weights
        w_pos = 10.0  # Penalty for negative velocity
        w_max = 10.0  # Penalty for exceeding max velocity
        w_acc = 10.0  # Penalty for exceeding max acceleration
        w_steer = 10.0  # Penalty for exceeding max steering rate
        w_slack = 1.0  # Penalty for slack variables

        # Initial state constraint
        constraints.append(x[:, 0] == vertcat(*self.state))

        for k in range(self.horizon):
            x_k = x[:, k]
            x_k_next = x[:, k + 1]
            u_k = u[:, k]

            # Dynamics constraints
            x_next_pred = vertcat(
                x_k[0] + self.dt * x_k[3] * casadi.cos(x_k[2]),
                x_k[1] + self.dt * x_k[3] * casadi.sin(x_k[2]),
                x_k[2] + self.dt * u_k[1],
                x_k[3] + self.dt * u_k[0]
            )
            constraints.append(x_k_next == x_next_pred)

            # Velocity penalties
            cost += w_pos * casadi.fmax(0, -x_k[3]) ** 2
            cost += w_max * casadi.fmax(0, x_k[3] - self.max_speed) ** 2

            # Acceleration and steering penalties
            cost += w_acc * casadi.fmax(0, casadi.fabs(u_k[0]) - self.max_acceleration) ** 2
            cost += w_steer * casadi.fmax(0, casadi.fabs(u_k[1]) - self.max_steering_rate) ** 2

            # Path-following cost with slack
            target_state = vertcat(self.global_path[k][0], self.global_path[k][1], 0.0, 0.0)
            state_diff = x_k[:2] - target_state[:2]
            cost += casadi.mtimes(state_diff.T, casadi.mtimes(self.Q[:2, :2], state_diff))[0, 0]
            cost += w_slack * slack[k] ** 2  # Penalize slack variable

        # Define NLP problem
        nlp = {
            "x": casadi.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)), slack),
            "f": cost,
            "g": casadi.vertcat(*constraints)
        }

        solver = nlpsol("solver", "ipopt", nlp)

        # Initial guess
        num_states = 4 * (self.horizon + 1)
        num_controls = 2 * self.horizon
        num_slack = self.horizon
        initial_guess = np.zeros(num_states + num_controls + num_slack)

        # Solve the optimization problem
        solution = solver(
            x0=initial_guess,
            lbg=0,
            ubg=0
        )

        if solver.stats()["success"]:
            # Extract control solution
            control_solution = np.array(solution["x"])[num_states:num_states + 2]
            self.control = [control_solution[0], control_solution[1]]
            self.publish_cmd_drive()
        else:
            self.get_logger().warn("MPC optimization failed.")


    def publish_cmd_drive(self):
        """Publishes the control commands as AckermannDriveStamped."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        speed = max(min(self.control[0], self.max_speed), -self.max_speed)
        steering_angle = max(min(self.control[1], self.max_steering_angle), -self.max_steering_angle)
        msg.drive.speed = speed
        msg.drive.steering_angle = steering_angle

        self.cmd_drive_publisher.publish(msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down MPC Controller.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
