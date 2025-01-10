import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import numpy as np
import yaml
import os
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
import casadi
from casadi import SX, vertcat, nlpsol

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Load parameters from YAML file
        self.load_config()
        self.global_path = None
        self.cost_map = None  # Placeholder for the 2.5D cost map

        # State and control variables
        self.state = np.zeros(4)  # [x, y, theta, v]
        self.control = [0.0, 0.0]  # [acceleration, steering_rate]

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/global_path', self.global_path_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap', self.cost_map_callback, 10)

        # Publishers
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 10)
        self.cmd_drive_publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)

        # Timer for MPC loop
        self.create_timer(self.dt, self.run_mpc)
        self.state_updated = False  # Flag to track if odometry is updated

        self.get_logger().info("MPC Controller Initialized")

    def load_config(self):
        """Load parameters from YAML file."""
        package_dir = get_package_share_directory('mpc_controller')
        config_path = os.path.join(package_dir, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.validate_config(config)

        # Load car specifications
        car_config = config['car']
        self.wheel_base = car_config['wheel_base']
        self.max_speed = car_config['max_speed']
        self.max_steering_angle = np.deg2rad(car_config['max_steering_angle'])

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
        """Updates and validates the cost map."""
        # Extract cost map details
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        # Convert map data into a 2D numpy array
        map_data = np.array(msg.data).reshape((height, width))
        self.cost_map = np.where(map_data == -1, self.out_of_bounds_penalty, map_data)
        self.map_resolution = resolution
        self.map_origin = origin

    def global_path_callback(self, msg):
        """Updates the global path."""
        self.global_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

    # def global_path_callback(self, msg):
    #     self.global_path = [(pose.pose.position.x, pose.pose.position.y, getattr(pose.pose, 'speed', 0.0)) for pose in msg.poses]
    #     self.get_logger().info(f"Global path updated with {len(self.global_path)} waypoints.")


    def find_closest_point(self):
        """Finds the closest point on the global path to the current state."""
        if self.global_path is None or len(self.global_path) == 0:
            self.get_logger().warn("Global path is unavailable.")
            return None

        current_position = np.array([self.state[0], self.state[1]])
        distances = [np.linalg.norm(current_position - np.array(point)) for point in self.global_path]
        return np.argmin(distances)

    def publish_local_path(self, solution):
        """Publishes the predicted local path from the MPC solution."""
        path_msg = Path()
        path_msg.header.frame_id = "base_link"  # Local frame for visualization
        path_msg.header.stamp = self.get_clock().now().to_msg()

        num_states = 4 * (self.horizon + 1)
        states = np.array(solution["x"][:num_states]).reshape(4, self.horizon + 1)

        for k in range(self.horizon + 1):
            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = states[0, k]
            pose.pose.position.y = states[1, k]
            pose.pose.orientation.z = np.sin(states[2, k] / 2)
            pose.pose.orientation.w = np.cos(states[2, k] / 2)
            path_msg.poses.append(pose)

        self.local_path_publisher.publish(path_msg)
        self.get_logger().info(f"Published local path with {len(path_msg.poses)} points.")

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

        closest_idx = self.find_closest_point()
        if closest_idx is None:
            self.get_logger().warn("No valid closest point found. Skipping MPC computation.")
            return

        local_path = self.global_path[closest_idx:closest_idx + self.horizon + 1]

        x = SX.sym("x", 4, self.horizon + 1)
        u = SX.sym("u", 2, self.horizon)

        cost = 0.0
        constraints = []

        constraints.append(x[:, 0] == vertcat(*self.state))

        for k in range(self.horizon):
            x_next = vertcat(
                x[0, k] + self.dt * x[3, k] * casadi.cos(x[2, k]),
                x[1, k] + self.dt * x[3, k] * casadi.sin(x[2, k]),
                x[2, k] + self.dt * x[3, k] / self.wheel_base * casadi.tan(u[1, k]),
                x[3, k] + self.dt * u[0, k]
            )
            constraints.append(x[:, k + 1] - x_next)

            # Path-following cost
            if k >= len(local_path):
                path_ref = local_path[-1]
            else:
                path_ref = local_path[k]
            
            # Add position deviation cost
            pos_diff = x[:2, k] - path_ref[:2]
            cost += casadi.mtimes(pos_diff.T, self.Q[:2, :2] @ pos_diff)

            # # Add speed deviation cost
            # desired_speed = path_ref[2]
            # speed_diff = x[3, k] - desired_speed
            # cost += casadi.mtimes(speed_diff.T, self.Q[3, 3] * speed_diff)
            
            # Control effort cost
            cost += casadi.mtimes(u[:, k].T, self.R @ u[:, k])

        nlp = {
            "x": casadi.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1))),
            "f": cost,
            "g": casadi.vertcat(*constraints),
        }
        solver = casadi.nlpsol("solver", "ipopt", nlp)

        num_states = 4 * (self.horizon + 1)
        num_controls = 2 * self.horizon
        num_constraints = 4 + 4 * self.horizon

        initial_guess = np.zeros(num_states + num_controls)
        initial_guess[:4 * (self.horizon + 1)] = np.tile(self.state, (self.horizon + 1))

        lbg = np.zeros(num_constraints)
        ubg = np.zeros(num_constraints)

        solution = solver(x0=initial_guess, lbg=lbg, ubg=ubg)

        if solver.stats()["success"]:
            control_solution = np.array(solution["x"])[num_states:num_states + 2]
            self.publish_cmd_drive(control_solution)
            self.publish_local_path(solution)
        else:
            self.get_logger().warn("MPC optimization failed.")


    def publish_cmd_drive(self, control_solution):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        speed = float(max(min(control_solution[0], self.max_speed), 0))
        steering_angle = float(max(min(control_solution[1], self.max_steering_angle), -self.max_steering_angle))

        msg.drive.speed = speed
        msg.drive.steering_angle = steering_angle

        self.cmd_drive_publisher.publish(msg)
        self.get_logger().info(f"Published command: speed={speed:.2f}, angle={steering_angle:.2f}")

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
