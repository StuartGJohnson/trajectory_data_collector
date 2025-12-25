""" quick test of bag recorder. A simulated or real robot should be running, although possibly not active."""

import rclpy
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from std_msgs.msg import Bool
from rclpy.subscription import Subscription
import logging
import numpy as np
import copy
from torch_traj_utils.scalar_field_interpolator import OccupancyMap
from torch_traj_utils.diff_drive_solver import SolverParams
from control_trajectory_planner import ControlTrajectoryScenario, ControlTrajectoryPlanner, RobotEnvParams, ControlTrajectory
from trajectory_data_collector.action import ControlTrajectoryScenarioMsg
from follow_path_monitor import FollowPathMonitor
import threading
import time
from typing import Tuple
from builtin_interfaces.msg import Time
import uuid
from bag_recorder import BagRecorder, qos_state, qos_sensor, qos_latched
import os
from bag_recorder import BagRecorder

collection_topics = [
    "/cmd_vel",
    "/odom",
    "/tf",
    "/tf_static",
    "/joint_states",
    "/gt_pose"
]

per_topic_qos = {
    "/tf_static": qos_latched(depth=1),   # important: late-joiners get the static transforms
    "/tf": qos_state(depth=200),          # tf is bursty; deeper queue helps
    "/odom": qos_state(depth=50),
    "/ground_truth": qos_state(depth=50),
    "/joint_states": qos_state(depth=50),
    "/gt_pose": qos_state(depth=50),

    # Sensors often publish BEST_EFFORT; match that to avoid incompatibility
    "/imu/data": qos_sensor(depth=200),
    "/imu/mag": qos_sensor(depth=200),

    # Depending on your publisher, this might be BEST_EFFORT; start sensor-ish
    "/lidar_odom": qos_sensor(depth=100),

    # cmd_vel is Twist (no header) and can be best-effort; start sensor-ish
    "/cmd_vel": qos_sensor(depth=50),
}

class TestBag(Node):
    def __init__(self):
        super().__init__('TestBag')
        self.busy: bool = False
        self.done: bool = False
        self.data_recorder_ = BagRecorder(self,per_topic_qos=per_topic_qos)
        self.timer = self.create_timer(1.0, self.bag_it)
        # this is wall time
        self.run_time_sec = 5.0
        self.start_time = 0.0

    def bag_it(self):
        if not self.busy and not self.done:
            # create an output directory
            self.start_time = time.time()
            run_time = time.strftime("%Y%m%d_%H%M%S")
            # dc is data_collector - to distinguish from explore, etc
            bag_dir = os.path.join("trajectory_data_collector","dc_test_" + run_time)
            self.data_recorder_.start(bag_dir=bag_dir, topics=collection_topics)
            self.busy = True
        elif self.busy and not self.done: # stop if it is time
            current_time = time.time()
            if (current_time - self.start_time) > self.run_time_sec:
                self.data_recorder_.stop()
        else:
            return

def main(args=None):
    rclpy.init(args=args)
    collector = TestBag()
    # Give everything a moment to initialize
    collector.get_logger().info("System initializing... waiting 5 seconds.")
    time.sleep(5)  # Give map, AMCL, Nav2 time to settle
    collector.get_logger().info("Starting...")
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Keyboard Interrupt detected, shutting down.")
    finally:
        if rclpy.ok():  # Only destroy node if rclpy hasn't already shut down
            collector.destroy_node()
        if rclpy.ok():  # Only shutdown if not already shut down
            rclpy.shutdown()

if __name__ == '__main__':
    main()
