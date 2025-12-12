#!/usr/bin/env python
"""
A python implementation of a trajectory planner and execution node which:
  1) subscribes to /map to obtain an occupancy map
  2) converts the occupancy map to a numpy array
  3) accepts (as a ROS2 ActionServer) request for control trajectories
        - not automated past this task level
        - assumes a human is planning these trajectories with some level
            of foresight
        - keeps track of
  4) uses trajectory tools to determine one or more paths which:
        - avoid obstacles
        - follow control preferences
  5) instructs nav2 to closely follow these trajectories
  6) collects bag files during trajectory execution, with names and locations managed
  7) monitors nav2 state so that unexpected behavior is handled well
"""
import rclpy
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import PoseStamped, Point, Quaternion
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

def uuid_to_string(uuid_msg) -> str:
    return str(uuid.UUID(bytes=bytes(uuid_msg.uuid)))


def from_msg(ctsm: ControlTrajectoryScenarioMsg.Goal) -> Tuple[ControlTrajectoryScenario, bool]:
     cts = ControlTrajectoryScenario(
         s0=np.array([]),
         u_final=np.array(ctsm.u_final),
         u_goal=np.array(ctsm.u_goal),
         T=ctsm.t,
         N=ctsm.n,
         u_min=np.array(ctsm.u_min)[None,:]
     )
     return cts, ctsm.dryrun

def euler_to_quaternion(roll, pitch, yaw) -> Tuple:
    # From TF2 examples (simplified for 2D yaw)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return x, y, z, w

def quaternion_to_euler(q: Quaternion) -> Tuple:
    x, y, z, w = q.x, q.y, q.z, q.w

    t0 = +2.0 * (q.w * q.x + q.y * q.z)
    t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (q.w * q.y - q.z * q.x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw

def from_trajectory(ct: ControlTrajectory, ros_time:Time) -> Path:
    # note that the timestamps on the PoseStamped objects
    # are ignored by Nav2 - at least, according to GPT5
    n_pts = ct.s.shape[0]
    path_msg = Path()
    path_msg.header.stamp = ros_time
    pose_frame: str = "map"
    path_msg.header.frame_id = pose_frame
    poses: list[PoseStamped] = []
    for i in range(0, n_pts):
        pose = PoseStamped()
        pose.header.stamp = ros_time
        pose.header.frame_id = pose_frame
        pose.pose.position = Point(x=ct.s[i,0],y=ct.s[i,1])
        x, y, z, w = euler_to_quaternion(0, 0, ct.s[i,2])
        pose.pose.orientation = Quaternion(x=x,y=y,z=z,w=w)
        poses.append(pose)
    path_msg.poses = poses
    return path_msg

latched_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

class DataCollector(Node):
    busy_: bool
    request_map_update_: bool
    request_pose_update_: bool
    map_sub_: Subscription
    occ_map_: OccupancyMap | None
    pose_: PoseStamped | None
    occ_map_thresh_: float
    action_server_: ActionServer
    trajectory_planner_: ControlTrajectoryPlanner
    ep_: RobotEnvParams
    sp_: SolverParams

    def __init__(self):
        super().__init__('DataCollector')
        self.occ_map_ = None
        self.pose_ = None
        self.request_map_update_ = False
        self.request_pose_update_ = False
        self.busy_ = False
        self.map_callback_ = self.create_subscription(OccupancyGrid, '/map', self.map_callback, latched_qos)
        self.pose_callback_ = self.create_subscription(PoseStamped, '/gt_pose', self.pose_callback, 10)
        self.path_pub_ = self.create_publisher(Path, "control_trajectory", latched_qos)
        self._action_server = ActionServer(
            self,
            ControlTrajectoryScenarioMsg,
            'ControlTrajectoryScenarioMsg',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.follow_path_client_: ActionClient = ActionClient(
            self,
            FollowPath,
            "follow_path",
        )
        #self.follow_path_monitor_ = FollowPathMonitor(self, latched_qos)
        self.ep_ = RobotEnvParams(
            robot_radius=0.2,
            obstacle_radius=0.2,
            max_robot_v=0.3,
            max_robot_omega=1.0,
            occupied_thresh=50)
        u_max = np.array([self.ep_.max_robot_v, self.ep_.max_robot_omega])[None, :]
        self.sp_ = SolverParams(
            dt=0.1,
            P=1.0 * np.eye(3),
            Q=np.eye(3),
            R=np.eye(2),
            rho=0.02,
            rho_u=0.02,
            eps=0.001,
            cvxpy_eps=.001,
            max_iters=10000,
            u_max=u_max,
            s_max=np.array([]))
        self.trajectory_planner_ = \
            ControlTrajectoryPlanner(solver_params=self.sp_, env_params=self.ep_)

    def unused_execute_callback(self, oal_handle: ServerGoalHandle):
        # This will never be called if handle_accepted_callback takes over,
        # but we must provide it to satisfy the API.
        raise RuntimeError("execute_callback should not be called in this server")

    def goal_callback(self, goal_request: ControlTrajectoryScenarioMsg.Goal):
        self.get_logger().info(f'Received goal: {goal_request.u_goal}')

        if self.busy_:
            self.get_logger().warn('DataCollector is busy; rejecting goal')
            return GoalResponse.REJECT

        # You might do other quick checks here (bounds, resource availability, etc.)
        return GoalResponse.ACCEPT

    # Ensure long-running work is done in a separate thread
    def handle_accepted_callback(self, goal_handle: ServerGoalHandle):
        # Offload to thread so we don't block the executor
        thread = threading.Thread(target=self.execute_callback, args=(goal_handle,), daemon=True)
        thread.start()

    # Optional: decide if cancellation is allowed
    def cancel_callback(self, goal_handle: ServerGoalHandle):
        goal_id =  goal_handle.goal_id
        goal: ControlTrajectoryScenarioMsg.Goal = goal_handle.request
        gug = uuid_to_string(goal.u_goal)
        self.get_logger().info(f'Received cancel request for {gug}')
        # You can put logic here: if too late, you can reject cancellation
        # tell the nav2 handler to stop, or whatever is appropriate
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle):
        """
        Long-running execution of the goal.

        This runs in a worker thread. It should:
        - periodically check for cancellation
        - publish feedback
        - eventually call succeed()/abort()/canceled() and return a Result
        """

        goal: ControlTrajectoryScenarioMsg.Goal = goal_handle.request
        goal_id = goal_handle.goal_id
        gug = uuid_to_string(goal_id)
        self.get_logger().info(f'Starting execution for {gug}')

        feedback_msg: ControlTrajectoryScenarioMsg.Feedback = ControlTrajectoryScenarioMsg.Feedback()

        # build a scenario for the planner
        cts, dryrun = from_msg(goal)

        # request robot position and grab last map update
        while (self.occ_map_ is None) | (self.pose_ is None):
            # poll until data available
            self.get_logger().info(f'Waiting for robot pose and map data for goal: {gug}')
            time.sleep(1.0)

        # start the planner (this is a solid block of time - for now)

        # get roll, pitch, yaw
        euler_angles = quaternion_to_euler(self.pose_.pose.orientation)

        s0 = np.array(
            [self.pose_.pose.position.x,
             self.pose_.pose.position.y,
             euler_angles[2]])

        cts.s0 = s0

        self.trajectory_planner_.reset_map(self.occ_map_)
        self.get_logger().info(f'Computing control trajectory for goal: {gug}')
        ct = self.trajectory_planner_.trajectory(cts)

        if not ct.conv:
            goal_handle.abort()
            result = ControlTrajectoryScenarioMsg.Result()
            result.success = False
            result.message = 'Trajectory solver did not converge.'
            self.get_logger().info(f'Goal {gug} failed.')
            return result

        # form and post the path
        path = from_trajectory(ct, self.get_clock().now().to_msg())
        self.get_logger().info("Publishing control_trajectory with %d points" % len(path.poses))
        self.path_pub_.publish(path)

        # save the trajectory diagnostic plot
        #self.trajectory_planner_.do_diagnostic_plot("/home/sjohnson/ugv_ws/ctp_diag.png", ct)

        # none of the nav2 handshaking is working, but some bits of code are
        # left in place for future cleanup (or my own follow-path node)
        nav2_success = True
        if not dryrun:
            # reset monitor
            # self.follow_path_monitor_.set_unknown()
            # hand off path to nav2 for following
            # note: we may want to use the async version here
            nav2_goal = FollowPath.Goal()
            nav2_goal.path = path
            nav2_goal.controller_id = "FollowPath"
            nav2_goal.goal_checker_id = "general_goal_checker"
            nav2_goal_future = self.follow_path_client_.send_goal_async(nav2_goal)

            # none of this nav2 goal synch/status check stuff works - presumably
            # nav2 either has a bug, or the nav2 behavior tree is doing what
            # we want in a different way

            # let nav2 swallow before checking status
            #time.sleep(2.0)

            # while self.follow_path_monitor_.is_busy:
            #     time.sleep(1.0)

            # while not nav2_goal_future.done():
            #     time.sleep(1.0)
            #
            # nav2_goal_handle = nav2_goal_future.result()
            # if not nav2_goal_handle.accepted:
            #     self.get_logger().info(f'Goal {gug} not accepted by nav2.')
            #     nav2_success = False
            # else:
            #     nav2_result_future = nav2_goal_handle.get_result_async()
            #     while not nav2_result_future.done():
            #         time.sleep(1.0)

        # If we are here, "success"!
        goal_handle.succeed()
        result = ControlTrajectoryScenarioMsg.Result()
        if nav2_success:
            result.success = True
            result.message = 'Trajectory planning completed and nav2 trajectory started (probably).'
            self.get_logger().info(f'Goal {gug} succeeded')
        else:
            # not yet supported, since I can't seem to get any feedback from nav2
            result.success = False
            result.message = 'An error occurred in nav2.'
            self.get_logger().info(f'Goal {gug} succeeded')
        self.busy_ = False
        return result

    def map_callback(self, occ: OccupancyGrid):
        tmp_map = np.frombuffer(occ.data, dtype=np.int8).reshape(occ.info.height, occ.info.width)
        map = (tmp_map > self.ep_.occupied_thresh)
        res = occ.info.resolution
        origin = np.array([occ.info.origin.position.x, occ.info.origin.position.y])
        self.occ_map_ = OccupancyMap(map=map, res=res, origin=origin)

    def pose_callback(self, pose: PoseStamped):
        self.pose_ = copy.deepcopy(pose)


def main(args=None):
    rclpy.init(args=args)
    collector = DataCollector()

    # Give everything a moment to initialize
    collector.get_logger().info("System initializing... will be ready for new goals in 5 seconds.")
    time.sleep(5)  # Give map, AMCL, Nav2 time to settle

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


