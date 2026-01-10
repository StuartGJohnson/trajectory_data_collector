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
from typing import Any, Dict, List, Tuple
from builtin_interfaces.msg import Time
import uuid
from bag_recorder import BagRecorder, qos_state, qos_sensor, qos_latched
import os
from pathlib import Path as PPath
from ament_index_python.packages import get_package_share_directory
import argparse

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

# collection_topics = [
#     "/cmd_vel",
#     "/odom",
#     "/tf",
#     "/tf_static",
#     "/joint_states",
#     "/gt_pose"
# ]

# per_topic_qos = {
#     "/tf_static": qos_latched(depth=1),   # important: late-joiners get the static transforms
#     "/tf": qos_state(depth=200),          # tf is bursty; deeper queue helps
#     "/odom": qos_state(depth=50),
#     "/ground_truth": qos_state(depth=50),
#     "/joint_states": qos_state(depth=50),
#     "/gt_pose": qos_state(depth=50),
#
#     # Sensors often publish BEST_EFFORT; match that to avoid incompatibility
#     "/imu/data": qos_sensor(depth=200),
#     "/imu/mag": qos_sensor(depth=200),
#
#     # Depending on your publisher, this might be BEST_EFFORT; start sensor-ish
#     "/lidar_odom": qos_sensor(depth=100),
#
#     # cmd_vel is Twist (no header) and can be best-effort; start sensor-ish
#     "/cmd_vel": qos_sensor(depth=50),
# }

def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (absolute or relative)"
    )
    return parser.parse_known_args()

def resolve_config_path(path: str) -> PPath:
    p = PPath(path)

    if p.is_absolute():
        resolved = p
    else:
        cwd_candidate = (PPath.cwd() / p).resolve()
        if cwd_candidate.exists():
            resolved = cwd_candidate
        else:
            # try package share/config
            pkg_share = PPath(get_package_share_directory("trajectory_data_collector"))
            pkg_candidate = (pkg_share / "config" / p).resolve()
            if pkg_candidate.exists():
                resolved = pkg_candidate
            else:
                raise FileNotFoundError(
                    f"Config not found as absolute, cwd-relative, or package config: {path}"
                )

    return resolved


from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)

_HISTORY = {
    "keep_last": QoSHistoryPolicy.KEEP_LAST,
    "keep_all": QoSHistoryPolicy.KEEP_ALL,
}
_RELIABILITY = {
    "reliable": QoSReliabilityPolicy.RELIABLE,
    "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
}
_DURABILITY = {
    "volatile": QoSDurabilityPolicy.VOLATILE,
    "transient_local": QoSDurabilityPolicy.TRANSIENT_LOCAL,
}

def qos_from_dict(d: dict) -> QoSProfile:
    history = _HISTORY.get(str(d.get("history", "keep_last")).lower(), QoSHistoryPolicy.KEEP_LAST)
    reliability = _RELIABILITY.get(str(d.get("reliability", "best_effort")).lower(), QoSReliabilityPolicy.BEST_EFFORT)
    durability = _DURABILITY.get(str(d.get("durability", "volatile")).lower(), QoSDurabilityPolicy.VOLATILE)
    depth = int(d.get("depth", 10))

    return QoSProfile(
        history=history,
        depth=depth,
        reliability=reliability,
        durability=durability,
    )

def _mat_from_param(obj: Dict[str, Any], name: str) -> np.ndarray:
    """
    obj = {"shape": [r,c], "data": [ ... ]}
    """
    if not isinstance(obj, dict) or "shape" not in obj or "data" not in obj:
        raise ValueError(f"{name} must be a dict with keys shape and data")
    shape = tuple(int(x) for x in obj["shape"])
    data = [float(x) for x in obj["data"]]
    arr = np.array(data, dtype=float)
    if arr.size != shape[0] * shape[1]:
        raise ValueError(f"{name}.data has {arr.size} elems but shape implies {shape[0]*shape[1]}")
    return arr.reshape(shape)

def _vec_from_param(lst: Any, name: str, n: int) -> np.ndarray:
    if not isinstance(lst, list):
        raise ValueError(f"{name} must be a list")
    if len(lst) != n:
        raise ValueError(f"{name} must have length {n}, got {len(lst)}")
    return np.array([float(x) for x in lst], dtype=float)

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
    ct_: ControlTrajectory | None
    ep_: RobotEnvParams
    sp_: SolverParams
    path_: Path | None
    path_available_: bool
    last_cmd_vel_time_: Time | None
    nav2_timeout_: Duration
    data_recorder_: BagRecorder
    topics_: List[str]
    per_topic_qos: Dict
    output_dir_: str

    def __init__(self, cfg: Dict):
        super().__init__('DataCollector')
        self.output_dir_ = cfg["output_dir"]
        self.topics_ = list(cfg["collection_topics"])
        qos_cfg = cfg["per_topic_qos"]
        self.per_topic_qos_ = {t: qos_from_dict(cfg_item) for t, cfg_item in qos_cfg.items()}
        self.occ_map_ = None
        self.ct_ = None
        self.pose_ = None
        self.request_map_update_ = False
        self.request_pose_update_ = False
        self.busy_ = False
        self.path_ = None
        self.path_available_ = False
        self.map_callback_ = self.create_subscription(OccupancyGrid, cfg['map_topic'], self.map_callback, latched_qos)
        self.pose_callback_ = self.create_subscription(PoseStamped, cfg['gt_topic'], self.pose_callback, 10)
        self.do_traj_callback_ = self.create_subscription(Bool, cfg['do_traj_topic'], self.do_traj_callback, 10)
        self.path_pub_ = self.create_publisher(Path, cfg['traj_topic'], latched_qos)
        self.cmd_vel_callback_ = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.last_cmd_vel_time_ = None
        self.nav2_timeout_ = Duration(seconds=cfg['nav2_timeout'])
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
        # self.ep_ = RobotEnvParams(
        #     robot_radius=0.2,
        #     obstacle_radius=0.2,
        #     max_robot_v=0.3,
        #     max_robot_omega=1.0,
        #     occupied_thresh=50)
        env_cfg = cfg["env_params"]
        self.ep_ = RobotEnvParams(
            robot_radius=float(env_cfg["robot_radius"]),
            obstacle_radius=float(env_cfg["obstacle_radius"]),
            max_robot_v=float(env_cfg["max_robot_v"]),
            max_robot_omega=float(env_cfg["max_robot_omega"]),
            occupied_thresh=int(env_cfg["occupied_thresh"]),
        )
        u_max = np.array([self.ep_.max_robot_v, self.ep_.max_robot_omega])[None, :]
        # self.sp_ = SolverParams(
        #     dt=0.1,
        #     P=1.0 * np.eye(3),
        #     Q=np.eye(3),
        #     R=np.eye(2),
        #     rho=0.02,
        #     rho_u=0.02,
        #     eps=0.001,
        #     cvxpy_eps=.001,
        #     max_iters=10000,
        #     u_max=u_max,
        #     s_max=np.array([]),
        #     max_solve_secs=90.0
        # )
        solver_cfg = cfg["solver_params"]
        self.sp_ = SolverParams(
            dt=float(solver_cfg["dt"]),
            P=_mat_from_param(solver_cfg["P"], "solver_params.P"),
            Q=_mat_from_param(solver_cfg["Q"], "solver_params.Q"),
            R=_mat_from_param(solver_cfg["R"], "solver_params.R"),
            Rd=np.array([]),
            rho=float(solver_cfg["rho"]),
            rho_u=float(solver_cfg["rho_u"]),
            eps=float(solver_cfg["eps"]),
            cvxpy_eps=float(solver_cfg["cvxpy_eps"]),
            max_iters=int(solver_cfg["max_iters"]),
            u_max=u_max,
            s_max=np.array([]),
            max_solve_secs=float(solver_cfg["max_solve_secs"]),
            solver_type=solver_cfg["solver_type"]
        )
        self.trajectory_planner_ = \
            ControlTrajectoryPlanner(solver_params=self.sp_, env_params=self.ep_)
        self.data_recorder_ = BagRecorder(self,per_topic_qos=self.per_topic_qos_)

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
        self.busy_ = True
        self.path_available_ = False

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

        self.ct_ = self.trajectory_planner_.trajectory(cts)

        if not self.ct_.conv:
            goal_handle.abort()
            result = ControlTrajectoryScenarioMsg.Result()
            result.success = False
            result.message = 'Trajectory solver did not converge.'
            self.get_logger().info(f'Goal {gug} failed.')
            self.busy_ = False
            return result

        # form and post the path
        self.path_ = from_trajectory(self.ct_, self.get_clock().now().to_msg())
        self.path_available_ = True
        self.get_logger().info("Publishing control_trajectory with %d points" % len(self.path_.poses))
        self.path_pub_.publish(self.path_)

        # save the trajectory diagnostic plot
        #self.trajectory_planner_.do_diagnostic_plot("/home/sjohnson/ugv_ws/ctp_diag.png", ct)

        # If we are here, "success"!
        goal_handle.succeed()
        result = ControlTrajectoryScenarioMsg.Result()
        result.success = True
        result.message = 'Trajectory planning completed - post True to /do_traj to execute.'
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

    def cmd_vel_callback(self, msg: Twist):
        clock_now = self.get_clock().now()
        self.last_cmd_vel_time_ = clock_now

    def is_nav2_active(self) -> bool:
        if self.last_cmd_vel_time_ is None:
            return False
        ros2_now = self.get_clock().now()
        return (ros2_now - self.last_cmd_vel_time_) < self.nav2_timeout_

    def do_traj_callback(self, do_it: Bool):
        """if: we are not busy, and a computed trajectory is available, and not stale, do it."""
        if self.busy_:
            return
        elif self.path_available_ and do_it.data == True:
            # Offload to thread so we don't block the executor
            thread = threading.Thread(target=self.execute_do_traj_callback, args=(), daemon=True)
            thread.start()

    def execute_do_traj_callback(self):
        # create an output directory
        run_time = time.strftime("%Y%m%d_%H%M%S")
        # dc is data_collector - to distinguish from explore, etc
        # btw, the bag file writer doesn't like it when I pre-create the bag
        # directory, so I'll split output targets into two dirs.
        bag_dir = os.path.join(self.output_dir_,"dc_bag_" + run_time)
        etc_dir = os.path.join(self.output_dir_,"dc_etc_" + run_time)
        os.makedirs(etc_dir)
        # save the trajectory diagnostic plot
        self.trajectory_planner_.do_diagnostic_plot(os.path.join(etc_dir,"diag_plot.png"), self.ct_)
        # start the data recorder
        self.busy_ = True
        self.get_logger().info(f'Starting data recording.')
        self.data_recorder_.start(bag_dir=bag_dir, topics=self.topics_)
        nav2_goal = FollowPath.Goal()
        nav2_goal.path = self.path_
        nav2_goal.controller_id = "FollowPath"
        nav2_goal.goal_checker_id = "general_goal_checker"
        nav2_goal_future = self.follow_path_client_.send_goal_async(nav2_goal)
        # none of the nav2 goal sync/status check stuff works - presumably
        # nav2 either has a bug, or the nav2 behavior tree is doing what we want in
        # a different way. See follow_path_monitor.py.
        # mark our path as stale - let's assume the robot will at least change state
        # when confronted with the follow path goal.
        self.path_available_ = False

        # have the data_recorder wait for nav2 path completion
        # do an initial sleep to give nav2 a chance to get started
        time.sleep(1.0)
        while self.is_nav2_active():
            self.get_logger().info(f'Waiting for nav2 path completion.')
            time.sleep(1.0)
        self.get_logger().info(f'nav2 seems to be done. Stopping data recorder.')
        self.data_recorder_.stop()
        self.get_logger().info(f'Data recording stopped: done!')
        self.busy_ = False

def main():
    args, _ = parse_args()
    config_path = resolve_config_path(args.config)
    import yaml
    with open(config_path, "r") as f:
        cfg: Dict = yaml.safe_load(f)

    rclpy.init()
    collector = DataCollector(cfg)

    # Give everything a moment to initialize
    collector.get_logger().info("System initializing... will be ready for new goals in 5 seconds.")
    time.sleep(5)  # Give map, AMCL, Nav2 time to settle
    collector.get_logger().info("Ready for goals.")
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


