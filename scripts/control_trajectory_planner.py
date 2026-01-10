""" A functional wrapper of the SCPSolver subclass
DiffDriveSolver which knows about ROS2. """

from torch_traj_utils.diff_drive_solver import DiffDriveSolver, SDF, SolverParams
from dataclasses import dataclass
import numpy as np
from torch_traj_utils.scp_solver import SolverParams
from torch_traj_utils.scalar_field_interpolator import OccupancyMap, SDF, ScalarFieldInterpolator
import torch
import matplotlib.pyplot as plt
from torch.func import vmap

@dataclass
class ControlTrajectoryScenario:
    s0: np.ndarray
    u_goal: np.ndarray
    u_final: np.ndarray
    N: int
    u_min: np.ndarray
    # goal time of trajectory
    T: float

@dataclass
class ControlTrajectory:
    sc: ControlTrajectoryScenario
    s: np.ndarray
    u: np.ndarray
    J: np.ndarray
    dt: float
    N: int
    conv: bool
    status: str

@dataclass
class RobotEnvParams:
    robot_radius: float
    obstacle_radius: float
    occupied_thresh: int
    max_robot_v: float
    max_robot_omega: float

class ControlTrajectoryPlanner:
    # the core SCP solver for computing control trajectories
    sp: SolverParams
    ep: RobotEnvParams
    solver: DiffDriveSolver
    occ: OccupancyMap | None

    def __init__(self, solver_params: SolverParams, env_params: RobotEnvParams):
        self.sp = solver_params
        self.ep = env_params
        self.solver = DiffDriveSolver(sp=solver_params)
        self.occ = None
        self.sdf = None

    def reset_map(self, occ_map: OccupancyMap):
        self.occ = occ_map

    def trajectory(self, sc: ControlTrajectoryScenario) -> ControlTrajectory:
        """compute a trajectory to the time horizon"""
        # create the SDF based on the last map update
        self.sdf = SDF(self.occ, self.ep.robot_radius, self.ep.obstacle_radius)
        t = np.arange(0, sc.T + self.sp.dt, self.sp.dt)
        N = t.size - 1
        self.solver.reset_custom(sc.s0, sc.u_goal, sc.u_final, sc.u_min, N, sdf=self.sdf)
        self.solver.initialize_trajectory()
        s, u, J, conv, status, _, _ = self.solver.solve()
        # update with a rollout
        s,u = self.solver.get_ode().rollout(s, u, N)
        return ControlTrajectory(sc=sc,
                          s=s,
                          u=u,
                          J=J,
                          dt=self.sp.dt,
                          N=N,
                          conv=conv,
                          status=status)

    def do_diagnostic_plot(self, filename: str, traj:ControlTrajectory):
        """plot the sdf map with the trajectory as an overlay. Useful
        in comparison (for example) with what is showing up in rviz2."""
        # plot trajectory over obstacle constraints
        dx = 0.05
        x = np.arange(self.sdf.ox, self.sdf.ox + self.sdf.x_size + dx, dx)
        y = np.arange(self.sdf.oy, self.sdf.oy + self.sdf.y_size + dx, dx)
        xx, yy = np.meshgrid(x, y)

        s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u_pts = np.zeros(s_pts.shape)
        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        c = vmap(self.solver.sdf_interpolator.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        c_np = c.detach().cpu().numpy()
        c_np = np.reshape(c_np, (len(y), len(x)))
        plt.figure()
        plt.imshow(c_np,
                   origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
                   aspect='equal',  # or 'equal'
                   cmap='viridis'
                   )
        plt.colorbar()
        plt.title('trajectory and S.D.F.')
        plt.plot(traj.s[:, 0], traj.s[:, 1], linestyle="-", color="tab:red")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(filename, bbox_inches="tight")
