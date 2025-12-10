from typing import List

class Goal:
    u_goal: List[float]
    u_final: List[float]
    u_min: List[float]
    t: float
    n: int
    dryrun: bool

class Result:
    success: bool
    message: str

class Feedback:
    progress: float
    status: str

class ControlTrajectoryScenarioMsg:
    Goal: type[Goal]
    Result: type[Result]
    Feedback: type[Feedback]
