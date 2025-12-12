"""An interesting idea/code posed by GPT5. This does not work, BTW."""

from rclpy.node import Node
from action_msgs.msg import GoalStatusArray, GoalStatus

class FollowPathMonitor:
    """Tracks whether Nav2's FollowPath action is active."""

    ACTIVE_STATES = {
        GoalStatus.STATUS_ACCEPTED,
        GoalStatus.STATUS_EXECUTING,
        GoalStatus.STATUS_CANCELING,
    }

    def __init__(self, node: Node, qos):
        self._node = node
        self._is_active = False
        self.is_known_ = False
        self.qos_ = qos

        # Subscribe to the Nav2 FollowPath status array
        self._sub = node.create_subscription(
            GoalStatusArray,
            'follow_path/_action/status',   # This is the canonical status topic
            self._status_cb,
            self.qos_,
        )

    def _status_cb(self, msg: GoalStatusArray):
        # Nav2 may list 0 or more goals here
        self._is_active = any(
            s.status in self.ACTIVE_STATES for s in msg.status_list
        )
        self.is_known_ = True

    def set_unknown(self):
        self.is_known_ = False

    @property
    def is_busy(self) -> bool:
        """True if Nav2 is currently executing a FollowPath goal."""
        if not self.is_known_:
            return True
        return self._is_active
