from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.time import Time

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import SequentialWriter


def qos_sensor(depth: int = 50) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def qos_state(depth: int = 50) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def qos_latched(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class BagRecorder:
    """
    Record selected topics to MCAP using rosbag2_py, with per-topic QoS.

    - start(bag_dir, topics): begins recording
    - stop(): stops and closes bag

    Assumes your Node is already being spun by an executor.
    """

    def __init__(
        self,
        node: Node,
        serialization_format: str = "cdr",
        default_qos: Optional[QoSProfile] = None,
        per_topic_qos: Optional[Dict[str, QoSProfile]] = None,
    ) -> None:
        self._node = node
        self._serialization_format = serialization_format

        # Default: try to match "sensor-ish" to avoid incompatibilities.
        self._default_qos = default_qos or qos_sensor(depth=50)

        # Optional overrides: {"/topic": QoSProfile(...)}
        self._per_topic_qos = per_topic_qos or {}

        self._subs = []
        self._lock = threading.Lock()
        self._recording: bool = False
        self.logged_it_: bool = False

        self._topic_types: Dict[str, str] = {}

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self, bag_dir: str, topics: Sequence[str]) -> None:
        self.logged_it_ = False
        if self._recording:
            raise RuntimeError("Recorder already started")
        if not topics:
            raise ValueError("topics list is empty")

        # SequentialWriter is offended by this:
        #os.makedirs(bag_dir, exist_ok=True)

        topic_type_map = self._get_topic_type_map()

        writer = SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=self._serialization_format,
            output_serialization_format=self._serialization_format,
        )
        writer.open(storage_options, converter_options)

        self._subs.clear()
        self._topic_types.clear()
        
        self._recording = True

        for topic in topics:
            if topic not in topic_type_map:
                raise RuntimeError(f"Topic '{topic}' not found in ROS graph")

            type_str = topic_type_map[topic]
            msg_type = get_message(type_str)

            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=topic,
                    type=type_str,
                    serialization_format=self._serialization_format,
                    offered_qos_profiles="",
                )
            )

            qos = self._per_topic_qos.get(topic, self._default_qos)

            sub = self._node.create_subscription(
                msg_type,
                topic,
                self._make_cb(topic, writer),
                qos,
            )
            self._subs.append(sub)
            self._topic_types[topic] = type_str

            self._node.get_logger().info(f"Recording {topic} [{type_str}] QoS={self._qos_str(qos)}")

        self._node.get_logger().info(f"MCAP recording started: {bag_dir}")

    def stop(self) -> None:
        if not self._recording:
            return

        for sub in self._subs:
            try:
                self._node.destroy_subscription(sub)
            except Exception:
                pass
        self._subs.clear()

        self._recording = False
        self._node.get_logger().info("MCAP recording stopped.")

    def _make_cb(self, topic: str, writer):
        def _cb(msg):
            # ROS time (sim time compatible)
            t_ns = self._node.get_clock().now().nanoseconds
            data = serialize_message(msg)
            if topic == "/tf_static" and not self.logged_it_:
                self._node.get_logger().info("received tf_static.")
                self.logged_it_ = True
            with self._lock:
                if not self._recording:
                    return
                writer.write(topic, data, t_ns)

        return _cb

    def _get_topic_type_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, types in self._node.get_topic_names_and_types():
            if types:
                out[name] = types[0]
        return out

    @staticmethod
    def _qos_str(q: QoSProfile) -> str:
        return f"{q.reliability.name}/{q.durability.name}/depth={q.depth}"
