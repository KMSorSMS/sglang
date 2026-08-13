import sys
import uuid
from unittest.mock import patch

import pytest
import zmq

from sglang.srt.utils.network import get_zmq_socket
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class RecordingSocket:
    def __init__(self, events):
        self.events = events

    def setsockopt(self, option, value):
        self.events.append(("setsockopt", option, value))

    def connect(self, endpoint):
        self.events.append("connect")


class RecordingContext:
    def __init__(self, socket):
        self._socket = socket

    def socket(self, socket_type):
        return self._socket


class TestZmqSocketOptions(CustomTestCase):
    def test_custom_options_are_applied_to_real_socket(self):
        context = zmq.Context()
        endpoint = f"inproc://status-options-{uuid.uuid4().hex}"
        receiver = context.socket(zmq.PULL)
        receiver.bind(endpoint)
        socket = get_zmq_socket(
            context,
            zmq.PUSH,
            endpoint,
            bind=False,
            socket_options={
                zmq.CONFLATE: 1,
                zmq.SNDTIMEO: 0,
                zmq.LINGER: 0,
            },
        )
        try:
            self.assertEqual(socket.getsockopt(zmq.CONFLATE), 1)
            self.assertEqual(socket.getsockopt(zmq.SNDTIMEO), 0)
            self.assertEqual(socket.getsockopt(zmq.LINGER), 0)
        finally:
            socket.close(0)
            receiver.close(0)
            context.term()

    def test_custom_options_precede_connect(self):
        events = []
        socket = RecordingSocket(events)
        context = RecordingContext(socket)
        with patch(
            "sglang.srt.utils.network.config_socket",
            side_effect=lambda *_: events.append("defaults"),
        ):
            get_zmq_socket(
                context,
                zmq.PUSH,
                "tcp://127.0.0.1:12345",
                bind=False,
                socket_options={zmq.CONFLATE: 1},
            )
        self.assertEqual(
            events,
            ["defaults", ("setsockopt", zmq.CONFLATE, 1), "connect"],
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
