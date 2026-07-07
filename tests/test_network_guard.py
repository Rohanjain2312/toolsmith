"""Prove the global network guard blocks real socket connections."""

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_socket_connect_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
