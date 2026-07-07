"""Global test config: block real network access in every test by default."""

import pytest
from pytest_socket import disable_socket


@pytest.fixture(autouse=True)
def _block_network() -> None:
    disable_socket()
