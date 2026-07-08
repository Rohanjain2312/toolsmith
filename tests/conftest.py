"""Global test config: block real network access in every test by default."""

import pytest
from pytest_socket import disable_socket


@pytest.fixture(autouse=True)
def _block_network() -> None:
    # allow_unix_socket=True: asyncio's event-loop self-pipe (used internally by gradio,
    # fastmcp, and asyncio.new_event_loop() itself) is a local-only unix socketpair, not a
    # network call — blocking it produces false-positive failures in any test that merely
    # constructs a gr.Blocks() or spins up an event loop. Real TCP/UDP sockets stay blocked.
    disable_socket(allow_unix_socket=True)
