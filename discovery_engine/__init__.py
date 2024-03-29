import os

from nlm_utils.utils import generate_version

from .discovery_engine import DiscoveryEngine


VERSION = generate_version(
    [
        os.path.join(os.path.dirname(__file__), "../discovery_engine/"),
        os.path.join(os.path.dirname(__file__), "../de_utils/"),
        os.path.join(os.path.dirname(__file__), "../de_server/"),
        os.path.join(os.path.dirname(__file__), "../engines/"),
    ],
)

__all__ = ("DiscoveryEngine", "VERSION")
