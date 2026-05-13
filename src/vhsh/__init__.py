import logging
from importlib.metadata import PackageNotFoundError, version

from .app import VHSh as VHSh
from .scene import Parameter as Parameter, Scene as Scene

logger = logging.getLogger(__name__)

try:
    __version__ = version("vhsh")
except PackageNotFoundError:
    logger.debug("importlib.metadata: package not installed")
    __version__ = "0.0.0"
