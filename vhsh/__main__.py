import sys
import re
import argparse
import logging
import tomllib
from pathlib import Path
from typing import Optional

from .app import VHSh
from .types import Color


logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Color.BLUE,
        logging.INFO: Color.GREEN + Color.Style.BOLD,
        logging.WARNING: Color.Bright.YELLOW + Color.Style.BOLD,
        logging.ERROR: Color.RED + Color.Style.BOLD,
        logging.CRITICAL: \
            Color.WHITE + Color.Style.BOLD + Color.Background.RED,
    }

    def __init__(self, min_level = logging.DEBUG, *args, **kwargs):
        self._min_level_name = min_level
        super().__init__(*args, **kwargs)

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        _fmt = self._style._fmt
        try:
            self._style._fmt = re.sub(
                R"%\(levelname\)([^a-z]*)s(\s*)",
                (f"{color}%(levelname)\\1s{Color.RESET}\\2"
                 if record.levelno >= self._min_level_name
                 else ""),
                self._style._fmt
            )
            return logging.Formatter.format(self, record)
        finally:
            self._style._fmt = _fmt


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', nargs='+',
        help='Path to GLSL fragment shader', type=Path)
    parser.add_argument('-M', '--midi-mapping',
        help="Path to TOML file with system MIDI mappings")
    # TODO support seelction the microphone
    parser.add_argument('-t', '--mic', action="store_true",
        help="Make microphone levels available as uniform.")
    parser.add_argument('-v', '--verbose', action="store_true",
        help="Enable debugging output")
    parser.add_argument('-V', '--version', action="store_true",
        help="Print version information")
    return parser


def print_version():
    from . import __version__
    import os
    import imgui
    import OpenGL
    from .renderer import Renderer

    version_line = re.match("#version.*", Renderer.FRAGMENT_SHADER_PREAMBLE)
    gl_profile = version_line[0] if version_line is not None else ""

    print(f"{__package__} {__version__}  {os.path.dirname(__file__)}")
    print()
    print(f"Python {sys.version}  {sys.executable}")
    print(f"{OpenGL.__package__} {OpenGL.__version__}  {gl_profile}")
    print(f"{imgui.__package__} {imgui.__version__}")  # type: ignore


def configure_logging(verbose: bool):
    if verbose:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            ColorFormatter(
                fmt=
                f"{Color.Style.FAINT}%(asctime)s{Color.RESET}"
                f" %(levelname)-8s"
                f" %(name)s"
                f"{Color.Style.FAINT}"
                f" [%(threadName)s]"
                f" (%(filename)s:%(lineno)i)"
                f"{Color.RESET}:"
                f"\n%(message)s"
            )
        )
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[handler],
                            force=True)

    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter(fmt="%(levelname)s %(message)s",
                                            min_level=logging.WARNING))
        logging.basicConfig(level=logging.INFO,
                            handlers=[handler],
                            force=True)

        logging.getLogger('watchfiles.main').setLevel(logging.CRITICAL)


def main(argv: Optional[list[str]] = None):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    if args.version or args.verbose:
        print_version()
    if args.version:
        exit(0)

    configure_logging(args.verbose)

    midi_mapping = {}
    if args.midi_mapping:
        with open(args.midi_mapping, 'rb') as f:
            midi_mapping = tomllib.load(f)

    vhsh_renderer = VHSh(scenes=args.shader,
                         midi_mapping=midi_mapping,
                         microphone=args.mic)
    vhsh_renderer.run()


if __name__ == "__main__":
    main()
