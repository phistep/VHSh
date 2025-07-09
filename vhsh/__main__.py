import argparse
import tomllib
from pathlib import Path
from typing import Optional

from . import VHShRenderer


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('shader', nargs='+',
        help='Path to GLSL fragment shader', type=Path)
    parser.add_argument('-w', '--watch', action='store_true',
        help="Watch for file changes and automatically reload shader")
    parser.add_argument('-m', '--midi', action='store_true',
        help="Listen to MIDI messages for uniform control")
    parser.add_argument('-M', '--midi-mapping',
        help="Path to TOML file with system MIDI mappings")
    # TODO support seelction the microphone
    parser.add_argument('-t', '--mic', action="store_true",
        help="Make microphone levels available as uniform.")
    args = parser.parse_args(argv)

    midi_mapping = {}
    if args.midi_mapping:
        with open(args.midi_mapping, 'rb') as f:
            midi_mapping = tomllib.load(f)

    vhsh_renderer = VHShRenderer(args.shader,
                                 watch=args.watch,
                                 midi=args.midi,
                                 midi_mapping=midi_mapping,
                                 microphone=args.mic)
    vhsh_renderer.run()


if __name__ == "__main__":
    main()
