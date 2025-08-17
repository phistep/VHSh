import os
import textwrap
import logging
from pprint import pformat
from argparse import Namespace
from urllib.parse import urlparse
from pathlib import Path
from enum import StrEnum
from urllib.parse import urlparse

from .types import Color


logger = logging.getLogger(__name__)


def import_shadertoy(url: str, outfile: Path | None = None):
    import requests

    API_URL="https://www.shadertoy.com/api/v1/shaders/"
    VIEW_URL="https://www.shadertoy.com/view/"
    API_KEY="Nt8jhz"
    api_key = os.getenv('VHSH_API_KEY_SHADERTOY', API_KEY)

    if url.isalnum():
        id_ = url
    else:
        id_ = urlparse(url).path.split('/')[-1]

    url = f"// {VIEW_URL}{id_}"

    r = requests.get(API_URL + id_,
                     params={"key": api_key},
                     headers={"user-agent": "vhsh/0.1.0"})
    r.raise_for_status()
    shadertoy_info = r.json()
    logger.debug(pformat(shadertoy_info))

    info = shadertoy_info['Shader']['info']
    header = '\n'.join(
        (f"{Color.Style.BOLD}{k.title():>12}:{Color.RESET}"
         f" {v.replace('\n', ' ') if isinstance(v, str) else v}")
        for k, v in info.items()
    )
    logger.info(header)

    header = '\n'.join(
        f"// {k}: {v.replace('\n', '\n//   ') if isinstance(v, str) else v}"
        for k, v in info.items()
    )

    src = next(
        filter(lambda rp: rp['name'] == 'Image',
               shadertoy_info['Shader']['renderpass'])
    )['code']

    # uniform vec3      iResolution;           // viewport resolution (in pixels)
    # uniform float     iTime;                 // shader playback time (in seconds)
    # uniform float     iTimeDelta;            // render time (in seconds)
    # uniform float     iFrameRate;            // shader frame rate
    # uniform int       iFrame;                // shader playback frame
    # uniform float     iChannelTime[4];       // channel playback time (in seconds)
    # uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
    # uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
    # uniform samplerXX iChannel0..3;          // input channel. XX = 2D/Cube
    # uniform vec4      iDate;                 // (year, month, day, time in seconds)
    # uniform float     iSampleRate;           // sound sample rate (i.e., 44100)

    # TODO maybe arrays need to be defined as variables?
    # TODO maybe literals should be assigned to const variables?
    adapters = textwrap.dedent("""\
        #define iResolution vec3(Resolution, 0.0)
        #define iTime Time
        #define iTimeDelta 0.0
        #define iFrameRate 60.0
        #define iFrame (60.0 * Time)
        #define iChannelTime float[](Time, Time, Time, Time)
        #define iChannelResolution float[](vec3(Resolution, 0.0), vec3(Resolution, 0.0), vec3(Resolution, 0.0), vec3(Resolution, 0.0))
        #define iMouse vec4(0.0)
        // uniform samplerXX iChannel0..3; // input channel. XX = 2D/Cube
        #define iDate vec4(1970.0, 1.0, 1.0, 0.0)
        #define iSampleRate 44100.0
    """)

    main_func = textwrap.dedent("""\
        void main() {
            vec4 frag_color;
            mainImage(frag_color, gl_FragCoord.xy);
            FragColor = frag_color;
        }
    """)

    if outfile is None:
        safe_name = ''.join(c for c in info['name'].replace(' ', '-')
                            if c.isalnum() or c in ['_', '-'])
        outfile = Path(f"{info['id']}_{safe_name}.glsl")

    with open(outfile, 'w') as f:
        f.write('\n\n'.join([url, header, adapters, src, main_func]))
    logger.info(f"wrote '{outfile}'")


def import_scene(url: str, outfile: Path):
    parsed_url = urlparse(url)
    match parsed_url.hostname:
        case 'shadertoy.com' | 'www.shadertoy.com':
            logger.debug("matched shadertoy.com")
            import_shadertoy(url, outfile)
        case _:
            raise ValueError(f"{parsed_url.hostname} not supported")
