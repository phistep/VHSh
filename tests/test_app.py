import logging
import re
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest

from vhsh.app import VHSh
from vhsh.window import Window


@pytest.fixture
def scene_source(request) -> str:
    if hasattr(request, "param"):
        return request.param
    return dedent("""\
        /// @version 1
        void main() {
            vec2 uv = (gl_FragCoord.xy * 2. - Resolution.xy) / Resolution.y;
            float t = sin(2. * Time) / 2.;
            float m = Microphone[0];
            FragColor = vec4(uv.x, t, m, 1.);
        }
    """)


@pytest.fixture
def scene_file(scene_source: str, tmp_path: Path) -> Path:
    scene_path = tmp_path / "scene.glsl"
    scene_path.write_text(scene_source)
    return scene_path


@pytest.fixture
def run_one_frame(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        Window,
        "should_close",
        Mock(spec=Window.should_close, side_effect=[False, True]),
    )


@pytest.mark.parametrize("scene_source", ["xxx"], indirect=True)
@pytest.mark.usefixtures("run_one_frame")
def test_shader_compile_error_lineno(
    scene_file: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR):
        vhsh = VHSh([scene_file])
        vhsh.run()

    match = re.search(R"\d+:(\d+)", caplog.records[0].message)
    assert match is not None
    lineno = int(match.group(1))
    assert lineno == 1
