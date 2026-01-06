from pathlib import Path
from textwrap import dedent

import pytest

from vhsh.migration import MIGRATIONS, migrate
from vhsh.scene import Scene


@pytest.fixture
def scene_source() -> dict[int, str]:
    return {
        0: dedent("""\
            void main() {
                vec2 uv = (gl_FragCoord.xy * 2. - u_Resolution.xy) / u_Resolution.y;
                float t = sin(2. * u_Time) / 2.;
                float m = u_Microphone[0];
                FragColor = vec4(uv.x, t, m, 1.);
            }
        """),
        1: dedent("""\
            /// @version 1
            void main() {
                vec2 uv = (gl_FragCoord.xy * 2. - Resolution.xy) / Resolution.y;
                float t = sin(2. * Time) / 2.;
                float m = Microphone[0];
                FragColor = vec4(uv.x, t, m, 1.);
            }
        """),
    }


def test_migrate(tmp_path: Path, scene_source: dict[int, str]):
    """Test that all migrations are successfully performed."""
    scene_path = tmp_path / "scene.glsl"
    scene_path.write_text(scene_source[0])

    scene = migrate(scene_path)
    backups = list(scene_path.parent.glob(f"{scene_path.name}.*.*.bkp"))

    assert isinstance(scene, Scene)
    assert scene.source == next(reversed(scene_source.values()))  # scene_source[-1]
    assert len(backups) == len(MIGRATIONS)
    for version, backup in enumerate(sorted(backups)):
        assert backup.read_text() == scene_source[version]


@pytest.mark.parametrize(
    "to_version", range(1, len(MIGRATIONS) + 1), ids=lambda v: f"{v - 1}-{v}"
)
def test_migrate_vx(scene_source: str, to_version: int):
    from_version = to_version - 1
    migration_vx = MIGRATIONS[to_version]
    scene = migration_vx(scene_source[from_version])
    assert scene == scene_source[to_version]
