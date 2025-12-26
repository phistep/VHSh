import shutil
import logging
from pathlib import Path
from datetime import datetime

from .scene import Scene
from .app import VHSh
from .types import Color

# TODO move _backup, migrate to scene.Scene methods


logger = logging.getLogger(__name__)


def migrate_v1(source: str) -> str:
    source = "/// @version 1\n" + source
    source = source.replace("u_Resolution", "Resolution")
    source = source.replace("u_Time", "Time")
    source = source.replace("u_Microphone", "Microphone")
    return source


MIGRATIONS = {
    1: migrate_v1,
}

def _backup(path: Path,
            backup: Path | None = None,
            version: int | None = None) -> Path:
    if backup is None:
        if version is None:
            version = path.metadata.get('version', 0)
        backup = path.parent / (f"{path.name}"
                                f".{version}"
                                f".{datetime.now().isoformat()}"
                                f".bkp")
    shutil.copy2(path, backup)
    return backup


def _get_version(source: str) -> int:
    # TODO custom regex search
    return Scene._load_metadata(source).get("version", 0)


def _set_version(source: str, version: int):
    if "/// @version" not in source:
        source = f"/// @version {version}\n\n" + source
    else:
        source = "\n".join(
            f"/// @version {version}" if line.startswith("/// @version")
            else line
            for line in source.splitlines()
        ) + '\n'
    return source


def migrate(path: Path,
            from_version: int | None = None,
            to_version: int | None = None) -> Scene:
    if from_version is None:
        from_version = _get_version(path.read_text())
    if to_version is None:
        to_version = VHSh.SCENE_FORMAT_VERSION

    original_backup = None

    current_version = from_version
    while current_version < to_version:
        logging.info("Migrating from %i to %i...", current_version, to_version)

        backup_path = _backup(path, version=current_version)
        logging.info("Backup written to %s", backup_path)
        # TODO original backup to same dir, other backups to TMP
        if original_backup is None:
            original_backup = backup_path

        next_version = current_version + 1
        try:
            current_source = path.read_text()

            try:
                migrated_source = MIGRATIONS[next_version](current_source)
            except KeyError:
                logger.debug("No migration defined for %i. Skipping!", next_version)
            current_version += 1
            migrated_source = _set_version(migrated_source, current_version)
            path.write_text(migrated_source)
            logger.info(f"{Color.GREEN + Color.Style.BOLD}Migration successful!{Color.RESET}  {Color.Style.FAINT}[%s]{Color.RESET}", path)
        except Exception as e:
            shutil.copy2(backup_path, path)
            logger.critical("Migration from version %i to version %i failed.!"
                            "Restored backup from '%s' to '%s'!",
                            current_version, next_version,
                            path.absolute(), backup_path.absolute())
            raise e

    logging.info("All migrations done.")

    try:
        scene = Scene(path, required_version=to_version)
    except Exception as e:
        shutil.copy2(original_backup, path)
        logger.critical("Migration from version %i to version %i failed.!"
                        "Restored backup from '%s' to '%s'!",
                        from_version, to_version,
                        path.absolute(), original_backup.absolute())
        raise e

    logging.info("Scene read successfully. Migration successful!")
    return scene
