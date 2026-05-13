# Changelog

## 1.0.0 (TODO)

First stable release after a big refactoring.

### Features

- `run` subcommand
- `--midi` enabled by default (if installed)
- listen on all MIDI ports
- ranamed uniforms and versioned scene format: 1
- `migrate` subcommand migrate from scene format 0 to 1
- `import` subcommand to import from shaderdoy.com
- proper `<log>` scaling with external MIDI controlers
- human-friendly log output
- documentation and news viewer
- default testcard shader when started without arguments
- lots of bug fixes


### Development

- split up into multiple classes and files
- general architecutre overhaul with individual compoonents for
  - app, system paremeters
  - windowing
  - gui
  - rendering
  - scenes, parameters, presets
  - controlers:
    - midi
    - microphone
    - filewatch
- debug logs on `-v`
- managed with `pyproject.toml` and `uv`
- formatted and linted `ruff` 
- type checked with `ty`
- Some `pytest` tests


## 0.1.2 (2025-08-08)

- Python packaging with `pyproject.toml`
- published to PyPI


## 0.0.1 (2025-08-08)

- Removed forced full-screen
- Remove hard-coded MIDI controller name


## 2024-renate (2025-01-01)

First exhibited publicly at _Wilde Renate: Last NYE 2024/25_
