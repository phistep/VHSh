import time
import logging
from collections import defaultdict
from pprint import pformat
from datetime import datetime
from contextlib import ExitStack

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError as __mido_import_error__:
    MIDO_AVAILABLE = False

from .types import App, Controller

logger = logging.getLogger(__name__)


class MIDIController(Controller):
    def __init__(self, app: App, system_mapping: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not MIDO_AVAILABLE:
            logger.warning(
                "Not listenting to MIDI, external control disabled!"
                " 'mido' not installed, install with 'vhsh[midi]'")
            self.stop()
            return

        self._app = app

        logger.info("midi system mapping:\n%s", pformat(system_mapping))
        self._system_mapping = defaultdict(dict, system_mapping)
        self._parameter_mapping: dict[int, str] = {}

        self.devices = mido.get_input_names()
        logger.info("MIDI devices: %s", self.devices)

    def _handle_message(self, msg: "mido.Message"):
        logger.debug(f"Received MIDI message: #{msg.control} = {msg.value}")
        button_down = bool(msg.value)

        if msg.control == self._system_mapping['scene'].get('prev'):
            if button_down:
                self._app.prev_scene()
        elif msg.control == self._system_mapping['scene'].get('next'):
            if button_down:
                self._app.next_scene()

        elif msg.control == self._system_mapping['preset'].get('prev'):
            if button_down:
                self._app.scene.prev_preset()
        elif msg.control == self._system_mapping['preset'].get('next'):
            if button_down:
                self._app.scene.next_preset()
        elif msg.control == self._system_mapping['preset'].get('save'):
            if button_down:
                self._app.scene.write_file(new_preset=f"MIDI {datetime.now()}")

        elif (msg.control
                == self._system_mapping['uniform'] .get('time', {}).get('toggle')):
            self._app.time.running = bool(msg.value)

        elif msg.control == self._system_mapping['uniform'].get('toggle_ui'):
            self._app.gui.visible = bool(msg.value)

        else:
            parameter = None
            try:
                parameter = self._parameter_mapping[msg.control]
                assert 0 <= msg.value <= 127
                value = msg.value / 127.0
                self._app.scene.parameters[parameter].set_value_normalized(value)

            except KeyError as e:
                logger.warning(f"MIDI mapping not found for: {msg.control}")
                logger.debug(str(msg))
                logger.debug(pformat(self._parameter_mapping))

            except NotImplementedError as e:
                logger.error(f"setting uniform '{parameter}': {e}")

    def run(self):
        # we need this if `mido` is not installed and class is not initialized
        # due to early return.
        if self._stop_controller.is_set():
            return

        try:
            with ExitStack() as stack:
                ports = [stack.enter_context(mido.open_input(device))
                         for device in self.devices]
                logger.info(
                    f"listening for MIDI messages on '{[p.name for p in ports]}'...")
                inport = mido.ports.MultiPort(ports)

                while not self._stop_controller.is_set():
                    for message in inport.iter_pending():
                        self._handle_message(message)
                    time.sleep(1e-6)
        except OSError as e:
            logger.error("No MIDI devices found!")
            logger.exception(e)

    def update_pre(self):
        # we need this if `mido` is not installed and class is not initialized
        # due to early return.
        if self._stop_controller.is_set():
            return

        self._parameter_mapping = {}
        for parameter in self._app.scene.parameters.values():
            if parameter.midi is not None:
                self._parameter_mapping[parameter.midi] = parameter.name

    def update_post(self):
        # TODO send MIDI state
        ...
