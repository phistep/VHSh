# TODO all broken since it cannot control any shader thigns.
# need to refactor shader management first, before I can extract this.
# how to rebase so to remove the commit


import time
from collections import defaultdict
from threading import Thread, Event
from pprint import pprint
from datetime import datetime

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError as __mido_import_error__:
    MIDI_AVAILABLE = False

from .types import Actions


class MIDIManager(Thread):

    def __init__(self, actions: Actions, system_mapping: dict):
        if not MIDO_AVAILABLE:
            raise __mido_import_error__

        super().__init__(name="vhsh.midi.MIDIManager")

        self.actions = actions

        print("midi system mapping:")
        pprint(system_mapping)
        self.system_mapping = defaultdict(dict, system_mapping)

        self._stop_midi = Event()

    def stop(self):
        self._stop_midi.set()

    def run(self):
        try:
            with mido.open_input() as inport:  # type: ignore
                print(f"midi: listening for MIDI messages on '{inport.name}'...")

                while True:
                    if self._stop_midi.is_set():
                        break
                    for msg in inport.iter_pending():
                        # print(f"Received MIDI message: {msg}")
                        # print(f"Received MIDI message: #{msg.control} = {msg.value}")
                        button_down = bool(msg.value)

                        if msg.control == self.system_mapping['scene'].get('prev'):
                            if button_down:
                                self.actions.prev_shader()
                            continue
                        if msg.control == self.system_mapping['scene'].get('next'):
                            if button_down:
                                self.actions.next_shader()
                            continue

                        if msg.control == self.system_mapping['preset'].get('prev'):
                            if button_down:
                                self.actions.prev_preset()
                            continue
                        if msg.control == self.system_mapping['preset'].get('next'):
                            if button_down:
                                self.actions.next_preset()
                            continue
                        if msg.control == self.system_mapping['preset'].get('save'):
                            if button_down:
                                self.actions.write_file(
                                    uniforms=False,
                                    presets=True,
                                    new_preset=f"MIDI {datetime.now()}"
                                )
                            continue

                        if msg.control == self.system_mapping['preset'].get('next'):
                            if button_down:
                                self.actions.next_preset()
                            continue

                        if msg.control == self.system_mapping['uniform'].get('time', {}).get('toggle'):
                            self.actions.set_time_running(bool(msg.value))
                            continue

                        if msg.control == self.system_mapping['uniform'].get('toggle_ui'):
                            self.actions.set_show_gui(bool(msg.value))
                            continue

                        parameter = None
                        try:
                            parameter = self.actions.get_midi_mapping(msg.control)
                            assert 0 <= msg.value <= 127
                            uniform_value = msg.value / 127.0
                            self.actions.set_parameter_value(parameter, uniform_value, normalized=True)
                        except KeyError as e:
                            print(f"MIDI mapping not found for: {msg.control}")
                            # print(msg)
                            # pprint(self._midi_mapping)
                        except NotImplementedError as e:
                            self.actions._print_error(f"ERROR setting uniform '{parameter}': {e}")
                    time.sleep(1e-6)
        except OSError:
            print("No MIDI devices found!")
