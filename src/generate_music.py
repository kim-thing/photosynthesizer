from midiutil import MIDIFile
import random

# Define your instrument categories
BRIGHT_INSTRUMENTS = [0, 5, 11, 12, 14]   # Piano, Electric Piano, Vibraphone, Marimba, Xylophone
DARK_INSTRUMENTS = [32, 34, 43, 44, 48]   # Acoustic Bass, Electric Bass, Contrabass, Cello, String Ensemble

def pick_instrument_and_pitch(flower_type="unknown", flower_brightness=0):
    if flower_brightness > 0:
        instrument = random.choice(BRIGHT_INSTRUMENTS)
        pitch_shift = 12  # +1 octave
    else:
        instrument = random.choice(DARK_INSTRUMENTS)
        pitch_shift = -12  # -1 octave
    return instrument, pitch_shift

def generate_midi(seed_sequence, output_file="output.mid", flower_type="unknown", flower_brightness=0):
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0
    midi.addTrackName(track, time, "Generated Track")
    midi.addTempo(track, time, 120)

    # Get instrument and pitch shift
    instrument, pitch_shift = pick_instrument_and_pitch(flower_type, flower_brightness)

    # Set the instrument
    midi.addProgramChange(0, 0, 0, instrument)

    channel = 0
    duration = 1  # 1 beat per note
    volume = 100

    for i, pitch in enumerate(seed_sequence):
        base_note = int((pitch * 50) + 60) + pitch_shift
        base_note = max(0, min(base_note, 127))  # Clamp to MIDI range
        midi.addNote(track, channel, base_note, time + i, duration, volume)

    with open(output_file, "wb") as output_file_handle:
        midi.writeFile(output_file_handle)
