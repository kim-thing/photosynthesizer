from midiutil import MIDIFile
import random

def generate_midi(seed_sequence, output_file="output_music/music1.mid"):
    midi = MIDIFile(1)  # Single track
    track = 0
    time = 0  # Start time
    tempo = random.randint(80, 140)  # 🎵 Random tempo (slow or fast)
    midi.addTempo(track, time, tempo)

    # 🔥 Randomly pick instrument (MIDI program numbers)
    instruments = {
        "Piano": 0,
        "Guitar": 24,
        "Violin": 40,
        "Flute": 73,
        "Trumpet": 56
    }
    instrument_name, instrument_prog = random.choice(list(instruments.items()))
    midi.addProgramChange(track, 0, time, instrument_prog)
    print(f"🎸 Using instrument: {instrument_name}")

    # 🔥 Use a real scale (C major scale notes)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

    for val in seed_sequence:
        pitch_idx = int(abs(val)) % len(scale)
        pitch = scale[pitch_idx]

        # 🔥 Add some randomness to octave
        pitch += random.choice([0, 12, -12])

        velocity = random.randint(60, 127)  # how loud
        duration = random.choice([0.5, 1.0, 1.5])  # how long note holds

        midi.addNote(track, 0, pitch, time, duration, velocity)
        time += duration  # move time forward

    # Save MIDI file
    with open(output_file, "wb") as f:
        midi.writeFile(f)
