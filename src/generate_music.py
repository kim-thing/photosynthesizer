# from midiutil import MIDIFile

# def generate_midi(seed_sequence, output_file="output_music/music.mid"):
#     midi = MIDIFile(1)
#     midi.addTempo(0, 0, 120)

#     for i, val in enumerate(seed_sequence):
#         pitch = int(60 + val * 20)  # map trait value to pitch range
#         pitch = max(30, min(pitch, 90))  # clamp within valid range
#         midi.addNote(0, 0, pitch, i, 1, 100)

#     with open(output_file, "wb") as f:
#         midi.writeFile(f)


from midiutil import MIDIFile

def generate_midi(seed_sequence, output_file="output_music/music.mid"):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)

    for i, val in enumerate(seed_sequence):
        pitch = int(60 + val * 20)  # map trait value to pitch range
        pitch = max(30, min(pitch, 90))  # clamp within valid range
        midi.addNote(0, 0, pitch, i, 1, 100)

    with open(output_file, "wb") as f:
        midi.writeFile(f)
