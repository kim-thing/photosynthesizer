import pretty_midi
from pydub import AudioSegment
import os

from pydub import AudioSegment
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"  # Use `which ffmpeg` if unsure

# Tell pydub to use ffmpeg
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"  # adjust path if needed

MIDI_FILE = "output_music/music1.mid"
WAV_FILE = "output_music/music1.wav"

def convert_midi_to_wav(midi_path, wav_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio = midi_data.synthesize()

        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=44100,
            sample_width=4,  # 32-bit float
            channels=1
        )



        audio_segment.export(wav_path, format="wav")
        print(" 000 Saved to", wav_path)

    except Exception as e:
        print("xxx Error:", e)

if __name__ == "__main__":
    if os.path.exists(MIDI_FILE):
        convert_midi_to_wav(MIDI_FILE, WAV_FILE)
    else:
        print("xxx No MIDI file found.")
#loads but not fully right, i think i need a github pages