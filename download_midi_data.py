import kagglehub

def download_midi_kagglehub():
    path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
    print(f"---- Downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_midi_kagglehub()
