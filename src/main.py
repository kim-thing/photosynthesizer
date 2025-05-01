
import torch
from torchvision import transforms
from PIL import Image
from src.classify_plants import PlantClassifier
from src.generate_music import generate_midi
from src.lstm_model import MusicLSTM
import os
import glob
import random
from shutil import copyfile

def predict_traits(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)





    model = PlantClassifier()
    model.load_state_dict(torch.load("models/plant_classifier.pt"))
    model.eval()



    with torch.no_grad():
        traits = model(input_tensor).squeeze().numpy()
    return traits



def generate_music_sequence(traits, num_notes=50):
    #  input
    input_seq = torch.tensor(traits[:30]).float().unsqueeze(0).unsqueeze(-1)  # (1, 30, 1)



    # trained LSTM
    lstm = MusicLSTM()
    lstm.load_state_dict(torch.load("models/music_lstm.pt"))
    lstm.eval()
    sequence = []




    with torch.no_grad():
        for _ in range(num_notes):
            output = lstm(input_seq)  # predict next note
            sequence.append(output.item())  # saves prediction

            # prepares next input: slide window
            next_input = output.unsqueeze(1)  # ONLY one unsqueeze (batch, 1, 1), when i searche said this was best
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

    return sequence




if __name__ == "__main__":
    os.makedirs("output_music", exist_ok=True)

    # pick random flower
    image_list = sorted(glob.glob("data/flowers/jpg/image_*.jpg"))
    image_path = random.choice(image_list) if image_list else None

    if image_path is None:
        print("xxx No flower images found in data/flowers/jpg/")
        exit()

# Save the  flower 
    copyfile(image_path, "output_music/used_flower.jpg")
    print(f"*🌸* Selected flower: {os.path.basename(image_path)}")

# Predict 
    traits = predict_traits(image_path)
    # this is for dymaic but might be overwring lstm 
    traits = traits * 10.0  # Amplify
    traits = traits + (torch.randn_like(torch.tensor(traits)) * 0.05).numpy()  # Add slight noise



    print("*🔮* Predicted traits (first 30):", traits[:30])

    # Use trained LSTM music sequence
    sequence = generate_music_sequence(traits)

#  get MIDI 
    generate_midi(seed_sequence=sequence, output_file="output_music/music1.mid")

    print("==++==Music generated and saved to output_music/music1.mid")
