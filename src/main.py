# src/main.py

import torch
from torchvision import transforms
from PIL import Image
from src.classify_plants import PlantClassifier
from src.generate_music import generate_midi
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

if __name__ == "__main__":
    os.makedirs("output_music", exist_ok=True)

    # 🔥 Randomly pick a flower image
    image_list = sorted(glob.glob("data/flowers/jpg/image_*.jpg"))
    image_path = random.choice(image_list) if image_list else None

    if image_path is None:
        print("❌ No flower images found in data/flowers/jpg/")
        exit()

    # Save the selected flower image
    copyfile(image_path, "output_music/used_flower.jpg")
    print(f"🌸 Selected flower: {os.path.basename(image_path)}")

    # 🔥 Predict traits
    traits = predict_traits(image_path)

    # 🔥 Make traits more dynamic
    traits = traits * 10.0  # Amplify traits
    traits = traits + (torch.randn_like(torch.tensor(traits)) * 0.05).numpy()  # Add slight noise

    # 🔥 Print traits
    print("🔮 Predicted traits (first 30):", traits[:30])

    # 🔥 Generate MIDI from traits
    generate_midi(seed_sequence=traits[:30], output_file="output_music/music1.mid")

    print("✅ Music generated and saved to output_music/music1.mid")
