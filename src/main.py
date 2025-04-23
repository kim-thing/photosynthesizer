import torch
from torchvision import transforms
from PIL import Image
from src.classify_plants import PlantClassifier
from src.generate_music import generate_midi
import os
import glob

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

    
    image_list = sorted(glob.glob("data/flowers/jpg/image_*.jpg"))
    image_path = image_list[0] if image_list else None
#------------
    copyfile(image_path, "output_music/used_flower.jpg")
#-------------

    if image_path is None:
        print("❌ No flower images found in data/flowers/jpg/")
        exit()

    traits = predict_traits(image_path)
    print("Predicted traits:", traits[:10])  # Just show first few numbers

    generate_midi(seed_sequence=traits[:10], output_file="output_music/music1.mid")
    print("✅ MIDI saved to output_music/music1.mid")
