import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FlowerDataset(Dataset):
    def __init__(self, image_dir, indices, labels, transform=None):
        self.image_dir = image_dir
        self.indices = indices
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_id = self.indices[idx]
        image_path = os.path.join(self.image_dir, f"image_{img_id + 1:05d}.jpg")
        image = Image.open(image_path).convert("RGB")
        label = self.labels[img_id] - 1  # 1-based to 0-based
        return self.transform(image), label
