import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class FlowerDataset(Dataset):
    def __init__(self, image_dir, indices, labels, transform=None):
        self.image_dir = image_dir
        self.indices = indices
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_id = self.indices[idx]
        image_path = os.path.join(self.image_dir, f"image_{img_id + 1:05d}.jpg")
        image = Image.open(image_path).convert("RGB")





#mask still doesnt owkre 
        image_np = np.array(image)
        hsv = self.rgb_to_hsv(image_np)
        # Detect green background and mask it out
        lower_green = 35 / 360.0
        upper_green = 85 / 360.0
        mask = (hsv[:, :, 0] >= lower_green) & (hsv[:, :, 0] <= upper_green)
        image_np[mask] = [0, 0, 0]  # Set green areas to black





        # Convert back to PIL image
        masked_image = Image.fromarray(image_np)

        # Apply transforms
        image_tensor = self.transform(masked_image)
        label = self.labels[img_id] - 1  # 1-based to 0-based

        return image_tensor, label
    def rgb_to_hsv(self, rgb):
        rgb = rgb.astype('float32') / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)

        v = maxc
        s = np.divide(maxc - minc, maxc + 1e-6) # s = (maxc - minc) / maxc 
        s[maxc == 0] = 0


        #had to search this part ours was wrong 

        rc = (maxc - r) / (maxc - minc + 1e-6)
        gc = (maxc - g) / (maxc - minc + 1e-6)
        bc = (maxc - b) / (maxc - minc + 1e-6)
        h = 4.0 + (rc - gc)
        h[minc == r] = (bc - gc)[minc == r]
        h[minc == g] = 2.0 + (rc - bc)[minc == g]
        h[minc == b] = 4.0 + (gc - rc)[minc == b]
        h = (h / 6.0) % 1.0
        h[minc == maxc] = 0.0

        hsv = np.stack([h, s, v], axis=-1)
        return hsv
