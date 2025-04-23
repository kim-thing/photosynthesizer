import os
import urllib.request

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# (URL, filename)
FILES = [
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "102flowers.tgz"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", "imagelabels.mat"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", "setid.mat"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz", "102segmentations.tgz"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/distance_matrices102.mat", "distancematrices102.mat"),
]

def download_if_missing(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"⬇️ Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ Saved: {path}")
    else:
        print(f"✅ {filename} already exists, skipping.")

for url, fname in FILES:
    download_if_missing(url, fname)
