import scipy.io
import tarfile
import os

# === Paths ===
DATA_DIR = "data"
FLOWER_ARCHIVE = os.path.join(DATA_DIR, "102flowers.tgz")
LABELS_FILE = os.path.join(DATA_DIR, "imagelabels.mat")
SETID_FILE = os.path.join(DATA_DIR, "setid.mat")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "flowers")

def prepare_dataset():
    # === Step 1: Extract flowers.tgz ===
    if not os.path.exists(EXTRACTED_IMG_DIR):
        print("Extracting flower images...")
        with tarfile.open(FLOWER_ARCHIVE, "r:gz") as tar:
            tar.extractall(path=EXTRACTED_IMG_DIR)
        print("Extraction complete.")

    # === Step 2: Load labels ===
    print("Loading labels...")
    labels = scipy.io.loadmat(LABELS_FILE)["labels"][0]
    print(f"Loaded {len(labels)} labels")

    # === Step 3: Load dataset splits ===
    print("Loading data splits...")
    splits = scipy.io.loadmat(SETID_FILE)
    train_idx = splits["trnid"][0] - 1  # convert from 1-based to 0-based
    val_idx = splits["valid"][0] - 1
    test_idx = splits["tstid"][0] - 1

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    return labels, train_idx, val_idx, test_idx

if __name__ == "__main__":
    prepare_dataset()

#had to ask for alot of help on this one 