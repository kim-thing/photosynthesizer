# Photosythezier


This project uses a Convolutional Neural Network (CNN) to classify flower images and extract their visual features, which are then passed to a Long Short-Term Memory (LSTM) network to generate original music. The final output is a `.mid` file that reflects the visual characteristics of the flower.

---

## Setup Instructions (Run These in Order)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/musicPlants.git
cd musicPlants
```
### 2. Create and Activate a Virtual Environment (mac)


```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
we tried a requirments.txt file but ran into errors 

```bash
pip install torch torchvision numpy scipy midiutil scikit-learn matplotlib
```

### 4. Download/prep the Oxford 102 Flowers Dataset
```bash
python download_data.py
python src/prepare_data.py
```

### 5. Train  CNN Model Flower Classification

```bash
python src/classify_plants.py
```


### 6. Train  CNN Model Flower Classification

```bash
python train_music_lstm_from_midi.py
```

### 7. Generate Music from a Randomized Flower Image
```bash
PYTHONPATH=. python src/main.py
```
