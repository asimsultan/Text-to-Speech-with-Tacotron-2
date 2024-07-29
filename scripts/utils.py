
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

class TextToSpeechDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['inputs'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item['inputs'], item['targets']

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(examples):
    # Add your preprocessing logic here
    preprocessed_data = {
        "inputs": [],
        "targets": []
    }
    for example in examples:
        # Preprocess each audio file and add to preprocessed_data
        pass
    return preprocessed_data
