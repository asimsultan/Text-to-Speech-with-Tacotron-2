
# Text-to-Speech with Tacotron 2

Welcome to the Text-to-Speech with Tacotron 2 project! This project focuses on generating speech from text using the Tacotron 2 model.

## Introduction

Text-to-speech (TTS) involves converting text into natural-sounding speech. In this project, we leverage the power of Tacotron 2 to perform TTS tasks using the LJSpeech dataset.

## Dataset

For this project, we will use the LJSpeech dataset. You can download the dataset and place it in the `data/LJSpeech-1.1` directory.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- TensorFlow
- NumPy
- Librosa

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/tacotron2_tts.git
cd tacotron2_tts

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes the LJSpeech dataset. Place these files in the data/ directory.

# To train the Tacotron 2 model for text-to-speech, run the following command:
python scripts/train.py --data_path data/LJSpeech-1.1

# To evaluate the performance of the trained model, run:
python scripts/evaluate.py --model_path models/tacotron2.pth --data_path data/LJSpeech-1.1
