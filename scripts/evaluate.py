
import torch
import argparse
import librosa
from tacotron2 import Tacotron2
from datasets import load_dataset
from utils import get_device, preprocess_data, TextToSpeechDataset

def main(model_path, data_path):
    # Load Model
    model = Tacotron2()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = load_dataset('ljspeech', data_files={'validation': data_path})
    preprocessed_datasets = dataset.map(preprocess_data, batched=True)

    # DataLoader
    eval_dataset = TextToSpeechDataset(preprocessed_datasets['validation'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = model.loss(outputs, targets)
                total_loss += loss.item()
                total_samples += 1

        avg_loss = total_loss / total_samples
        return avg_loss

    # Evaluate
    avg_loss = evaluate(model, eval_loader, device)
    print(f'Average Loss: {avg_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the LJSpeech dataset')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
