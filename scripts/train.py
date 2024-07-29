
import os
import torch
import argparse
from tacotron2 import Tacotron2
from datasets import load_dataset
from utils import get_device, preprocess_data, TextToSpeechDataset

def main(data_path):
    # Parameters
    model_name = 'tacotron2'
    batch_size = 32
    epochs = 50
    learning_rate = 1e-3

    # Load Dataset
    dataset = load_dataset('ljspeech', data_files={'train': data_path})

    # Preprocess Data
    preprocessed_datasets = dataset.map(preprocess_data, batched=True)

    # DataLoader
    train_dataset = TextToSpeechDataset(preprocessed_datasets['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = get_device()
    model = Tacotron2()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, 'tacotron2.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the LJSpeech dataset')
    args = parser.parse_args()
    main(args.data_path)
