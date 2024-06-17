"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
from torchvision import transforms

import data_setup
import engine
import model_builder
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch model with custom hyperparameters.")
    
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader")
    parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train", help="Directory for training data")
    parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test", help="Directory for testing data")

    return parser.parse_args()

def main():
    args = parse_args()
    
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir
    
    print(f"[INFO] Training with: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, hidden_units={HIDDEN_UNITS}, learning_rate={LEARNING_RATE}")
    print(f"[INFO] Training data path: {TRAIN_DIR}")
    print(f"[INFO] Testing data path: {TEST_DIR}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )
    
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="tinyvgg_model.pth"
    )
    print(f"[INFO] Model saved to models/tinyvgg_model.pth")

if __name__ == "__main__":
    main()
