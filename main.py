from config import Config
from data_processor import AudioProcessor
from dataset import prepare_datasets, create_dataloaders
from models import AudioCNN
import torch.optim as optim
import torch.nn as nn
from train import train_model, evaluate_model, plot_confusion_matrix
import numpy as np
import torch

def main():
    # Initialize configuration
    config = Config()
    
    # Prepare data
    train_data, val_data, test_data = prepare_datasets(config)
    
    # Process audio data
    processor = AudioProcessor(config)
    
    # Extract features (apply data augmentation to all datasets)
    print("\nProcessing training set...")
    train_features, train_labels = processor.process_dataset(train_data, augment=True)
    print("Processing validation set...")
    val_features, val_labels = processor.process_dataset(val_data, augment=True)
    print("Processing test set...")
    test_features, test_labels = processor.process_dataset(test_data, augment=True)
    
    # Print dataset sizes after augmentation
    print(f"\nDataset sizes after augmentation:")
    print(f"Training set: {len(train_features)} samples (Original {len(train_data)} + Augmented {len(train_features)-len(train_data)})")
    print(f"Validation set: {len(val_features)} samples (Original {len(val_data)} + Augmented {len(val_features)-len(val_data)})")
    print(f"Test set: {len(test_features)} samples (Original {len(test_data)} + Augmented {len(test_features)-len(test_data)})")
    
    # Calculate and print class distribution for all datasets
    train_class_dist = np.bincount(train_labels)
    val_class_dist = np.bincount(val_labels)
    test_class_dist = np.bincount(test_labels)
    
    print(f"\nClass distribution:")
    print("Training set:")
    print(f"True Story: {train_class_dist[0]} samples")
    print(f"Deceptive Story: {train_class_dist[1]} samples")
    
    print("\nValidation set:")
    print(f"True Story: {val_class_dist[0]} samples")
    print(f"Deceptive Story: {val_class_dist[1]} samples")
    
    print("\nTest set:")
    print(f"True Story: {test_class_dist[0]} samples")
    print(f"Deceptive Story: {test_class_dist[1]} samples")
    
    # Create data loaders
    train_loader = create_dataloaders(train_features, train_labels, config)
    val_loader = create_dataloaders(val_features, val_labels, config, shuffle=False)
    test_loader = create_dataloaders(test_features, test_labels, config, shuffle=False)
    
    # Initialize model using feature dimension
    input_size = train_features.shape[1]
    model = AudioCNN(input_size=input_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train model (will automatically plot training history)
    history = train_model(model, train_loader, val_loader, criterion, optimizer, config)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_acc, test_preds, test_labels, _ = evaluate_model(model, test_loader, device, criterion)
    print(f"Test set accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        test_labels, 
        test_preds,
        title=f"Confusion Matrix (Test Set)\nAccuracy: {test_acc:.4f}",
        normalize=True  # Add normalization option
    )

if __name__ == "__main__":
    main() 