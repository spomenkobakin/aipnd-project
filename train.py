import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def get_input_args():
    """
    Parse command line arguments for training the model.
    """
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')

    # Required argument
    parser.add_argument('data_dir', type=str, help='Path to the data directory')

    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--arch', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'vgg13', 'vgg16', 'resnet18'],
                        help='Model architecture (default: mobilenet_v2)')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='Learning rate (default: 0.003)')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units (default: 512)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')

    return parser.parse_args()

def load_data(data_dir):
    """
    Load and transform the training, validation, and test datasets.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }

    return dataloaders, image_datasets

def build_model(arch, hidden_units, num_classes=102):
    """
    Build the model with the specified architecture.
    """
    # Load pre-trained model
    if arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='DEFAULT')
        input_size = 1280
    elif arch == 'vgg13':
        model = models.vgg13(weights='DEFAULT')
        input_size = 25088
    elif arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        input_size = 25088
    elif arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        input_size = 512
    else:
        raise ValueError(f"Architecture {arch} not supported")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build classifier
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )

    # Replace classifier based on architecture
    if arch.startswith('vgg'):
        model.classifier = classifier
    elif arch == 'mobilenet_v2':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier

    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    """
    Train the model and validate after each epoch.
    """
    print(f"\nTraining on {device}")
    print("=" * 60)

    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_loss = running_loss / len(dataloaders['train'])
        valid_loss = valid_loss / len(dataloaders['valid'])
        valid_accuracy = accuracy / len(dataloaders['valid'])

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Valid loss: {valid_loss:.3f}.. "
              f"Valid accuracy: {valid_accuracy:.3f}")

    print("=" * 60)
    print("Training complete!\n")

    return model

def save_checkpoint(model, image_datasets, arch, hidden_units, epochs, save_dir):
    """
    Save the trained model checkpoint.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'architecture': arch,
        'hidden_units': hidden_units,
        'classifier': model.classifier if hasattr(model, 'classifier') else model.fc,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'epochs': epochs
    }

    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

def main():
    """
    Main function to orchestrate the training process.
    """
    # Get command line arguments
    args = get_input_args()

    print("\n" + "=" * 60)
    print("FLOWER CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Save directory: {args.save_dir}")
    print(f"GPU enabled: {args.gpu}")

    # Set device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu and not torch.cuda.is_available():
            print("\nWarning: GPU requested but not available. Using CPU instead.")

    # Load data
    print("\nLoading data...")
    dataloaders, image_datasets = load_data(args.data_dir)
    print(f"Training samples: {len(image_datasets['train'])}")
    print(f"Validation samples: {len(image_datasets['valid'])}")
    print(f"Test samples: {len(image_datasets['test'])}")

    # Build model
    print(f"\nBuilding {args.arch} model...")
    model = build_model(args.arch, args.hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    if hasattr(model, 'classifier'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, device, args.epochs)

    # Save checkpoint
    save_checkpoint(model, image_datasets, args.arch, args.hidden_units, args.epochs, args.save_dir)

    print("\nTraining session completed successfully!")

if __name__ == '__main__':
    main()
