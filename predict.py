import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json

def get_input_args():
    """
    Parse command line arguments for making predictions.
    """
    parser = argparse.ArgumentParser(description='Predict flower name from an image')

    # Required arguments
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes (default: 5)')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to category names JSON file (default: cat_to_name.json)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')

    return parser.parse_args()

def load_checkpoint(filepath, device='cpu'):
    """
    Load a checkpoint and rebuild the model.

    Args:
        filepath: Path to the checkpoint file
        device: Device to load the checkpoint to ('cpu' or 'cuda')

    Returns:
        model: The rebuilt model with loaded weights
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Get architecture from checkpoint
    arch = checkpoint['architecture']
    hidden_units = checkpoint.get('hidden_units', 512)

    # Rebuild the model based on architecture
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
        print(f"Warning: Architecture {arch} not recognized, defaulting to mobilenet_v2")
        model = models.mobilenet_v2(weights='DEFAULT')

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load the custom classifier
    if hasattr(model, 'classifier'):
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['classifier']

    # Load the state dict (weights)
    model.load_state_dict(checkpoint['state_dict'])

    # Load the class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    Args:
        image_path: Path to the image file

    Returns:
        np_image: Numpy array of processed image
    """
    # Load image
    image = Image.open(image_path)

    # Convert to RGB (removes alpha channel if present)
    image = image.convert('RGB')

    # Resize image so shortest side is 256 pixels, keeping aspect ratio
    if image.size[0] < image.size[1]:
        image.thumbnail((256, 256 * image.size[1] / image.size[0]))
    else:
        image.thumbnail((256 * image.size[0] / image.size[1], 256))

    # Center crop to 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array and scale to 0-1
    np_image = np.array(image) / 255.0

    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions: PyTorch expects (C, H, W) but PIL/numpy is (H, W, C)
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path: Path to the image file
        model: Trained model
        device: Device to run inference on (cpu or cuda)
        topk: Number of top predictions to return

    Returns:
        top_probs: Top K probabilities
        top_classes: Top K class labels
    """
    # Process the image
    img = process_image(image_path)

    # Convert to PyTorch tensor and add batch dimension
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Move to device
    img_tensor = img_tensor.to(device)
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Make prediction with no gradient calculation
    with torch.no_grad():
        output = model.forward(img_tensor)

    # Convert log probabilities to probabilities
    ps = torch.exp(output)

    # Get top K probabilities and indices
    top_probs, top_indices = ps.topk(topk, dim=1)

    # Convert to lists
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Invert class_to_idx dictionary to get idx_to_class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Convert indices to classes
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes

def load_category_names(filepath):
    """
    Load category to name mapping from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        cat_to_name: Dictionary mapping category to name
    """
    try:
        with open(filepath, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
    except FileNotFoundError:
        print(f"Warning: Category names file '{filepath}' not found.")
        print("Using class labels instead of flower names.")
        return None

def display_predictions(probs, classes, cat_to_name=None):
    """
    Display the predictions in a readable format.

    Args:
        probs: List of probabilities
        classes: List of class labels
        cat_to_name: Optional dictionary to map classes to names
    """
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    for i, (prob, cls) in enumerate(zip(probs, classes), 1):
        if cat_to_name:
            name = cat_to_name.get(cls, cls)
            print(f"{i}. {name}")
        else:
            print(f"{i}. Class {cls}")
        print(f"   Probability: {prob:.4f} ({prob*100:.2f}%)")
        if i == 1:
            print("   ★ TOP PREDICTION ★")
        print()

    print("=" * 60)

def main():
    """
    Main function to orchestrate the prediction process.
    """
    # Get command line arguments
    args = get_input_args()

    print("\n" + "=" * 60)
    print("FLOWER CLASSIFIER PREDICTION")
    print("=" * 60)
    print(f"Input image: {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top K predictions: {args.top_k}")
    print(f"Category names file: {args.category_names}")
    print(f"GPU enabled: {args.gpu}")

    # Set device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: GPU")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
        if args.gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Using CPU instead.")

    # Load checkpoint
    print("\nLoading checkpoint...")
    model = load_checkpoint(args.checkpoint, device)
    print("Checkpoint loaded successfully!")

    # Load category names if provided
    cat_to_name = load_category_names(args.category_names)

    # Make prediction
    print(f"\nAnalyzing image: {args.input}")
    probs, classes = predict(args.input, model, device, topk=args.top_k)

    # Display results
    display_predictions(probs, classes, cat_to_name)

    # Print top prediction summary
    top_class = classes[0]
    top_prob = probs[0]

    if cat_to_name:
        top_name = cat_to_name.get(top_class, top_class)
        print(f"\n✓ Predicted flower: {top_name}")
    else:
        print(f"\n✓ Predicted class: {top_class}")

    print(f"✓ Confidence: {top_prob*100:.2f}%\n")

if __name__ == '__main__':
    main()
