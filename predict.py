from pathlib import Path
import argparse
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision as tv


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def create_model(architecuture, hidden_units, dropout):
    model = getattr(tv.models, architecuture)(pretrained=True)

    # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad = False

    # Get number of input features for classifier
    classifier = model.classifier
    if hasattr(classifier, '__getitem__'):
        i = 0
        for i in range(len(classifier)):
            if hasattr(classifier[i], 'in_features'):
                break
        classifier = classifier[i]
    in_features = classifier.in_features

    new_classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = new_classifier
    return model


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    architecuture = checkpoint['architecture']
    dropout = checkpoint['dropout']
    hidden_units = checkpoint['hidden_units']
    model = create_model(architecuture, hidden_units, dropout)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an PyTorch Tensor
    '''
    image = transforms(image).float()
    return image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = Image.open(image_path)
    img_tensor = process_image(image)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_tensor.reshape((1, 3, 224, 224)))

    return torch.exp(output).topk(topk), img_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classifier - Predict')

    parser.add_argument('input', help='input image path', type=str)
    parser.add_argument('checkpoint', help='checkpoint path', type=str)
    parser.add_argument('-k', '--top_k', help='number of most likely classes', type=int, default=5)
    parser.add_argument('-c', '--category_names', help='path to file with category to names mapping',
                        type=str, default='cat_to_name.json')
    parser.add_argument('-g', '--gpu', help='use GPU', default=False, action='store_true')
    parser.add_argument('-t', '--true_cat', help='true category', type=str, default=None)
    args = parser.parse_args()

    print("\nImage Classifier - Predict\n")
    print("Options:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()

    img_path = Path(args.input)
    chkpt_path = Path(args.checkpoint)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # transforms for inference
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
        ])

    # Loading the checkpoint
    model = load_checkpoint(chkpt_path)

    # Class Prediction
    top_k_pred, img_tensor = predict(img_path, model, args.top_k)

    probs, class_idxs = top_k_pred
    probs = probs.data.numpy().squeeze()
    class_idxs = class_idxs.data.numpy().squeeze()
    idx_to_class = {v: k for k,v in model.class_to_idx.items()}
    class_names = [cat_to_name[idx_to_class[c]] for c in class_idxs]

    for c, p in zip(class_names, probs):
        print(f"{c:25}: {p:5.3f}")

    if args.true_cat:
        print(f"\nTrue class: {cat_to_name[args.true_cat]}")
