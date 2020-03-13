from pathlib import Path
import argparse
import json
from collections import OrderedDict
import numpy as np
import torch
import torchvision as tv
from torch import nn, optim


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def create_model(architecture, hidden_units, dropout, output_size):
    model = getattr(tv.models, architecture)(pretrained=True)

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
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = new_classifier
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classifier - Train')

    parser.add_argument('data_dir', help='data directory', type=str)
    parser.add_argument('-s', '--save_dir', help='output directory', type=str)
    parser.add_argument('-a', '--arch', help='network architecture', type=str, default='vgg16')
    parser.add_argument('-u', '--hidden_units', help='number of hidden units', type=int, default=512)
    parser.add_argument('-d', '--dropout', help='dropout rate', type=float, default=0.2)
    parser.add_argument('-l', '--learning_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-e', '--epochs', help='number of hidden units', type=int, default=6)
    parser.add_argument('-p', '--print_every', help='print every xth step', type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('-g', '--gpu', help='use GPU', default=False, action='store_true')
    args = parser.parse_args()

    print("\nImage Classifier - Train\n")
    print("Options:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()

    save_dir = Path(args.save_dir or ".")
    save_dir.mkdir(parents=True, exist_ok=True)
    chkpt_path = save_dir / f"checkpoint_{args.arch}_{args.hidden_units}_{args.dropout}_{args.learning_rate}_{args.epochs}.pth"

    #  Load the data
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir = data_dir / 'test'

    # transforms for the training, validation, and testing sets
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomRotation(30),
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
        ])

    valid_transforms = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
        ])

    test_transforms = valid_transforms

    # Load the datasets with ImageFolder
    train_data = tv.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = tv.datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = tv.datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    output_size = len(cat_to_name)

    # Build model
    model = create_model(args.arch, args.hidden_units, args.dropout, output_size)

    # Train network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    # Use GPU if it's requested and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device);
    model.train()

    steps = 0
    running_loss = 0

    for epoch in range(args.epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % args.print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{args.epochs}.. "
                      f"Train loss: {running_loss / args.print_every:.3f}.. "
                      f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    # Testing network
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_loader):.3f}.. "
          f"Test accuracy: {accuracy/len(test_loader):.3f}")

    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture' : args.arch,
                  'input_size': 224,
                  'dropout': args.dropout,
                  'hidden_units': args.hidden_units,
                  'output_size': output_size,
                  'learning_rate': args.learning_rate,
                  'epochs': args.epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, chkpt_path)
