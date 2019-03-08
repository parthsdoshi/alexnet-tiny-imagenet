import os
import argparse

import torch
import torch.nn as nn
import torch.optim as O
import torch.functional as F
import torchvision as tv

from nn_helpers import ImagePathDataset, TinyImageNetValSet, createAlexNet, train, graphTrainOutput, copyLayerWeightsExceptLast

parser = argparse.ArgumentParser(
    description='Train AlexNet on Tiny Imagenet 200.')
parser.add_argument('--data', nargs=1, help='dataset directory',
                    dest='data_path', default='./data/tiny-imagenet-200')
parser.add_argument('--save', nargs=1, help='directory to save the model',
                    dest='model_path', default='./saved_models')
args = parser.parse_args()

data_path = os.path.normpath(args.data_path)
model_path = os.path.normpath(args.model_path)
os.makedirs(model_path, exist_ok=True)

image_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = tv.datasets.ImageFolder(os.path.join(
    data_path, 'train'), transform=image_transforms)
train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=64)

val_dataset = TinyImageNetValSet(os.path.join(
    data_path, 'val'), transform=image_transforms)
val_loader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, batch_size=1000)

n_samples_in_epoch = len(train_loader)
epochs = 2
validate_every = 521

my_alexnet = createAlexNet()
pytorch_alexnet = tv.models.alexnet(pretrained=True)
copyLayerWeightsExceptLast(pytorch_alexnet, my_alexnet, requires_grad=False)

model_path_with_name = os.path.join(model_path, 'alexnet.pth')
o = train(my_alexnet, O.Adam, train_loader, val_loader, epochs=epochs,
          save=True, save_path=model_path_with_name, validate_every=validate_every)
graphTrainOutput(*o, epochs=epochs, n_samples_in_epoch=n_samples_in_epoch,
                 validate_every=validate_every)