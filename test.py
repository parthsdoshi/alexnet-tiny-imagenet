import os
import argparse

import torch
import torch.nn as nn
import torchvision as tv

import cv2
import matplotlib.pyplot as plt

from nn_helpers import createAlexNet, TinyImageNetValSet, camTest

parser = argparse.ArgumentParser(
    description='Test model on Tiny Imagenet 200.')
parser.add_argument('--model', nargs=1, help='directory of the saved model',
                    dest='model_path', default='./saved_models')
parser.add_argument('--data', nargs=1, help='dataset directory',
                    dest='data_path', default='./data/tiny-imagenet-200')
args = parser.parse_args()

data_path = os.path.normpath(args.data_path)
model_path = os.path.normpath(args.model_path)
model_file_path = os.path.join(model_path, 'alexnet.pth')

model = createAlexNet()
model.load_state_dict(torch.load(
    model_file_path, map_location='cpu'))
model.eval()

image_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
])
# inverse_normalize_transform = tv.transforms.Normalize(
#     mean=[-0.485 * 0.229, -0.456 * 0.224, -0.406 * 0.255],
#     std=[1/0.229, 1/0.224, 1/0.255]
# )
inverse_normalize_transform = tv.transforms.Compose([ tv.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                tv.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

labels = {}
labels_map_path = os.path.join(data_path, 'words.txt')
with open(labels_map_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        code, label = line.split('\t')
        code = code.strip()
        label = label.strip()
        labels[code] = label

subset_labels_map_path = os.path.join(data_path, 'wnids.txt')
subset_labels = []
with open(subset_labels_map_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        l = line.strip()
        if len(l) == 0:
            continue
        subset_labels.append(l)

subset_labels = sorted(subset_labels)

for i in range(len(subset_labels)):
    sl = subset_labels[i]
    subset_labels[i] = labels[sl]

# val_dataset = TinyImageNetValSet(os.path.join(
#     './data/tiny-imagenet-200', 'val'), transform=image_transforms)
# val_loader = torch.utils.data.DataLoader(
#     val_dataset, shuffle=False, batch_size=1)
# with torch.no_grad():
#     for j, (x, y) in enumerate(val_loader):
#         if j == 4:
#             break
#         y_hat = torch.argmax(model(x), dim=1)
#         print(f'y label: {y}\tmodel label: {y_hat}')
#         print(f'y text: {subset_labels[y]}\tmodel text: {subset_labels[y_hat]}')
#         x_inv = inverse_normalize_transform(x[0])
#         x_img = (x_inv.numpy().transpose(1,2,0) * 255)
#         plt.imshow(x_img)
#         plt.show()

camTest(model, image_transforms, subset_labels)