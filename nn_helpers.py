import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv
from torchvision.datasets.folder import default_loader

import cv2
from PIL import Image


def y_mse(Y, Y_hat):
    # first subtracts element wise from labels
    # then squares element wise
    # then reduces over columnns so that the dims become N * 1
    se = torch.sum((Y - Y_hat) ** 2, dim=1, keepdim=True)

    # then we sum rows and divide by number of rows, N
    mse = (1. / Y_hat.shape[0]) * torch.sum(se)

    return mse

# t is of dims N * 1 where N is the batch size
# C should be the number of values for the column


def oneHotEncodeOneCol(t, C=2):
    N = t.shape[0]
    onehot = torch.Tensor([
        [0] * C
    ] * N)
    for i, v in enumerate(t):
        onehot[i, v] = 1

    return onehot

# t is of dims N * m where N is the batch size and m is the number of features
# C should be an array of how many different values there are for each of your m features
# if you do not want to one hot encode a specific feature, set that C value to 0


def oneHotEncode(t, C):
    # not implemented yet...
    pass


def train(model, optimizer, train_loader, val_loader, lr=0.001, criterion=nn.CrossEntropyLoss(reduction='mean'), epochs=50, validate_every=2000, save=False, save_path=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if save and save_path is None:
        raise AssertionError(
            'Saving is enabled but no save path was inputted.')
    
    model = model.to(device)

    # lenet5_cifar100_dev = self.net
    opt = optimizer(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(reduction="mean")
    train_cross_entropy = []
    train_accuracy = []
    validation_cross_entropy = []
    validation_accuracy = []

    best_model_accuracy = 0

    for epoch in range(epochs):
        n_correct = 0
        n_total = 0
        for i, batch in enumerate(train_loader):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            N = x.shape[0]

            # training mode (for things like dropout)
            model.train()

            # clear previous gradients
            opt.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, labels)
            loss.backward()
            opt.step()

            train_cross_entropy.append(loss)

            n_correct += (torch.argmax(y_hat, dim=1)
                          == labels).sum().item()
            n_total += N

            # evaluation mode (e.g. adds dropped neurons back in)
            model.eval()
            if i % validate_every == 0:
                n_val_correct = 0
                n_val_total = 0
                v_cross_entropy_sum = 0
                n_total_batches = len(val_loader)

                # don't calculate gradients here
                with torch.no_grad():
                    for j, v_batch in enumerate(val_loader):
                        v_x, v_labels = v_batch
                        v_x, v_labels = v_x.to(
                            device), v_labels.to(device)
                        v_N = v_x.shape[0]

                        v_y_hat = model(v_x)
                        v_loss = criterion(v_y_hat, v_labels)
                        v_cross_entropy_sum += v_loss
                        n_val_correct += (torch.argmax(v_y_hat,
                                                       dim=1) == v_labels).sum().item()
                        n_val_total += v_N

                print(
                    f"[epoch {epoch + 1}, iteration {i}] \t accuracy: {n_val_correct / n_val_total} \t cross entropy: {v_cross_entropy_sum / n_total_batches}")
                validation_accuracy.append(n_val_correct / n_val_total)
                validation_cross_entropy.append(
                    v_cross_entropy_sum / n_total_batches)
                if n_val_correct / n_val_total >= best_model_accuracy:
                    best_model_accuracy = n_val_correct / n_val_total
                    if save:
                        print(f'Saving current best model to \'{save_path}\'.')
                        torch.save(model.state_dict(),
                                   save_path)

        print(
            f"epoch {epoch + 1} accumulated train accuracy: {n_correct / n_total}")
        train_accuracy.append(n_correct / n_total)

    return (train_cross_entropy, train_accuracy, validation_cross_entropy, validation_accuracy)

def graphTrainOutput(train_cross_entropy, train_accuracy, validation_cross_entropy, validation_accuracy, epochs=2, n_samples_in_epoch=60000, validate_every=2000):
    n_samples = len(train_cross_entropy)

    y = np.array(train_cross_entropy)[0::validate_every]
    y_test = np.array(validation_cross_entropy)

    x = np.array(list(range(0, n_samples, validate_every)))
    x = x / n_samples_in_epoch

    plt.plot(x, y_test, label='validation cross entropy')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy')
    plt.title('lenet5 mnist validation loss')
    plt.legend()
    plt.show()

    plt.plot(x, y, label='train cross entropy')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy')
    plt.title('lenet5 mnist train loss')
    plt.legend()
    plt.show()

def camTest(model, transform, index_to_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # converts from cv2 to PIL
        img = Image.fromarray(img.astype('uint8'))

        # should include ToTensor
        img = transform(img)
        img = img.to(device)
        # print(img.shape)
        # img = torch.tensor(img.numpy().transpose(1, 2, 0))
        # exit()

        with torch.no_grad():
            index = torch.argmax(model(img.view(1, *img.shape)), dim=1).item()

        label = index_to_labels[index]
        cv2.putText(frame, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        cv2.imshow("webcam - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# copies model_to_copy's weights to model
# stop offset is for if the last layer is a bias/softmax/etc
def copyLayerWeightsExceptLast(model_to_copy, model, requires_grad=False, stop_offset=2):
    params1 = list(model_to_copy.named_parameters())
    params2 = list(model.named_parameters())

    for i, (name1, param1) in enumerate(params1):
        # don't copy last layer
        if i == len(params1) - stop_offset:
            break

        name2 = params2[i][0]
        param2 = params2[i][1]

        if name1 == name2:
            param2.data.copy_(param1.data)
            param2.requires_grad = requires_grad

# flattens so we can go from conv layers to linear layers
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def createLenet5(in_channels=3, init_padding=(0, 0), classes=10, activation=nn.ReLU):
    lenet5 = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=(5, 5), padding=init_padding),
        activation(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(6, 16, kernel_size=(5, 5)),
        activation(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        Flatten(),
        nn.Linear(16*5*5, 120),
        activation(),
        nn.Linear(120, 84),
        activation(),
        nn.Linear(84, classes)
    )

    return lenet5

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, init_padding=(2, 2), classes=200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=init_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def createAlexNet(in_channels=3, init_padding=(2, 2), classes=200):
    return AlexNet(in_channels, init_padding, classes)


class ImagePathDataset(tv.datasets.ImageFolder):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        path = self.imgs[i][0]
        return (x, y, path)


class TinyImageNetValSet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.root = os.path.normpath(root)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        classes_file = os.path.join(self.root, '../wnids.txt')

        self.classes = []
        with open(classes_file, 'r') as f:
            self.classes = f.readlines()
            for i in range(len(self.classes)):
                self.classes[i] = self.classes[i].strip()
        self.classes.sort()

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        images_y_file = os.path.join(self.root, 'val_annotations.txt')
        with open(images_y_file, 'r') as f:
            lines = f.readlines()

        self.class_dict = {}
        for line in lines:
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            img_filename = cols[0].strip()
            img_class = cols[1].strip()
            self.class_dict[img_filename] = img_class

        self.samples = []
        images_dir = os.path.join(self.root, 'images')
        for _root, _dirs, files in sorted(os.walk(images_dir)):
            for image_name in files:
                image_path = os.path.join(_root, image_name)
                c = self.class_dict[image_name]
                idx = self.class_to_idx[c]
                self.samples.append((image_path, idx))

    def __getitem__(self, i):
        path, target = self.samples[i]
        print(path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    # copied from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str