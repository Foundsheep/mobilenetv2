import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader

from layers import MobileNetV2
from layers_v1 import MobileNet
from config import *

from datetime import datetime
from time import time
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams


class DataProvider:
    def __init__(self):
        self.root = ROOT
        self.transform = v2.Compose([
            ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def load_cifar10(self):
        train_dataset = CIFAR10(root=self.root, train=True, download=True, transform=self.transform)
        test_dataset = CIFAR10(root=self.root, train=False, download=True, transform=self.transform)
        return train_dataset, test_dataset


def train_loop(mobilenet_v1, mobilenet_v2, dataloader_train, loss_fn):
    history = {"loss_v1": [],
               "loss_v2": [],
               "time_v1": [],
               "time_v2": []}
    size = len(dataloader_train.dataset)
    num_batches = len(dataloader_train)
    time_v1 = 0
    time_v2 = 0
    adam_v1 = optim.Adam(mobilenet_v1.parameters(), lr=LEARNING_RATE)
    adam_v2 = optim.Adam(mobilenet_v2.parameters(), lr=LEARNING_RATE)

    mobilenet_v1.train()
    mobilenet_v2.train()
    for batch_idx, (X, y) in enumerate(dataloader_train):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        start_v1 = time()
        pred_v1 = mobilenet_v1(X)
        loss_v1 = loss_fn(pred_v1, y)
        loss_v1.backward()
        adam_v1.step()
        adam_v1.zero_grad()
        end_v1 = time()
        time_v1 += (end_v1 - start_v1)

        start_v2 = time()
        pred_v2 = mobilenet_v2(X)
        loss_v2 = loss_fn(pred_v2, y)
        loss_v2.backward()
        adam_v2.step()
        adam_v2.zero_grad()
        end_v2 = time()
        time_v2 += (end_v2 - start_v2)

        history["loss_v1"].append(loss_v1.item())
        history["time_v1"].append(time_v1)
        history["loss_v2"].append(loss_v2.item())
        history["time_v2"].append(time_v2)

        if batch_idx % 10:
            loss_v1_item = loss_v1.item()
            loss_v2_item = loss_v2.item()
            print(f"=========================\n"
                  f"--- loss_v1 : {loss_v1_item:>5f}\n"
                  f"--- loss_v2 : {loss_v2_item:>5f}\n"
                  f"--- [{(batch_idx + 1) * len(X):>5d} / {size:>5d}]"
                  f" time took : v1 = [{time_v1:0.3f}s] , v2 = [{time_v2:0.3f}]\n"
                  f"--- speed efficiency [{(time_v1 - time_v2) / time_v1 * 100 :0.2f}%] increased\n")

            time_v1, time_v2 = 0, 0
    return history


def test_loop(mobilenet_v1, mobilenet_v2, dataloader_test, loss_fn):
    history = {"accuracy_v1": [], "accuracy_v2": []}
    mobilenet_v1.eval()
    mobilenet_v2.eval()
    size = len(dataloader_test.dataset)
    num_batches = len(dataloader_test)
    test_loss_v1 = 0
    test_loss_v2 = 0
    correct_v1 = 0
    correct_v2 = 0
    with torch.no_grad():
        for X, y in dataloader_test:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred_v1 = mobilenet_v1(X)
            test_loss_v1 += loss_fn(pred_v1, y).item()
            correct_v1 += (pred_v1.argmax(1) == y).type(torch.float).sum().item()

            pred_v2 = mobilenet_v2(X)
            test_loss_v2 += loss_fn(pred_v2, y).item()
            correct_v2 += (pred_v2.argmax(1) == y).type(torch.float).sum().item()

    print(f"=========================="
          f"--- test_loss_v1 : {test_loss_v1/size:>5f}\n"
          f"--- test_loss_v2 : {test_loss_v2/size:>5f}\n"
          f"--- correct_v1 : {correct_v1:>5f}\n"
          f"--- correct_v2 : {correct_v2:>5f}\n"
          f"--- accuracy_v1 : {correct_v1 / num_batches}\n"
          f"--- accuracy_v2 : {correct_v2 / num_batches}")

    history["accuracy_v1"].append(correct_v1 / num_batches)
    history["accuracy_v2"].append(correct_v2 / num_batches)
    return history


def run():
    print(f"DEVICE : {DEVICE}")
    dp = DataProvider()
    train_dataset, test_dataset = dp.load_cifar10()
    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()

    mobilenet_v1 = MobileNet(WIDTH_MULTIPLIER, RESOLUTION_MULTIPLIER, is_mobile=True, num_classes=len(train_dataset.classes))
    mobilenet_v1.to(DEVICE)
    mobilenet_v2 = MobileNetV2(class_num=len(train_dataset.classes))
    mobilenet_v2.to(DEVICE)

    total_history = {"loss_v1": [],
                     "loss_v2": [],
                     "time_v1": [],
                     "time_v2": [],
                     "accuracy_v1": [],
                     "accuracy_v2": []}
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n")
        history_train = train_loop(mobilenet_v1, mobilenet_v2, dataloader_train, loss_fn)
        history_test = test_loop(mobilenet_v1, mobilenet_v2, dataloader_test, loss_fn)
        total_history["loss_v1"].extend(history_train["loss_v1"])
        total_history["loss_v2"].extend(history_train["loss_v2"])
        total_history["time_v1"].extend(history_train["time_v1"])
        total_history["time_v2"].extend(history_train["time_v2"])
        total_history["accuracy_v1"].extend(history_test["accuracy_v1"])
        total_history["accuracy_v2"].extend(history_test["accuracy_v2"])
    print("========= END ==============")

    plot(total_history, len(dataloader_train))


def plot(history, skip_number):
    rcParams["figure.figsize"] = 15, 16
    plt.subplot(311)

    plt.plot(history["loss_v1"], label="loss_mobilenet_v1")
    plt.plot(history["loss_v2"], label="loss_mobilenet_v2")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("loss comparison")
    plt.legend()

    plt.subplot(312)
    plt.plot(history["accuracy_v1"], label="accuracy_mobilenet_v1")
    plt.plot(history["accuracy_v2"], label="accuracy_mobilenet_v2")
    plt.ylabel("accuracy")
    plt.xlabel("iteration")
    plt.title("accuracy comparison")
    plt.legend()

    plt.subplot(313)
    plt.plot(history["time_v1"][skip_number:], label="time_mobilenet_v1")   # remove the first its, as they tend to be outliers
    plt.plot(history["time_v2"][skip_number:], label="time_mobilenet_v2")
    plt.ylabel("seconds")
    plt.xlabel("iteration")
    plt.title("time comparison")
    plt.legend()

    now = datetime.now(tz=SEOUL_TZ).strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./plot_{now}.jpg")


if __name__ == "__main__":
    run()