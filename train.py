import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader

from layers import MobileNetV2
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


def train_loop(mobilenet, mobilenet_cnn, dataloader_train, loss_fn):
    history = {"loss_m": [],
               "loss_c": [],
               "time_m": [],
               "time_c": []}
    size = len(dataloader_train.dataset)
    num_batches = len(dataloader_train)
    time_m = 0
    time_c = 0
    adam_m = optim.Adam(mobilenet.parameters(), lr=LEARNING_RATE)
    adam_c = optim.Adam(mobilenet_cnn.parameters(), lr=LEARNING_RATE)

    mobilenet.train()
    mobilenet_cnn.train()
    for batch_idx, (X, y) in enumerate(dataloader_train):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        start_m = time()
        pred_m = mobilenet(X)
        loss_m = loss_fn(pred_m, y)
        loss_m.backward()
        adam_m.step()
        adam_m.zero_grad()
        end_m = time()
        time_m += (end_m - start_m)

        start_c = time()
        pred_c = mobilenet_cnn(X)
        loss_c = loss_fn(pred_c, y)
        loss_c.backward()
        adam_c.step()
        adam_c.zero_grad()
        end_c = time()
        time_c += (end_c - start_c)

        history["loss_m"].append(loss_m.item())
        history["time_m"].append(time_m)
        history["loss_c"].append(loss_c.item())
        history["time_c"].append(time_c)

        if batch_idx % 10:
            loss_m_item = loss_m.item()
            loss_c_item = loss_c.item()
            print(f"=========================\n"
                  f"--- loss_m : {loss_m_item:>5f}\n"
                  f"--- loss_c : {loss_c_item:>5f}\n"
                  f"--- [{(batch_idx + 1) * len(X):>5d} / {size:>5d}]"
                  f" time took : m = [{time_m:0.3f}s] , c = [{time_c:0.3f}]\n"
                  f"--- speed efficiency [{(time_c - time_m) / time_c * 100 :2f}%]\n")

            time_m, time_c = 0, 0
    return history


def test_loop(mobilenet, mobilenet_cnn, dataloader_test, loss_fn):
    history = {"accuracy_m": [], "accuracy_c": []}
    mobilenet.eval()
    mobilenet_cnn.eval()
    size = len(dataloader_test.dataset)
    num_batches = len(dataloader_test)
    test_loss_m = 0
    test_loss_c = 0
    correct_m = 0
    correct_c = 0
    with torch.no_grad():
        for X, y in dataloader_test:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred_m = mobilenet(X)
            test_loss_m += loss_fn(pred_m, y).item()
            correct_m += (pred_m.argmax(1) == y).type(torch.float).sum().item()

            pred_c = mobilenet_cnn(X)
            test_loss_c += loss_fn(pred_c, y).item()
            correct_c += (pred_c.argmax(1) == y).type(torch.float).sum().item()

    print(f"=========================="
          f"--- test_loss_m : {test_loss_m/size:>5f}\n"
          f"--- test_loss_c : {test_loss_c/size:>5f}\n"
          f"--- correct_m : {correct_m:>5f}\n"
          f"--- correct_c : {correct_c:>5f}\n"
          f"--- accuracy_m : {correct_m / num_batches}\n"
          f"--- accuracy_c : {correct_c / num_batches}")

    history["accuracy_m"].append(correct_m / num_batches)
    history["accuracy_c"].append(correct_c / num_batches)
    return history


def run():
    print(f"DEVICE : {DEVICE}")
    dp = DataProvider()
    train_dataset, test_dataset = dp.load_cifar10()
    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()

    mobilenet = MobileNet(WIDTH_MULTIPLIER, RESOLUTION_MULTIPLIER, is_mobile=True, num_classes=len(train_dataset.classes))
    mobilenet.to(DEVICE)
    mobilenet_cnn = MobileNet(WIDTH_MULTIPLIER, RESOLUTION_MULTIPLIER, is_mobile=False, num_classes=len(train_dataset.classes))
    mobilenet_cnn.to(DEVICE)

    total_history = {"loss_m": [],
                     "loss_c": [],
                     "time_m": [],
                     "time_c": [],
                     "accuracy_m": [],
                     "accuracy_c": []}
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n")
        history_train = train_loop(mobilenet, mobilenet_cnn, dataloader_train, loss_fn)
        history_test = test_loop(mobilenet, mobilenet_cnn, dataloader_test, loss_fn)
        total_history["loss_m"].extend(history_train["loss_m"])
        total_history["loss_c"].extend(history_train["loss_c"])
        total_history["time_m"].extend(history_train["time_m"])
        total_history["time_c"].extend(history_train["time_c"])
        total_history["accuracy_m"].extend(history_test["accuracy_m"])
        total_history["accuracy_c"].extend(history_test["accuracy_c"])
    print("========= END ==============")

    plot(total_history, len(dataloader_train))


def plot(history, skip_number):
    rcParams["figure.figsize"] = 15, 16
    plt.subplot(311)

    plt.plot(history["loss_m"], label="loss_mobilenet")
    plt.plot(history["loss_c"], label="loss_mobilenet_cnn")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("loss comparison")
    plt.legend()

    plt.subplot(312)
    plt.plot(history["accuracy_m"], label="accuracy_mobilenet")
    plt.plot(history["accuracy_c"], label="accuracy_mobilenet_cnn")
    plt.ylabel("accuracy")
    plt.xlabel("iteration")
    plt.title("accuracy comparison")
    plt.legend()

    plt.subplot(313)
    plt.plot(history["time_m"][skip_number:], label="time_mobilenet")   # remove the first its, as they tend to be outliers
    plt.plot(history["time_c"][skip_number:], label="time_mobilenet_cnn")
    plt.ylabel("seconds")
    plt.xlabel("iteration")
    plt.title("time comparison")
    plt.legend()

    now = datetime.now(tz=SEOUL_TZ).strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./plot_{now}.jpg")


if __name__ == "__main__":
    run()