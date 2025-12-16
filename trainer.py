import copy
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from skimage.transform import radon, rescale
from scipy.ndimage import rotate
import random
from sklearn.metrics import f1_score


def test(model, test_loader, device, final_test=False):
    # test over the full rotated test set
    total = 0
    correct = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test_loader):
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            y = y.view(-1, 10)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()

            # Collect predictions and targets for F1 score calculation
            all_predictions.extend(prediction.cpu().numpy())
            all_targets.extend(t.cpu().numpy())

    f1 = f1_score(all_targets, all_predictions, average='weighted')
    if final_test:
        print(f"[Final Test] Acc: {correct/total*100.}  |  F1-Score: {f1}\n")

    return correct/total*100.0, f1


def train(model, epochs, train_loader, val_loader, test_loader, lr=1e-4, wd=1e-4, device='cuda'):
    best_acc = 0.0
    best_model = None
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        model.train()
        for i, (x, t) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            y = y.view(-1, 10)
            loss = loss_function(y, t)

            loss.backward()

            optimizer.step()
            del x, y, t, loss
        if (epoch + 1) % 1 == 0:
            accuracy, _ = test(model, val_loader,  device=device)
            print(f"epoch {epoch + 1} | validation accuracy: {accuracy}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = copy.deepcopy(model.to('cpu'))
                model = model.to('cuda')

    print(f"Max validation accuracy: {best_acc}\n")
    best_model = best_model.to('cuda')
    score, f1 = test(best_model, test_loader, device=device, final_test=True)
    del best_model
    del model
    return score, f1
