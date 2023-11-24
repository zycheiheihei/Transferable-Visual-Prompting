import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np


def get_index(train_set, model, device):

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=12)

    #print(len(train_set))

    # append every
    num_of_classes = len(train_set.classes)
    d = [[] for x in range(0, num_of_classes)]
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        d[labels].append(images)
    model.eval()

    index = []
    with torch.no_grad():

        for i in range(0, num_of_classes):
            print('now is on the class ', i)
            each_class_images = d[i]
            
            arg_index = []
            for j in range(int(len(train_set) / num_of_classes)):
                images = each_class_images[j].to(device)
                probs = model(images)
                arg_index.append(torch.argmax(probs, dim=-1))
            arg_index = torch.tensor(arg_index)
            index.append(torch.mode(arg_index, -1))

    
    d = []
    return index
