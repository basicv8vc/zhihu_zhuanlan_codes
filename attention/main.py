# encoding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from agnew import AGNewDataset
from model import LSTMMHSAN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 200
num_classes = 4
# batch_size = 32
batch_size = 1
num_epochs = 20
learning_rate = 0.001

def train():
    dataset = AGNewDataset('./data/')
    print("training set size: {}".format(len(dataset)))
    print("test set size: {}".format(len(dataset.test)))
    
    model = LSTMMHSAN(len(dataset.word2id), input_size).to(device)
    print("model")
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (texts, labels) in enumerate(train_loader):
            texts = texts.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if (i+1) % 1000 == 0:
            #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            if i % 2000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                evaluation(model, dataset)

def evaluation(model, dataset):
    model.eval()
    test = dataset.test
    real = []
    predict = []
    for (text, label) in test:
        real.append(label)
        output = model(torch.from_numpy(np.array(text, dtype=np.int64)).unsqueeze(0).to(device))
        predict.append(torch.argmax(output).cpu().item())
    real = np.array(real)
    predict = np.array(predict)
    acc = (real.shape[0] - np.count_nonzero(real-predict)) / real.shape[0]
    print("evaluation accuracy: {}".format(acc))


    model.train()
    

if __name__ == '__main__':
    train()
