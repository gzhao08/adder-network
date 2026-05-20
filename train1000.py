from adderNet import MLP
from MnistDataloader import MnistDataloader

import random
import matplotlib.pyplot as plt
import numpy as np

# number of images to include in training; must be < 60,000
NUM_IMAGES = 1000

# each batch is a subset of the training data
batch_size = 16
num_epochs = 10
learning_rate = 0.02

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# flatten images into row vectors
x_train = x_train[:NUM_IMAGES]
x_train = [img.flatten() for img in x_train]

y_train = y_train[:NUM_IMAGES]


# Model:
nn = MLP(28*28, [64,10])


for epoch in range(num_epochs):

    for start in range(0, len(x_train), batch_size):
        # pick batch_size number of random indicies
        indices = random.sample(range(1, NUM_IMAGES), batch_size)

        x_batch = [x_train[i] for i in indices]
        y_batch = [y_train[i] for i in indices]

        ypred = nn(x_batch)

        # softmax
        probs = []
        for row in ypred:
            exps = [x.exp() for x in row]
            denom = sum(exps)
            probs.append([x / denom for x in exps])

        losses = [-probs[i][y_batch[i]].log() for i in range(batch_size)]
        loss = sum(losses) / batch_size
        print(loss.data)

        nn.zero_grad()
        loss.backward()
        nn.step(learning_rate)


