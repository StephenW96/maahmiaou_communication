import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from sklearn import metrics
import pandas as pd
import time
from dataset_loader import AudioDataset
from CNN_model import CNNetwork
import pickle as pk
import random
import numpy as np

# Set Random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#### Training Model

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

# Train file with labels
#TR_ANNOTATIONS_FILE = path to label file

# Train audio directory
#TR_AUDIO_DIR = path to audio directory

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000


def train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device):

    loss_sum = 0

    i = 0
    targets_total = []
    predictions_total = []

    for input, target in train_dataloader:

        # put data to device
        input = input.to(device)
        target = target.to(device)

        # calculate loss
        prediction = model(input.to(device))
        loss = loss_fn(prediction, target).to(device)

        # accumulate loss for average
        loss_sum += loss

        # argmax of classes = prediction
        prediction_acc = torch.argmax(prediction, dim=1).to(device)

        # Convert prediction tensor to np array and concatenate into list of predictions
        prediction_acc = prediction_acc.cpu()
        prediction_acc = prediction_acc.numpy()
        predictions_total += list(prediction_acc)

        # Convert target tensor to np array and concatenate into list of targets
        target = target.cpu()
        target_acc = target.numpy()
        targets_total += list(target_acc)

        # print batch number
        i += 1
        print(i)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    train_loss = loss_sum/len(train_dataloader)
    print(f"Training loss: {train_loss}")

    print(metrics.classification_report(targets_total, predictions_total))

    # Accuracy score for the epoch (train)
    train_acc = metrics.accuracy_score(targets_total, predictions_total)


    loss_val_sum = 0
    targets_val_total = []
    predictions_val_total = []
    i = 0
    with torch.no_grad():
        for input_val, target_val in val_dataloader:

            # put data to device
            input_val = input_val.to(device)
            target_val = target_val.to(device)

            # calculate loss
            prediction_val = model(input_val.to(device))
            dev_loss = loss_fn(prediction_val, target_val).to(device)

            # accumulate loss for average
            loss_val_sum += dev_loss

            # argmax of classes = prediction
            prediction_val_acc = torch.argmax(prediction_val, dim=1).to(device)

            # Convert prediction tensor to np array and concatenate into list of predictions
            prediction_val_acc = prediction_val_acc.cpu()
            prediction_val_acc = prediction_val_acc.numpy()
            predictions_val_total += list(prediction_val_acc)

            # Convert target tensor to np array and concatenate  into list of targets
            target_val = target_val.cpu()
            target_val_acc = target_val.numpy()
            targets_val_total += list(target_val_acc)

            # print batch number
            i+=1
            # print("Val",i)

            # No backpropagation for validation set
    dev_loss = loss_val_sum/len(val_dataloader)
    print(f"Dev loss: {(dev_loss)}")

    print(metrics.classification_report(targets_val_total, predictions_val_total))

    # Accuracy score for the epoch (dev)
    dev_acc = metrics.accuracy_score(targets_val_total, predictions_val_total)

    return train_loss, dev_loss, train_acc, dev_acc


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):

    min = 100000
    counter = 0
    train_loss_array = []
    dev_loss_array = []
    train_acc_array = []
    dev_acc_array = []


    for i in range(epochs):
        # Time epoch
        a = time.time()
        print(f"Epoch {i + 1}")
        # Train epoch
        train_loss, dev_loss, train_acc, dev_acc = train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device)
        b = time.time()
        print(f'Epoch {i + 1} time: {b - a}')
        print("---------------------------")

        # Convert loss tensor to scalar
        train_loss = train_loss.cpu()
        train_loss = train_loss.detach().numpy()
        dev_loss = dev_loss.cpu()
        dev_loss = dev_loss.numpy()

        # Append train & dev losses to array
        train_loss_array.append(train_loss)
        dev_loss_array.append(dev_loss)

        # Append train & dev accuracies to array
        train_acc_array.append(train_acc)
        dev_acc_array.append(dev_acc)

        # if current epochs val loss value < best loss so far --> set new best loss
        if dev_loss < min:
            min = dev_loss
            counter = 0

            # Save current best model
            torch.save(cnn_net.state_dict(), "./l1_classifier_mfcc.pth")

        # if current loss is worse than best loss --> counter
        else:
            counter +=1

        # if loss doesnt improve in 5 successive epochs end training
        if counter == 100:
            print("Finished training")
            quit()

    print("Finished training")

    with open('./l1_classifier_mfcc_losses.pk', 'wb') as f:
        pk.dump((train_loss_array, dev_loss_array), f, protocol=pk.HIGHEST_PROTOCOL)

    with open('./l1_classifier_mfcc_accuracies.pk', 'wb') as f:
        pk.dump((train_acc_array, dev_acc_array), f, protocol=pk.HIGHEST_PROTOCOL)

# Take data
# Sort into sizes
# take batch
# run batch through collate

def my_collate(batch):
#     class_mapping guide
#         'Angry': 0,
#         'Defence': 1,
#         'Fighting': 2,
#         'Happy': 3,
#         'HuntingMind': 4,
#         'Mating': 5,
#         'Mothercall': 6,
#         'Paining': 7,
#         'Resting': 8,
#         'Warning': 9,

    x, y = [], []
    for utterance, target in batch:
        for second in utterance:
            x.append(second)
            y.append(class_mapping[target])
    y = torch.LongTensor(y)
    # print([i.shape for i in x])
    x = torch.stack(x)

    return x, y


if __name__ == "__main__":

    # if the GPU is available - use it
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")


    hop_length = 512
    n_fft = 1024
    # Mel-spectrogram input
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64
    )

    # MFCC alternative to mel-spectrogram - calculates the MFCC on the DB-scaled Mel spectrogram
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        # 12-13 is sufficient for English - 20 for tonal langs, maybe accent info
        n_mfcc=20,
        # for alternate MFCC experiment
        # n_mfcc=12,
        melkwargs={'hop_length': hop_length,
                   'n_fft': n_fft}
    )

    # Load in data csv
    df = pd.read_csv(TR_ANNOTATIONS_FILE)
    train_sub, val_sub = [], []

    # % of data used in training
    cut = 0.90

    # Train-Validation split
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut)
        train_sub.append(el[:thres])
        val_sub.append(el[thres:])
    train_sub = pd.concat(train_sub)
    val_sub = pd.concat(val_sub)

    # Hop for chunks of audio (overlap between chunks)
    hop_length_cut = 400

    # instantiating our dataset object and create data loader
    # Training data
    train_data = NatLangsDataset(train_sub, TR_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut,
                                 device)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Validation data
    val_data = NatLangsDataset(val_sub, TR_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut,
                               device)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)


    # construct model and assign it to device
    cnn_net = CNNNetwork().to(device)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss() # L2reg
    optimiser = torch.optim.Adam(cnn_net.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # train model
    train(cnn_net, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

