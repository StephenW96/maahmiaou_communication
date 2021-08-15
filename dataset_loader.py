import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import numpy as np

## Creates a native language dataset of mel spectrograms and labels for training classifier
class AudioDataset(Dataset):

    def __init__(self, dataframe, audio_dir, transformation, target_sample_rate, num_samples, hop_length, device):
        # file with path, name, labels

        # self.annotations = dataframe.iloc[200:210]
        self.annotations = dataframe

        # path from cslu to each lang and speaker
        self.audio_dir = audio_dir
        # cpu or cuda
        self.device = device
        # mel spectrogram/ mfcc
        self.transformation = transformation.to(self.device)
        # 8000
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.hop_length = hop_length


    def __len__(self):
        # how much data
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        # signal -> (num_channels, samples) -> (2, 16000) --> (1, 16000) ---> look for my data
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        # Cut audio into 1 second chunks
        signal_list = self._cut_and_pad(signal)

        # signal = self._right_pad_if_necessary(signal)

        # transformation to MelSpec or MFCC for each element in signal_list

        signal_list = [self.transformation(i) for i in signal_list]
        # print(signal_list)
        # print(signal_list[0].shape)

        # signal_list = torch.stack(signal_list)
        # signal_list = torch.unsqueeze(signal_list, dim=0)

        return signal_list, label

    def _cut_and_pad(self, signal):
        signal_list = []

        # First 1sec chunk
        signal_list.append(signal[:, :self.num_samples])

        prev_cut = self.hop_length

        # Slice chunks of 1sec with overlap of hop size
        while prev_cut + self.num_samples <= signal.shape[1]:
            signal_list.append(signal[:, prev_cut:prev_cut+self.num_samples])
            prev_cut += self.hop_length

        # Add leftover signal (only if there is more than one chunk - so to not add 1st chunk twice)
        if signal.shape[1] - prev_cut > 0 and len(signal_list) > 1:
            signal_list.append(signal[:, prev_cut:])

        # Pad leftover signal with zeros to fit sample rate
        if signal_list[-1].shape[1] < self.num_samples:
            padding = torch.zeros(signal_list[-1].shape[0], self.num_samples-signal_list[-1].shape[1]).to(self.device)
            signal_list[-1] = torch.cat([signal_list[-1], padding], dim=1)

        return signal_list



    # def _right_pad_if_necessary(self, signal):
    #     length_signal = signal.shape[1]
    #     if length_signal < self.num_samples:
    #         # [1, 1, 1] -> [1, 1, 1, 0, 0]
    #         num_missing_samples = self.num_samples - length_signal
    #         last_dim_padding = (0, num_missing_samples)
    #         signal = torch.nn.functional.pad(signal, last_dim_padding)
    #
    #     for el in signal_list:
    #         if len(el) < len(num_samples)
    #         # add 0s to match
    #
    #     return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        #specific to annotation file -> coords change depending on this
        folders = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, folders, self.annotations.iloc[index, 1])
        return path

    def _get_audio_sample_label(self, index) :
        #for my data -> first 2 characs of each file is its class, e.g. FR, HU, etc.
        return self.annotations.iloc[index, 2]

if __name__ == "__main__":
    ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/foreign_data.csv'
    AUDIO_DIR = '/group/corporapublic/cslu_22_lang/speech/'
    SAMPLE_RATE = 8000
    NUM_SAMPLES = 8000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=64
    )

    nld = NatLangsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f'There are {len(nld)} samples in the dataset.')
    print(nld[0])

    signal, label = nld[0]
