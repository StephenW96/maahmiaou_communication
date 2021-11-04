import os
import librosa
import soundfile

ROOT = "Path to dataset"

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if ".mp3" in file:
            new_file = file[:-4]
            y, sr = librosa.load(root+file)

            soundfile.write(new_file+".wav", y, sr)
