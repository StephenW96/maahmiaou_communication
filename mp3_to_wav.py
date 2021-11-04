import os
import librosa
import soundfile

ROOT = "/Users/mopper/Desktop/maahmiaou_communication/Happy"

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if ".mp3" in file:
            new_file = file[:-4]
            print(new_file)
            y, sr = librosa.load(root+'/'+file)

            soundfile.write('/Users/mopper/Desktop/maahmiaou_communication/output/'+new_file+".wav", y, sr)
