import os
import numpy as np
from tqdm import tqdm
import librosa
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

import hyperparameter as hp
from tokenizer import Tokenizer
from data_hdf5 import HDF5DatasetWriter
from audio import Audio

MAX_LEN_TEXT = 300
MAX_LEN_AUDIO = 1595

audio = Audio(hp)
tokenizer = Tokenizer(alphabet=hp.alphabet)

data_list = os.listdir(hp.data_path)
tokens = []

audio_links = []
label_links = []


def process_wav(wav_path):
    y, sr = audio.load_wav(wav_path)
    mel = audio.mel_spectrogram(y)
    assert mel.shape[1] == audio.config.mel_channels, len(mel.shape) == 2
    start_token = np.ones((1, hp.mel_channels)) * hp.mel_start_value
    end_token = np.ones((1, hp.mel_channels)) * hp.mel_end_value
    mel = np.concatenate([start_token, mel], 0)
    mel = np.concatenate([mel, end_token], 0)
    return mel


def pad_text(text, length=MAX_LEN_TEXT):
    while len(text) < length:
        text = np.concatenate([text, np.array([tokenizer.pad_token_index])], 0)
    return text


def pad_audio(samples, length=MAX_LEN_AUDIO):
    if len(samples) >= length:
        return samples
    else:
        while len(samples) < length:
            samples = np.concatenate([samples, np.zeros((1, hp.mel_channels))], 0)
        return samples


def pad_stop(stop, length=MAX_LEN_AUDIO):
    while len(stop) < length:
        stop = np.concatenate([stop, np.array([0])], 0)
    return stop


def filter_data(file_name):
    if '.wav' in file_name:
        _, data = wavfile.read(file_name)
        txt_file = file_name.split('.')[0] + '.txt'
        f = open(txt_file, "r")

        text = f.read()
        text = text.lower()
        text = text.strip()
        text = text.replace('\n', '')
        text = text.replace('\'', '')

        for char in text:
            if char not in hp.alphabet:
                return

        if len(text) <= MAX_LEN_TEXT + 2:
            audio_links.append(file_name)
            label_links.append(txt_file)


def read_data(wav_file, txt_file):

    f = open(txt_file, "r")
    text = f.read()
    text = text.lower()
    text = text.strip()
    text = text.replace('\n', '')
    text = text.replace('\'', '')
    text = tokenizer(text)
    text_length = len(text)
    text = pad_text(text, MAX_LEN_TEXT + 2)
    text = np.expand_dims(text, 0)

    mel = process_wav(wav_file)
    mel_length = len(mel)
    mel = pad_audio(mel, MAX_LEN_AUDIO + 2)
    mel = np.expand_dims(mel, 0)

    stop_token = np.ones((mel_length,))
    stop_token[len(stop_token) - 1] = 2
    stop_token = pad_stop(stop_token, MAX_LEN_AUDIO + 2)
    stop_token = np.expand_dims(stop_token, 0)

    return mel, text, np.array([[mel_length]]), np.array([[text_length]]), stop_token


def dum_data(audioes, labels, path):
    length = len(audioes)
    dump_data = HDF5DatasetWriter((length, MAX_LEN_AUDIO + 2, hp.mel_channels), (length, MAX_LEN_TEXT + 2), path)

    with tqdm(total=length) as pbar:
        for _audio, _label in zip(audioes, labels):
            _mel, _text, _mel_len, _text_len, _stop_token = read_data(_audio, _label)
            dump_data.add(_mel, _text, _mel_len, _text_len, _stop_token)
            pbar.update(1)


for file_name in data_list:
    filter_data(hp.data_path + '/' + file_name)

audio_train, audio_test, label_train, label_test = train_test_split(audio_links, label_links, test_size=0.1,
                                                                    random_state=42)

print(len(audio_train))
print(len(audio_test))
print(len(label_train))
print(len(label_test))

dum_data(audio_train, label_train, 'tts_train.hdf5')
dum_data(audio_test[:770], label_test[:770], 'tts_test.hdf5')

