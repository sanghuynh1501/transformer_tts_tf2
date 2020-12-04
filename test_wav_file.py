import numpy as np
import librosa
import tensorflow as tf

from audio import Audio
import hyperparameter as hp
from config import Config

audio = Audio(hp)

config = Config("autoregressive")
transformer = config.get_model()
transformer.load_weights('checkpoints/ckpt-37')


checkpoint_path = "checkpoints"

ckpt = tf.train.Checkpoint(transformer=transformer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

sentence = 'xin chao cac ban'
out = transformer.predict(sentence)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
audio.save_wav(wav, "test.wav")

