import numpy as np
from tqdm import tqdm
import tensorflow as tf

from config import Config
import hyperparameter as hp
from data_hdf5 import HDF5DatasetGenerator
from losses import new_scaled_crossentropy, masked_mean_absolute_error


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


epoch = 100
batch_size = 16

train_data = HDF5DatasetGenerator('train.hdf5', batch_size)
test_data = HDF5DatasetGenerator('test.hdf5', batch_size)

train_total = train_data.get_total_samples()
test_total = test_data.get_total_samples()

learning_rate = CustomSchedule(hp.encoder_prenet_dimension)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

config = Config("autoregressive")
transformer = config.get_model()


def loss_function(tar_real, tar_stop_prob, final_output, stop_prob, mel_linear):
    final_loss = masked_mean_absolute_error(tar_real, final_output)
    linear_loss = masked_mean_absolute_error(tar_real, mel_linear)
    stop_loss = new_scaled_crossentropy(index=2, scaling=hp.stop_loss_scaling)(tar_stop_prob, stop_prob)

    return final_loss + linear_loss + stop_loss


def train_step(inp, tar, stop_prob, mel_len):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    tar_stop_prob = stop_prob[:, 1:]

    tar_mel = tar_inp[:, 0::10, :]

    with tf.GradientTape() as tape:
        mel, mel_linear, final_output, stop_prob, _, _, _ = transformer(inp, tar_mel, True)
        loss = loss_function(tar_real, tar_stop_prob, final_output[:, :mel_len, :], stop_prob[:, :mel_len, :], mel_linear[:, :mel_len, :])

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)


def padding_data(audios, labels, stop_tokens, audio_len, text_len):
    max_audio_len = np.max(audio_len)
    max_text_len = np.max(text_len)

    audios_batch = audios[:, :max_audio_len, :]
    labels_batch = labels[:, :max_text_len]
    stop_tokens_batch = stop_tokens[:, :max_audio_len]

    return audios_batch, labels_batch, stop_tokens_batch, max_audio_len


min_loss = float('inf')

for epoch in range(1000):

    train_loss.reset_states()
    test_loss.reset_states()

    with tqdm(total=train_total) as pbar:
        for audios, labels, length, text_length, stop_tokens in train_data.generator():
            audios, labels, stop_tokens, mel_len = padding_data(audios, labels, stop_tokens, length, text_length)
            train_step(labels, audios, stop_tokens, mel_len - 1)
            pbar.update(batch_size)

    print('Epoch {} Train Loss {:.4f}'.format(epoch + 1, train_loss.result()))
