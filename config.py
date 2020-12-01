import subprocess
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import ruamel.yaml

import hyperparameter as hp
from model import AutoregressiveTransformer


class Config:

    def __init__(self, model_kind: str):
        if model_kind not in ['autoregressive', 'forward']:
            raise TypeError(f"model_kind must be in {['autoregressive', 'forward']}")

        self.max_r = np.array(hp.reduction_factor_schedule)[0, 1].astype(np.int32)
        self.stop_scaling = hp.stop_loss_scaling

    def get_model(self):
        return AutoregressiveTransformer(mel_channels=hp.mel_channels,
                                         encoder_model_dimension=hp.encoder_model_dimension,
                                         decoder_model_dimension=hp.decoder_model_dimension,
                                         encoder_num_heads=hp.encoder_num_heads,
                                         decoder_num_heads=hp.decoder_num_heads,
                                         encoder_feed_forward_dimension=hp.encoder_feed_forward_dimension,
                                         decoder_feed_forward_dimension=hp.decoder_feed_forward_dimension,
                                         encoder_maximum_position_encoding=hp.encoder_max_position_encoding,
                                         decoder_maximum_position_encoding=hp.decoder_max_position_encoding,
                                         encoder_dense_blocks=hp.encoder_dense_blocks,
                                         decoder_dense_blocks=hp.decoder_dense_blocks,
                                         decoder_prenet_dimension=hp.decoder_prenet_dimension,
                                         encoder_prenet_dimension=hp.encoder_prenet_dimension,
                                         encoder_attention_conv_kernel=hp.encoder_attention_conv_kernel,
                                         decoder_attention_conv_kernel=hp.decoder_attention_conv_kernel,
                                         encoder_attention_conv_filters=hp.encoder_attention_conv_filters,
                                         decoder_attention_conv_filters=hp.decoder_attention_conv_filters,
                                         postnet_conv_filters=hp.postnet_conv_filters,
                                         postnet_conv_layers=hp.postnet_conv_layers,
                                         postnet_kernel_size=hp.postnet_kernel_size,
                                         dropout_rate=hp.dropout_rate,
                                         max_r=self.max_r,
                                         mel_start_value=hp.mel_start_value,
                                         mel_end_value=hp.mel_end_value)

    @staticmethod
    def new_adam(learning_rate, beta_1=0.9, beta_2=0.98, ):
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=beta_1,
                                        beta_2=beta_2,
                                        epsilon=1e-9)


if __name__ == '__main__':
    config = Config("autoregressive")
    model = config.get_model()

    input_text = np.ones((1, 12))
    target_text = np.ones((1, 500, 80))
    mel, mel_linear, final_output, stop_prob, dec_output, attention_weights, encoder_attention = \
        model(input_text, target_text, False)
