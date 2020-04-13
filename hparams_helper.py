import numpy as np


def apply_hparams(hparams):
    hparams.add_hparam('cleaners', 'english_cleaners')

    hparams.add_hparam('num_freq', 513)
    hparams.add_hparam('rescale', True)
    hparams.add_hparam('trim_silence', True)
    hparams.add_hparam('use_lws', True)

    hparams.add_hparam('frame_shift_ms', None)

    hparams.add_hparam('signal_normalization', True)
    hparams.add_hparam('symmetric_mels', False)

    hparams.add_hparam('griffin_lim_iters', 60)


    hparams.add_hparam('stop_at_any', True)

    hparams.add_hparam('embedding_dim', 512)

    hparams.add_hparam('enc_conv_num_layers', 3)
    hparams.add_hparam('enc_conv_kernel_size', (5, ))
    hparams.add_hparam('enc_conv_channels', 512)
    hparams.add_hparam('encoder_lstm_units', 256)

    hparams.add_hparam('smoothing', False)
    hparams.add_hparam('attention_dim', 128)
    hparams.add_hparam('attention_filters', 32)
    hparams.add_hparam('attention_kernel', (31, ))
    hparams.add_hparam('cumulative_weights', True)

    hparams.add_hparam('prenet_layers', [256, 256])
    hparams.add_hparam('decoder_layers', 2)
    hparams.add_hparam('decoder_lstm_units', 1024)
    hparams.add_hparam('max_iters', 2500)

    hparams.add_hparam('postnet_num_layers', 5)
    hparams.add_hparam('postnet_kernel_size', (5, ))
    hparams.add_hparam('postnet_channels', 512)

    hparams.add_hparam('mask_encoder', False)
    hparams.add_hparam('impute_finished', False)
    hparams.add_hparam('mask_finished', False)

    hparams.add_hparam('predict_linear', False)


    hparams.add_hparam('tacotron_batch_size', 16)
    hparams.add_hparam('tacotron_reg_weight', 1e-6)
    hparams.add_hparam('tacotron_scale_regularization', True)

    hparams.add_hparam('tacotron_decay_learning_rate', True)
    hparams.add_hparam('tacotron_start_decay', 50000)
    hparams.add_hparam('tacotron_decay_steps', 50000)
    hparams.add_hparam('tacotron_decay_rate', 0.4)
    hparams.add_hparam('tacotron_initial_learning_rate', 1e-3)
    hparams.add_hparam('tacotron_final_learning_rate', 1e-5)

    hparams.add_hparam('tacotron_adam_beta1', 0.9)
    hparams.add_hparam('tacotron_adam_beta2', 0.999)
    hparams.add_hparam('tacotron_adam_epsilon', 1e-6)

    hparams.add_hparam('tacotron_zoneout_rate', 0.1)
    hparams.add_hparam('tacotron_dropout_rate', 0.5)

    hparams.add_hparam('tacotron_teacher_forcing_ratio', 1.0)