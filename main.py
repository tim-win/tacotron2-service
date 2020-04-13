# Entrypoint script  for tacotron2-service
import os
import sys
sys.path.append('/opt/Tacotron-2')
import torch
import tensorflow as tf
import pysptk
import numpy as np
print(tf.__version__, pysptk.__version__, np.__version__)

import librosa.display
from hparams import hparams
from train import build_model
from synthesis import wavegen
import torch

from glob import glob
from tqdm import tqdm

from hparams_helper import apply_hparams


def main():
    # Pulled from tacotron-2's synthesize.py
    hparams.add_hparam('max_abs_value', 4.0)
    hparams.add_hparam('power', 1.1)
    hparams.add_hparam('outputs_per_step', 1)

    # Do all the rest
    apply_hparams(hparams)

    from tacotron.synthesize import run_eval
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    output_dir = 'tacotron_' + 'output/'

    # try:
    checkpoint_path = tf.train.get_checkpoint_state('/opt/Tacotron-2/logs-Tacotron/pretrained').model_checkpoint_path
    print('loaded model at {}'.format(checkpoint_path))
    #except:
    #    raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(args.checkpoint))

    run_eval(None, checkpoint_path, output_dir, 'Hello, Tim Winter.')


if __name__ == '__main__':
    sys.exit(main())
