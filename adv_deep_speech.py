
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio
from absl import app, flags

import os
import sys

import scipy.io.wavfile as wav

sys.path.append("DeepSpeech")
from DeepSpeech import create_inference_graph, create_model, rnn_impl_lstmblockfusedcell, create_overlapping_windows, try_loading
from util.feeding import samples_to_mfccs
from util.flags import create_flags, FLAGS
from util.config import Config, initialize_globals
from util.feeding import audiofile_to_features
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


class AdvDeepSpeech:
    """
    """

    def __init__(self, model_path, audio_path):
        self.sess = tf.Session()
        self.model_path = model_path
        self.audio_path = audio_path
        self.init(debug=True)

    def load_checkpoint(self):
        # Create a saver using variables from the above newly created graph
        print('restore model from:', self.model_path)
        saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state(self.model_path, 'best_dev_checkpoint')
        if not checkpoint:
            print('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(self.model_path))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        print('checkpoint_path:', checkpoint_path)
        saver.restore(self.sess, checkpoint_path)

    def load_audio(self, audio_path):
        samples = tf.io.read_file(audio_path)
        decoded = contrib_audio.decode_wav(samples, desired_channels=1)
        audio = decoded.audio.eval(session=self.sess)
        rate = decoded.sample_rate.eval(session=self.sess)
        print('decoded.audio:', audio.shape)
        print('decoded.sample_rate:', rate)
        return audio

    def init(self, debug=False):
        '''
        '''
        if debug:
            sample = self.load_audio(self.audio_path)

        self.audio = tf.placeholder(tf.float32, [None,1])
        features, features_len = samples_to_mfccs(self.audio, FLAGS.audio_sample_rate, train_phase=False)
        if debug:
            print('MFCC features:', features.eval(feed_dict = {self.audio:sample},session=self.sess))

        # Add batch dimension
        features = tf.expand_dims(features, 0)
        features_len = tf.expand_dims(features_len, 0)
        
        # Evaluate
        features = create_overlapping_windows(features)
        if debug:
            print('After creating overlapping windows:', features.eval(feed_dict = {self.audio:sample},session=self.sess))
            print('Feature length:', features_len.eval(feed_dict = {self.audio:sample},session=self.sess))
            #print('features.shape:', features.shape, ', features_len:', features_len)
        previous_state_c = tf.zeros([1, Config.n_cell_dim])
        previous_state_h = tf.zeros([1, Config.n_cell_dim])
        previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)

        # One rate per layer
        no_dropout = [None] * 6
        rnn_impl = rnn_impl_lstmblockfusedcell
        self.logits, _ = create_model(batch_x=features,
                                    batch_size=1,
                                    seq_length=features_len,
                                    dropout=no_dropout,
                                    previous_state=previous_state,
                                    overlap=False,
                                    rnn_impl=rnn_impl)

        # Load checkpoint from file
        self.load_checkpoint()

    def classify(self, audio_path):
        """
        Classify with 3rd-party ctc beam search
        """
        sample = self.load_audio(audio_path)

        # Apply softmax for CTC decoder
        logits = tf.nn.softmax(self.logits, name='logits')
        
        logits = logits.eval(feed_dict = {self.audio:sample}, session=self.sess)
        print('logits:', logits)

        logits = np.squeeze(logits)

        if FLAGS.lm_binary_path:
            self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            os.path.join('DeepSpeech', FLAGS.lm_binary_path),
                            os.path.join('DeepSpeech', FLAGS.lm_trie_path),
                            Config.alphabet)
        else:
            self.scorer = None

        r = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width,
                                    scorer=self.scorer, cutoff_prob=FLAGS.cutoff_prob,
                                    cutoff_top_n=FLAGS.cutoff_top_n)

        # Print highest probability result
        print(r[0][1])

    def classify2(self, audio_path):
        """
        classify with tf.nn.
        """
        sample = self.load_audio(audio_path)
        length = (sample.shape[0]-1)//320

        # logits = tf.nn.softmax(self.logits, name='logits')
        # logits = logits.eval(feed_dict = {self.audio:sample}, session=self.sess)
        # print('logits:', logits)
        length_var = tf.placeholder(tf.int32, [1])
        decoded, _ = tf.nn.ctc_beam_search_decoder(self.logits, length_var, merge_repeated=False, beam_width=500)
        r = self.sess.run(decoded, {self.audio:sample, length_var:[length]})

        print("-"*80)
        print("Classification:")
        print("".join([toks[x] for x in r[0].values]))
        print("-"*80)

    def attack(self, iteration):
        pass


def main(_):
    initialize_globals()
    if FLAGS.cmd != 'classify' and FLAGS.cmd != 'attack':
        print('Unsupport command, please check your cmd parameter')
    else:
        adv_ds = AdvDeepSpeech(FLAGS.model_path, FLAGS.input)
        if FLAGS.cmd == 'classify':
            adv_ds.classify2(FLAGS.input)
        else:
            adv_ds.attack(FLAGS.iteration)

if __name__ == '__main__':
    FLAGS = flags.FLAGS 
    flags.DEFINE_string("cmd", '', "support classify and attack")
    flags.DEFINE_string("model_path", '', "Path to the DeepSpeech checkpoint (model0.6.1)")
    flags.DEFINE_string("input", '', "Input audio .wav file(s), at 16KHz (separated by spaces)")
    flags.DEFINE_integer("iteration", 1000, "Iteration in attack, default is 1000.")
    create_flags()
    FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
    app.run(main)
