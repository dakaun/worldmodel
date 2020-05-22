import tensorflow as tf
import numpy as np
import os
import json
from collections import namedtuple

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

model_rnn_size = 256
model_num_mixture = 5

# hyperparameters for our model. I was doing this on an older tf version, when HParams was not available ...
HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'num_actions',
                                         'encoded_img_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])


def default_hps():
  return HyperParams(num_steps=2000,  # train model for 2000 steps
                     max_seq_len=400,  # train on sequences of 1000
                     num_actions=4,
                     encoded_img_width=32,
                     rnn_size=model_rnn_size,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,   # number of mixtures in MDN
                     #restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=0)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

# MDN-RNN model tailored for doomrnn
class RNNModel():
    def __init__(self, hps, env_name, gpu_mode=True, reuse=False):
        self.hps = hps
        self.num_actions = 4 if 'Breakout' in env_name else 3
        with tf.variable_scope('mdn_rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device("/cpu:0"):
                    #print("model using cpu")
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self.build_model(self.hps)
            else:
                print("model using gpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model(self.hps)
        self.init_session()

    def build_model(self, hps):

        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture  # 5 mixtures
        INWIDTH = hps.encoded_img_width + self.num_actions # 36 channels
        OUTWIDTH = hps.encoded_img_width # 32 channels
        LENGTH = self.hps.max_seq_len # 230 timesteps

        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell  # use LayerNormLSTM

        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True  # False
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True  # False
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True  # False
        is_training = False if self.hps.is_training == 0 else True  # True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True  # False

        if use_recurrent_dropout:
            cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else: # here
            cell = cell_fn(num_units=hps.rnn_size, # nb of units in lstm cell
                           layer_norm=use_layer_norm) # False

        # multi-layer, and dropout:
        #print("input dropout mode =", use_input_dropout)
        #print("output dropout mode =", use_output_dropout)
        #print("recurrent dropout mode =", use_recurrent_dropout)
        if use_input_dropout:
            print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell

        self.sequence_lengths = LENGTH  # assume every sample has same length.
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, INWIDTH])
        self.output_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, OUTWIDTH])

        actual_input_x = self.input_x
        self.initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)

        NOUT = OUTWIDTH * KMIX * 3

        with tf.variable_scope('RNN'):
            output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        output, last_state = tf.nn.dynamic_rnn(cell, inputs=actual_input_x, initial_state=self.initial_state,
                                               time_major=False, swap_memory=True, dtype=tf.float32, scope="RNN")

        output = tf.reshape(output, [-1, hps.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        output = tf.reshape(output, [-1, KMIX * 3])
        self.final_state = last_state

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))


        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

        def get_lossfunc(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.reduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)

        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd

        out_logmix, out_mean, out_logstd = get_mdn_coef(output) # split into 3 variable: log_pi, mu, log_sigma

        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.output_x, [-1, 1])

        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)
        # calculation of z vector reconstruction loss: negative log-likelihood of observing true z, under the mixture
        # distribution parameterized by the output from MDN-RNN.

        self.cost = tf.reduce_mean(lossfunc)

        if self.hps.is_training == 1:
            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

        # initialize vars
        self.init = tf.global_variables_initializer()

        t_vars = tf.trainable_variables()
        self.assign_ops = {}
        for var in t_vars:
            #if var.name.startswith('mdn_rnn'):
            pshape = var.get_shape()
            pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
            assign_op = var.assign(pl)
            self.assign_ops[var] = (assign_op, pl)

    def init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def save_model(self, model_save_path, epoch):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'doomcover_rnn')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, epoch)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                #if var.name.startswith('mdn_rnn'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                #if var.name.startswith('mdn_rnn'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                idx += 1

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up!
        return rparam

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def load_json(self, jsonfile='rnn.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile='rnn.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))


def get_pi_idx(x, pdf):
    # samples from a categorial distribution
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print('error with sampling ensemble')
    return -1


def sample_sequence(sess, s_model, hps, init_z, actions, temperature=1.0, seq_len=1000):
    # generates a random sequence using the trained model

    OUTWIDTH = hps.output_seq_width
    INWIDTH = hps.input_seq_width

    prev_x = np.zeros((1, 1, OUTWIDTH))
    prev_x[0][0] = init_z

    prev_state = sess.run(s_model.initial_state)

    '''
    if prev_data is not None:
      # encode the previous data into the hidden state first
      for i in range(prev_data.shape[0]):
        prev_x[0][0] = prev_data[i]
        feed = {s_model.input_x: prev_x, s_model.initial_state:prev_state}
        [next_state] = sess.run([s_model.final_state], feed)
        prev_state = next_state
    '''

    strokes = np.zeros((seq_len, OUTWIDTH), dtype=np.float32)

    for i in range(seq_len):
        input_x = np.concatenate((prev_x, actions[i].reshape((1, 1, 3))), axis=2)
        feed = {s_model.input_x: input_x, s_model.initial_state: prev_state}
        [logmix, mean, logstd, next_state] = sess.run(
            [s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)

        # adjust temperatures
        logmix2 = np.copy(logmix) / temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)
        for j in range(OUTWIDTH):
            idx = get_pi_idx(np.random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = np.random.randn(OUTWIDTH) * np.sqrt(temperature)
        next_x = chosen_mean + np.exp(chosen_logstd) * rand_gaussian

        strokes[i, :] = next_x

        prev_x[0][0] = next_x
        prev_state = next_state

    return strokes


def rnn_init_state(rnn):
    return rnn.sess.run(rnn.initial_state)


def rnn_next_state(rnn, z, a, prev_state, env_name):
    if 'Breakout' in env_name:
        input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 4))), axis=2)
    elif 'CarRacing' in env_name:
        input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 3))), axis=2)
    feed = {rnn.input_x: input_x, rnn.initial_state: prev_state}
    return rnn.sess.run(rnn.final_state, feed)

def rnn_output_size():
  return (32+256)

def rnn_output(state, z):
  return np.concatenate([z, state.h[0]])

