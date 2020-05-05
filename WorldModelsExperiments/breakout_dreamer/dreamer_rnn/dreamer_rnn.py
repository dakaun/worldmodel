import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
from collections import namedtuple
import json
import time

model_path_name = 'tf_models'
model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.
model_state_space = 2 # includes C and H concatenated if 2, otherwise just H

TEMPERATURE = 1.25 # train with this temperature

HyperParams = namedtuple('HyperParams', ['max_seq_len',
                                         'seq_width',
                                         'num_actions',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'restart_factor',
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
  return HyperParams(max_seq_len=1000, # train on sequences of 1000
                     seq_width=64,    # width of our data (64)
                     num_actions=4,
                     rnn_size=model_rnn_size,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,   # number of mixtures in MDN
                     restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
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
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

class RNNModel():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    with tf.variable_scope('mdn_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu")
        self.g = tf.Graph()
        with self.g.as_default():
          self.build_model(hps)
    self.init_session()
  def build_model(self, hps):
    
    self.num_mixture = hps.num_mixture
    KMIX = self.num_mixture # 5 mixtures
    WIDTH = hps.seq_width # 64 channels
    LENGTH = self.hps.max_seq_len-1 # 999 timesteps

    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell # use LayerNormLSTM

    use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True
    is_training = False if self.hps.is_training == 0 else True
    use_layer_norm = False if self.hps.use_layer_norm == 0 else True

    if use_recurrent_dropout:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
    else:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)

    # multi-layer, and dropout:
    print("input dropout mode =", use_input_dropout)
    print("output dropout mode =", use_output_dropout)
    print("recurrent dropout mode =", use_recurrent_dropout)
    if use_input_dropout:
      print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = LENGTH # assume every sample has same length.
    self.batch_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, WIDTH])
    self.batch_action = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, self.hps.num_actions])
    self.batch_restart = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len])

    self.input_z = self.batch_z[:, :LENGTH, :]
    self.input_action = self.batch_action[:, :LENGTH, :]
    self.input_restart = self.batch_restart[: , :LENGTH]

    self.target_z = self.batch_z[:, 1:, :]
    self.target_restart = self.batch_restart[: , 1:]
    
    self.input_seq = tf.concat([self.input_z,
                                self.input_action,
                                tf.reshape(self.input_restart, [self.hps.batch_size, LENGTH, 1])], axis=2)

    self.zero_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
    self.initial_state = self.zero_state

    inputs = tf.unstack(self.input_seq, axis=1)

    def custom_rnn_autodecoder(decoder_inputs, input_restart, initial_state, cell, scope=None):
      # customized rnn_decoder for the task of dealing with restart
      with tf.variable_scope(scope or "RNN"):
        state = initial_state
        zero_c, zero_h = self.zero_state
        outputs = []
        prev = None

        for i in range(LENGTH):
          inp = decoder_inputs[i]
          if i > 0:
            tf.get_variable_scope().reuse_variables()

          # if restart is 1, then set lstm state to zero
          restart_flag = tf.greater(input_restart[:, i], 0.5)

          c, h = state

          c = tf.where(restart_flag, zero_c, c)
          h = tf.where(restart_flag, zero_h, h)

          output, state = cell(inp, tf.nn.rnn_cell.LSTMStateTuple(c, h))
          outputs.append(output)

      return outputs, state

    outputs, final_state = custom_rnn_autodecoder(inputs, self.input_restart, self.initial_state, self.cell)
    output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.hps.rnn_size])

    NOUT = WIDTH * KMIX * 3 + 1 # plus 1 to predict the restart state.

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    output = tf.reshape(output, [-1, hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    
    self.out_restart_logits = output[:, 0]
    output = output[:, 1:]
    
    output = tf.reshape(output, [-1, KMIX * 3])
    self.final_state = final_state    

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

    out_logmix, out_mean, out_logstd = get_mdn_coef(output)

    self.out_logmix = out_logmix
    self.out_mean = out_mean
    self.out_logstd = out_logstd

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_z,[-1, 1])

    lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)

    self.z_cost = tf.reduce_mean(lossfunc)

    flat_target_restart = tf.reshape(self.target_restart, [-1, 1])

    self.r_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_restart,
                                                          logits=tf.reshape(self.out_restart_logits,[-1, 1]))
    
    factor = tf.ones_like(self.r_cost) + flat_target_restart * (self.hps.restart_factor-1.0)

    self.r_cost = tf.reduce_mean(tf.multiply(factor, self.r_cost))

    self.cost = self.z_cost + self.r_cost

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
        pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
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
    saver.save(sess, checkpoint_path, epoch) # just keep one
  
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
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = tuple(var.get_shape().as_list())
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op, pl = self.assign_ops[var]
        self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
        idx += 1
  
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up!
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
