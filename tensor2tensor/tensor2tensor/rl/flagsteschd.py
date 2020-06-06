import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', 'here-dir', 'description of dir')
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")
cur_dir = FLAGS.output_dir
print(cur_dir)
#print(FLAGS.work_id)
print(FLAGS.loop_hparams)

