import tensorflow as tf
import random
import numpy as np
import os
import json
from WorldModelsExperiments.breakout_dreamer.dreamer_vae.dreamer_vae import ConvVAE, reset_graph

# Hyperparameters for ConvVAE
z_size=64
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "/home/dakaun/PycharmProjects/world_model/WorldModelsExperiments/breakout/record"

model_save_path = "tf_vaedream"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
print('Nb of trials: ', len(filelist))

# get total nb of images
nb_images = (len(filelist)*500) #get_nb_of_total_images(dataset)
print('Total nb of images: ', nb_images)

# split into batches:
num_batches = int(np.floor(nb_images/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

#len dataset
len_data = len(np.load(os.path.join(DATA_DIR, filelist[0]))['obs'])
len_data = int(int(len_data)/100)
print('length of each trial: ', len_data)

# reload trained model and retrain:
if 'vae.json' in os.listdir('tf_vaedream'):
  vae.load_json('tf_vaedream/vae.json')

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
train_param = []

for epoch in range(NUM_EPOCH):
  np.random.shuffle(filelist) # shuffle complete dataset

  for idx, filename in enumerate(filelist):
    raw_obs = np.load(os.path.join(DATA_DIR, filename))['obs']
    np.random.shuffle(raw_obs)
    for i in range(len_data):
      obs = raw_obs[i*batch_size:(i+1)*batch_size]

      feed = {vae.x: obs,}

      (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
      ], feed)

      train_param.append({
        'train_loss': train_loss.tolist(),
        'r_loss': r_loss.tolist(),
        'kl_loss': kl_loss.tolist(),
        'train_step': train_step.tolist()
      })

      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss, kl_loss)
      if ((train_step+1) % 5000 == 0):
        vae.save_json("tf_vaedream/vae.json")
with open('tf_vaedream/vae_train_param.json', 'w') as pfile:
  json.dump(train_param, pfile, sort_keys=True)
# finished, final model:
print('vae dreamer training done, save final model')
vae.save_json("tf_vaedream/vae.json")
