'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems
import json
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from WorldModelsExperiments.breakout.env import reset_graph
from WorldModelsExperiments.breakout.vae.vae import ConvVAE

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 20
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def load_raw_data_list(filelist): # load only observation
  data_list = []
  for i, f_name in enumerate(filelist):
    raw_data = np.load(os.path.join(DATA_DIR, f_name))['obs']
    data_list.append(raw_data)
    if ((i+1) % 1000 == 0): # crashed nach 2000
      print("loading file", (i+1))
  return data_list

def count_length_of_raw_data(raw_data_list):
  min_len = 100000
  max_len = 0
  N = len(raw_data_list)
  total_length = 0
  for i in range(N):
    l = len(raw_data_list[i])
    if l > max_len:
      max_len = l
    if l < min_len:
      min_len = l
    if l < 10:
      print(i)
    total_length += l
  return  total_length

def create_dataset(raw_data_list):
  N = len(raw_data_list)
  M = count_length_of_raw_data(raw_data_list)
  data = np.zeros((M, 64, 64, 3), dtype=np.float32)
  idx = 0
  for i in range(N):
    raw_data = raw_data_list[i]
    l = len(raw_data)
    if (idx+l) > M:
      data = data[0:idx]
      break
    data[idx:idx+l] = raw_data
    idx += l
  return data

def get_nb_of_total_images(dataset):
  total_nb_images = 0
  for set in dataset:
    total_nb_images+=len(set)
  return total_nb_images

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
print('Nb of trials: ', len(filelist))

#dataset = load_raw_data_list(filelist) #just able to load 2000
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
if 'vae.json' in os.listdir('tf_vae'):
  vae.load_json('tf_vae/vae.json')

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
        vae.save_json("tf_vae/vae.json")
with open('tf_vae/vae_train_param3.json', 'w') as pfile:
  json.dump(train_param, pfile, sort_keys=True)
# finished, final model:
print('training done, save final model')
vae.save_json("tf_vae/vae.json")
