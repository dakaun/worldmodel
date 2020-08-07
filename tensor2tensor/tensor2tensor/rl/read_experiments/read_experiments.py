import tensorflow as tf
import pandas as pd

experiments = pd.read_pickle('/home/student/Downloads/SimPLE_experiments.pkl')
pong_exp = experiments[experiments['game']== 'pong']
pong_exp_141 = experiments[(experiments['game']== 'pong') &
                           (experiments['experiment_id'] == 141)]