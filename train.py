#!/usr/bin/env python3
import os
import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# Keras model imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, Dense, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#========================================================
# Utility functions (from your first block)
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path):
    """
    Reads a ROS 2 bag via rosbag2_py and returns
    lidar scans, steering (angular.z), speeds (linear.x), and timestamps.
    """
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        # Convert nanoseconds to seconds
        t = t_ns * 1e-9

        if topic == 'scan':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)

        elif topic == 'odom':
            msg = deserialize_message(serialized_msg, Odometry)
            servo_data.append(msg.twist.twist.angular.z)
            speed_data.append(msg.twist.twist.linear.x)
            # align timestamp for control measurements too
            # (you can choose to append t here or ignore if you only need LIDAR dt)
            # timestamps.append(t)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps)
    )

#========================================================
# Sequence builder (unchanged)
#========================================================
def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
    """
    Build sliding-window sequences of LiDAR frames with time-delta features
    and corresponding steering/speed targets.
    """
    X, y = [], []
    num_ranges = lidar_data.shape[1]

    for i in range(len(lidar_data) - sequence_length):
        # stack the raw scans [seq_len x num_ranges]
        frames = np.stack(lidar_data[i : i + sequence_length], axis=0)  # (seq_len, num_ranges)

        # compute deltas dt between frames, shape (seq_len, 1)
        dt = np.diff(timestamps[i : i + sequence_length + 1]).reshape(sequence_length, 1)

        # replicate dt across all range bins: becomes (seq_len, num_ranges)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)

        # now stack channels: (seq_len, num_ranges, 2)
        seq = np.concatenate([
            frames[..., None],      # (seq_len, num_ranges, 1)
            dt_tiled[..., None]     # (seq_len, num_ranges, 1)
        ], axis=2)

        X.append(seq)
        y.append([servo_data[i + sequence_length], speed_data[i + sequence_length]])

    return np.array(X), np.array(y)


#========================================================
# Model definition (RNN + Attention)
#========================================================
def build_spatiotemporal_model(seq_len, num_ranges):
    inp = Input(shape=(seq_len, num_ranges, 2), name='lidar_sequence')
    x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inp)
    x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
    x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    q = Dense(64)(lstm_out)
    k = Dense(64)(lstm_out)
    v = Dense(64)(lstm_out)
    attn = Attention()([q, v, k])
    context = tf.reduce_mean(attn, axis=1)
    out = Dense(2, activation='tanh', name='controls')(context)
    return Model(inp, out, name='RNN_Attention_Controller')

#========================================================
# RL environment and policy for PPO2
#========================================================
class LidarSequenceEnv(gym.Env):
    """Minimal gym environment over the prerecorded lidar sequences."""

    def __init__(self, sequences, targets):
        super().__init__()
        self.sequences = sequences
        self.targets = targets
        self.idx = 0
        obs_shape = sequences.shape[1:]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        self.idx = 0
        return self.sequences[self.idx]

    def step(self, action):
        target = self.targets[self.idx]
        reward = -np.mean((action - target) ** 2)
        self.idx += 1
        done = self.idx >= len(self.sequences)
        obs = self.sequences[self.idx] if not done else np.zeros_like(self.sequences[0])
        return obs, reward, done, {}


class RLnPolicy(ActorCriticPolicy):
    """Custom policy with Conv -> Bi-LSTM -> Attention."""

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(RLnPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[64], **kwargs
        )
        seq_len = ob_space.shape[0]
        num_ranges = ob_space.shape[1]
        with tf.variable_scope("rl2net"):
            x = tf.reshape(self.processed_obs, [-1, seq_len, num_ranges, 2])
            # example convolutional feature extractor
            x = tf.reshape(x, [-1, num_ranges, 2])
            x = tf.layers.conv1d(x, 24, 10, 4, activation=tf.nn.relu)
            x = tf.layers.conv1d(x, 36, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv1d(x, 48, 4, 2, activation=tf.nn.relu)
            x = tf.layers.flatten(x)
            x = tf.reshape(x, [n_batch, seq_len, -1])
            lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.keras.layers.LSTMCell(64),
                tf.keras.layers.LSTMCell(64),
                x,
                dtype=tf.float32,
            )
            lstm_out = tf.concat(lstm_out, axis=-1)
            q = tf.layers.dense(lstm_out, 64)
            k = tf.layers.dense(lstm_out, 64)
            v = tf.layers.dense(lstm_out, 64)
            attn = tf.keras.layers.Attention()([q, v, k])
            context = tf.reduce_mean(attn, axis=1)
            pi_h = tf.layers.dense(context, 64, activation=tf.nn.tanh)
            vf_h = tf.layers.dense(context, 64, activation=tf.nn.tanh)

        self.pi_latent = pi_h
        self.vf_latent = vf_h
        self._setup_init()

#========================================================
# Main
#========================================================
if __name__ == '__main__':
    # Check for GPU
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))

    # --- Parameters ---
    bag_paths = ['./scripts/car_Dataset/controller_slow_5min/controller_slow_5min_0.db3', './scripts/car_Dataset/controller_slow_10min/controller_slow_10.db3']
    seq_len    = 5
    batch_size = 64
    lr         = 5e-5
    epochs     = 20

    # --- Load & concatenate all bags ---
    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)

    all_lidar = np.array(all_lidar)
    all_servo = np.array(all_servo)
    all_speed = np.array(all_speed)
    all_ts    = np.array(all_ts)

    # Normalize speed 0→1
    min_s, max_s = all_speed.min(), all_speed.max()
    all_speed = linear_map(all_speed, min_s, max_s, 0, 1)

    # Build sequences
    X, y = create_lidar_sequences(all_lidar, all_servo, all_speed, all_ts, seq_len)
    n_samples, _, num_ranges, _ = X.shape
    print(f'Total sequences: {n_samples}, ranges per scan: {num_ranges}')

    # Shuffle and split
    X, y = shuffle(X, y, random_state=42)
    split = int(0.85 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build & compile model
    model = build_spatiotemporal_model(seq_len, num_ranges)
    model.compile(optimizer=Adam(lr), loss='huber')
    print(model.summary())

    # Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    print(f'Training done in {int(time.time() - t0)}s')

    # Plot loss curve
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('Figures/loss_curve.png')
    plt.close()

    # Convert & save TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    os.makedirs('Models', exist_ok=True)
    # with open('Models/RNN_Attn_Controller.tflite', 'wb') as f:
    with open('Models/test.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite model saved.')

    # Final evaluation
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final test loss: {test_loss:.4f}')

    # ================= RL fine tuning with PPO2 =================
    env = DummyVecEnv([lambda: LidarSequenceEnv(X_train, y_train)])
    model_rl = PPO2(
        policy=RLnPolicy,
        env=env,
        n_steps=seq_len * 20,
        nminibatches=1,
        lam=0.95,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./rl2_tb/",
    )

    # Optionally load pretrained weights (if available)
    if os.path.exists("Models/ppo_rln.zip"):
        model_rl.load("Models/ppo_rln")

    model_rl.learn(total_timesteps=100000)
    os.makedirs("Models", exist_ok=True)
    model_rl.save("Models/ppo_rln")
