import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import MultiHeadAttention

# Training parameters
GRAD_CLIP = 1000.0  # Threshold for gradient clipping
RNN_SIZE = 128  # Size of the hidden layer
STATE_SIZE = 13  # Size of the state vector
ACTIONS_SIZE = 10  # Size of the action space
L2_REG = 0.001  # Coefficient for L2 regularization
EPSILON = 0.2  # Clipping parameter for PPO
LEARNING_RATE = 0.0001  # Learning rate for Adam optimizer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()

config.gpu_options.allow_growth = False  # Disable GPU
session = tf.Session(config=config)

# Weight initialization function
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# Actor-Critic Network (ACNet) implementation
class ACNet:
    def __init__(self, scope, TRAINING, trainer=None):
        self.trainer = trainer
        with tf.variable_scope(str(scope) + '/qvalues'):
            self.inputs = tf.placeholder(shape=[None, STATE_SIZE], dtype=tf.float32)  # Input state

            # Build neural network
            self.policy, self.value = self._build_net(self.inputs, RNN_SIZE, ACTIONS_SIZE)

        if TRAINING:
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, ACTIONS_SIZE, dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            self.old_policy = tf.placeholder(shape=[None, ACTIONS_SIZE], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.old_responsible_outputs = tf.reduce_sum(self.old_policy * self.actions_onehot, [1])

            ratio = self.responsible_outputs / (tf.clip_by_value(self.old_responsible_outputs, 1e-10, 1.0))
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            self.policy_loss = -tf.reduce_sum(tf.minimum(ratio * self.advantages, clipped_ratio * self.advantages))
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
            self.entropy = -0.01 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope + '/qvalues')
            self.loss = self.value_loss + self.policy_loss - self.entropy + tf.reduce_sum(reg_losses)

            trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + '/qvalues')
            self.gradients = tf.gradients(self.loss, trainable_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)
            self.apply_grads = trainer.apply_gradients(zip(grads, trainable_vars))
        print("Initialized network scope: " + str(scope))

    def _build_net(self, inputs, RNN_SIZE, a_size):
        w_init = tf.initializers.random_normal(stddev=0.1)

        # Reshape input
        inputs_reshaped = tf.reshape(inputs, [-1, 1, STATE_SIZE, 1])

        # 第一组卷积层
        conv1 = layers.Conv2D(
            filters=RNN_SIZE // 4,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(inputs_reshaped)
        conv1_bn = layers.BatchNormalization()(conv1)

        conv2 = layers.Conv2D(
            filters=RNN_SIZE // 2,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(conv1_bn)
        conv2_bn = layers.BatchNormalization()(conv2)
        
        # 增加新的卷积层
        conv3 = layers.Conv2D(
            filters=RNN_SIZE,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(conv2_bn)
        conv3_bn = layers.BatchNormalization()(conv3)
        
        conv4 = layers.Conv2D(
            filters=RNN_SIZE * 2,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(conv3_bn)
        conv4_bn = layers.BatchNormalization()(conv4)

        pool1 = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(conv4_bn)

        # Flatten the pooled output
        flat = layers.Flatten()(pool1)

        # 第一组全连接层
        fc1 = layers.Dense(
            units=RNN_SIZE,
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(flat)
        fc1_bn = layers.BatchNormalization()(fc1)

        # 增加新的全连接层
        fc2 = layers.Dense(
            units=RNN_SIZE * 2,
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(fc1_bn)
        fc2_bn = layers.BatchNormalization()(fc2)

        # 调整 fc1_bn 的维度以匹配 fc2_bn
        fc1_proj = layers.Dense(
            units=RNN_SIZE * 2,  # 调整维度
            kernel_initializer=w_init,
            activation=None,
            kernel_regularizer=regularizers.l2(L2_REG)
        )(fc1_bn)

        # 残差连接
        residual1 = tf.nn.relu(fc2_bn + fc1_proj)

        # 多头注意力层
        residual1_expanded = tf.expand_dims(residual1, axis=1)
        mha1 = MultiHeadAttention(num_heads=4, key_dim=RNN_SIZE // 4)(residual1_expanded, residual1_expanded)
        mha1_squeezed = tf.squeeze(mha1, axis=1)

        # 第二组全连接层
        fc3 = layers.Dense(
            units=RNN_SIZE * 2,
            kernel_initializer=w_init,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG)
        )(mha1_squeezed)
        fc3_bn = layers.BatchNormalization()(fc3)

        # 第二个残差连接
        residual2 = tf.nn.relu(fc3_bn + residual1)

        # 输出层：策略和价值
        policy_layer = layers.Dense(
            units=a_size,
            kernel_initializer=normalized_columns_initializer(1. / float(a_size)),
            activation=None,
            kernel_regularizer=regularizers.l2(L2_REG)
        )(residual2)
        policy = tf.nn.softmax(policy_layer)

        value = layers.Dense(
            units=1,
            kernel_initializer=normalized_columns_initializer(1.0),
            activation=None,
            kernel_regularizer=regularizers.l2(L2_REG)
        )(residual2)

        return policy, value


# Example to create the ACNet object
if __name__ == "__main__":
    ac_net = ACNet(scope="ACNet", TRAINING=True)
    
    # Simulate a single training step (you would need a proper state input)
    state_input = np.random.rand(1, STATE_SIZE)  # Example input state
    policy, value = ac_net._build_net(state_input, RNN_SIZE, ACTIONS_SIZE)

    print("Policy:", policy)
    print("Value:", value)

