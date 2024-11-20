import gym
import numpy as np
import tensorflow as tf
import time
import imageio
import os
import scipy.signal as signal
from cargo_lander_env import CargoLanderEnv  # Import the custom lunar lander environment
from NN import ACNet  # Import the ACNet implementation

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Parameters
gamma = 0.99  # Discount factor
max_episode_length = 10000  # Maximum steps per episode
SUMMARY_WINDOW = 1  # Summary window size
TRAINING = True  # Training mode enabled
MODEL_PATH = './lander_model'
EXPERIENCE_BUFFER_SIZE = 512  # Increase buffer size for larger batch training
LR_Q = 1e-4  # Learning rate

# TensorBoard records
episode_rewards = []
episode_lengths = []
episode_mean_values = []

# Ensure model directory exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Disable Eager Execution
tf.compat.v1.disable_eager_execution()

# Global variable definition
episode_count = 0  # Initialize episode count at 0
success_count = 0
failure_count = 0

# Worker class
class Worker:
    def __init__(self, env, global_network):
        self.env = env
        self.global_network = global_network

    def discount_rewards(self, rewards, gamma):
        return signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout, dtype=object)
        observ = np.vstack(rollout[:, 0])  # Ensure it's a 2D array
        actions = np.array(rollout[:, 1], dtype=int).flatten()
        rewards = np.array(rollout[:, 2], dtype=float)
        values = np.array(rollout[:, 4], dtype=float)
        old_policies = np.vstack(rollout[:, 5])  # Get old policy distribution

        # Advantage estimation and discounted reward calculation
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount_rewards(self.rewards_plus, gamma)[:-1]
        advantages = rewards + gamma * np.append(values[1:], bootstrap_value) - values
        advantages = self.discount_rewards(advantages, gamma)

        # Ensure `discounted_rewards` and `advantages` are 1D arrays
        discounted_rewards = discounted_rewards.flatten()
        advantages = advantages.flatten()

        # Create training feed_dict
        feed_dict = {
            self.global_network.target_v: discounted_rewards,
            self.global_network.inputs: observ,
            self.global_network.actions: actions,
            self.global_network.advantages: advantages,
            self.global_network.old_policy: old_policies,  # Provide values for old_policy
        }

        # Perform training step
        v_l, p_l, e_l, _ = sess.run(
            [
                self.global_network.value_loss,
                self.global_network.policy_loss,
                self.global_network.entropy,
                self.global_network.apply_grads,
            ],
            feed_dict=feed_dict,
        )

        return v_l, p_l, e_l, np.sum(rewards)

    def work(self, max_episode_length, gamma, sess, saver):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, success_count, failure_count

        metrics = {}  # Store metrics for the episode

        # Load metrics if available
        metrics_path = f"{MODEL_PATH}/metrics.npy"
        if os.path.exists(metrics_path):
            metrics = np.load(metrics_path, allow_pickle=True).item()
            if metrics:
                episode_count = max(metrics.keys()) + 1  # Continue from the last episode
                success_count = sum(1 for m in metrics.values() if m["success"])
                failure_count = sum(1 for m in metrics.values() if not m["success"])
                print(f"Loaded metrics from {metrics_path}. Resuming from episode {episode_count}.")
                print(f"Successes: {success_count}, Failures: {failure_count}")

        while episode_count < 100000:
            start_time = time.time()  # Start the timer
            episode_buffer = []
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            episode_action_count = 0  # Track actions per episode
            done = False
            s = self.env.reset()

            while not done and episode_step_count < max_episode_length:
                # Get action and value
                a_dist, v = sess.run(
                    [self.global_network.policy, self.global_network.value],
                    feed_dict={self.global_network.inputs: [s]},
                )
                action = np.random.choice(len(a_dist[0]), p=a_dist[0])

                # Interact with the environment
                s1, r, fuel_consume, done, info, legs = self.env.step(action)

                episode_buffer.append([s, action, r, s1, v[0, 0], a_dist[0]])
                episode_values.append(v[0, 0])

                episode_reward += r
                s = s1
                episode_step_count += 1
                episode_action_count += 1  # Increment per step

                # Update network when buffer is full
                if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE or done:
                    s1_value = 0 if done else sess.run(
                        self.global_network.value,
                        feed_dict={self.global_network.inputs: [s1]},
                    )[0, 0]
                    v_l, p_l, e_l, episode_total_reward = self.train(
                        episode_buffer, sess, gamma, s1_value
                    )
                    episode_buffer = []

            # Record success/failure
            if self.env.success:
                success_count += 1
            else:
                failure_count += 1

            # Save metrics for the episode
            metrics[episode_count] = {
                'reward': episode_reward,
                'steps': episode_step_count,
                'actions': episode_action_count,
                'success': self.env.success,
            }
            
            elapsed_time = time.time() - start_time

            print(
                f"Episode {episode_count}, Reward: {episode_reward}, Actions: {episode_action_count}, \n"
                f"Value Loss: {v_l:.3f}, Policy Loss: {p_l:.3f}, Entropy: {e_l:.3f}, \n"
                f"Successes: {success_count}, Failures: {failure_count}\n"
                f"Contact point is:{legs} \n"
                f"Time Spent: {elapsed_time:.2f} seconds\n"
            )

            # Save metrics and model periodically
            if episode_count % SUMMARY_WINDOW == 0:
                saver.save(sess, f"{MODEL_PATH}/model-{episode_count}.cptk")
                np.save(f"{MODEL_PATH}/metrics.npy", metrics)  # Save metrics

            episode_count += 1


if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()

    # Configure TensorFlow for CPU usage
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True  # Allow operations to fall back to CPU

    trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR_Q)
    master_network = ACNet("global", TRAINING=True, trainer=trainer)
    env = CargoLanderEnv()
    worker = Worker(env, master_network)

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Load checkpoint if exists
        checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
        if checkpoint:
            print(f"Restoring model from checkpoint: {checkpoint}")
            saver.restore(sess, checkpoint)
        else:
            print("No checkpoint found. Starting training from scratch.")

        worker.work(max_episode_length, gamma, sess, saver)
