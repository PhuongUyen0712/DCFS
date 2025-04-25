import numpy as np
import tensorflow as tf


class genreal_func:
    def __init__(self):
        return

    def create_batch(self, memory, discount_rate, agent):
        agent.clear_inference_model_state()

        sample = np.asarray(memory)
        s = sample[:, 0]
        a = sample[:, 1].astype(np.int8)
        r = sample[:, 2].astype(float)
        s2 = sample[:, 3]
        d = sample[:, 4] * 1.0
        available_actions = sample[:, 5]  ##added available_actions
        available_actions = np.vstack(available_actions)
        available_actions = available_actions.reshape(
            1, available_actions.shape[0], available_actions.shape[1]
        )

        X_batch = np.vstack(s)
        X_batch = X_batch.reshape(1, X_batch.shape[0], X_batch.shape[1])
        ## padding with 0 in the begining
        X_batch = np.pad(
            X_batch,
            [(0, 0), ((X_batch.shape[2] - X_batch.shape[1]), 0), (0, 0)],
            mode="constant",
        )

        s2 = np.vstack(s2)
        s2 = s2.reshape(1, s2.shape[0], s2.shape[1])

        future_rewards = agent.predict_target_network(s2)
        # discount factor * expected future reward
        #         print(f"future_rewards: {future_rewards}")
        #         print(f"available_actions: {available_actions}")
        #         discount_future_rewards = discount_rate * tf.reduce_max(future_rewards*available_actions, axis=1)
        discount_future_rewards = 0 * tf.reduce_max(
            future_rewards * available_actions, axis=1
        )

        # if <0 set to 0
        discount_future_rewards = tf.maximum(discount_future_rewards, 0)
        # Q value = reward + discount_future_rewards
        updated_q_values = r + discount_future_rewards
        #         print(updated_q_values)
        #         updated_q_values = r
        #         print(updated_q_values)
        # If final state set the last value to -1
        updated_q_values = updated_q_values * (1 - d)

        #         print(f"r:{r}")
        # print(f"a: {a}")
        #         print(f"y:{updated_q_values.numpy()}")
        return X_batch, a, updated_q_values.numpy()

    #         return X_batch, a, updated_q_values

    def internal_reward(self, e, initial_error, discount_rate):
        ## Compute the gamma-discounted rewards over an episode
        r = np.zeros_like(e)
        r[0] = initial_error - e[0]
        for t in range(1, len(e)):
            r[t] = e[t - 1] - e[t]
        #         return r

        discounted_r, cum_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cum_r = r[t] + (np.max(cum_r, 0) * discount_rate)
            discounted_r[t] = round(cum_r, 5)
        return discounted_r

    def learningRate(self, epi, l_r_schedule, episodes):
        l_r = 0.0
        for episode_boundary, lr in l_r_schedule:
            if epi >= episodes * episode_boundary:
                l_r = lr
        print("epoch, lr: ", epi, l_r)
        return l_r

    def epsilon(self, epi, epsilon_schedule, episodes):
        eps = 0.0
        for episode_boundary, epsilon in epsilon_schedule:
            if epi >= episodes * episode_boundary:
                eps = epsilon
        print("epoch, epsilon: ", epi, eps)
        return eps
