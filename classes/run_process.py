import numpy as np
from collections import deque
from classes.general_functions import genreal_func
genreal_func = genreal_func()


class run_process():

    def __init__(self):
        return

    def runprocess(self, env, agent, episode_size, eps,discount_rate):

        memory = deque()

        initial_error = 1 - (1 / env.number_of_classes)

        episode_data = env.get_data(episode_size, 1, "train")
        X, y = env.data_separate(episode_data)
        X_train_main, X_test_main, y_train, y_test = env.data_split(X, y)

        s = env.action_space
        selected_actions = env.action_space  # new
        done = 0

        agent.clear_inference_model_state()

        while not done:
            a, a_Q = agent.act(s, 0, eps, selected_actions)
            if a_Q >= agent.internal_threshold:
                s2, selected_actions = env.s2(s, a, selected_actions)
                accuracy = env.accuracy(selected_actions, X_train_main, X_test_main, y_train, y_test)
                error = 1 - accuracy
                done = agent.is_done(a_Q, selected_actions)
                available_actions = np.ones(len(s)) - selected_actions
                memory.append([s, a, error, s2, done, available_actions]) ##added available_actions
                s = s2
            else:
                done = 1

        # Convert memory to numpy array, specifying dtype=object to handle mixed types/shapes
        episode_memory = np.asarray(memory, dtype=object)

        if len(episode_memory) > 0:
#             print(f"actions: {episode_memory[:, 1].astype(np.int)}")
            errors = episode_memory[:, 2].astype(float)
            rewards = genreal_func.internal_reward(errors, initial_error, discount_rate)
            episode_memory[:, 2] = rewards
            states = episode_memory[:, 0]

        return episode_memory