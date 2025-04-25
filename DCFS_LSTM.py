import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import shutil
import sys
import timeit
import tensorflow.keras.backend as K
from collections import deque
import threading
import pickle
import tensorflow as tf

sys.path.append(os.path.abspath(""))

from classes.agent import Agent
from classes.enviorement import Enviorement
from classes.general_functions import genreal_func
from classes.run_process import run_process

genreal_func = genreal_func()
run_process = run_process()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# parameters

file = sys.argv[1]
discount_rate = float(sys.argv[2])
iterations = int(sys.argv[3])
location = sys.argv[4]
experiment_name = sys.argv[5]
rnn_unit = 10


K.clear_session()
experiment = (
    str(experiment_name)
    + "_"
    + str(file)
    + "_LSTM_"
    + str(discount_rate)
    + "_"
    + str(iterations)
)


number_of_experiment = 3

# Dataset parameters #


train_file_name = file + "_Train_Data.csv"
test_file_name = file + "_Val_Data.csv"


copy = 5
final_threads = 5
l_r_schedule = [(0, 0.1), (0.25, 0.01), (0.5, 0.001), (0.75, 0.0001)]
epsilon_schedule = [(0, 0.9), (0.25, 0.5), (0.5, 0.3), (0.75, 0.01)]
threads_schedule = [(0, final_threads)]


# Learner and episode parameters


episode_size = 800
external_threshold = 0
internal_threshold = -1000
learner_model = "DT"


# Experiments folder management:
if not os.path.exists("Experiments/" + str(experiment)):
    os.makedirs("Experiments/" + str(experiment))
else:
    shutil.rmtree("Experiments/" + str(experiment))
    os.makedirs("Experiments/" + str(experiment))

# Create Excel writer
excel_file_path = os.path.join("Experiments", str(experiment), "results.xlsx")
writer = pd.ExcelWriter(excel_file_path)


# Set the number of episodes to run:
episodes = iterations
print("number of episodes: {}".format(episodes))

# define data frame to save episode policies results
df = pd.DataFrame(
    columns=(
        "episode",
        "policy_columns",
        "policy_accuracy_train",
        "policy_accuracy_test",
    )
)


class actorthread(threading.Thread):
    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self.thread_id = thread_id

    def run(self, env, agent, episode_size, eps):
        global x_for_train
        global y_for_train
        global a_for_train

        threadLock.acquire()

        episode_memory_store = run_process.runprocess(
            env, agent, episode_size, eps, discount_rate
        )
        if np.array(episode_memory_store).shape[0] > 0:
            episode_memory_store = episode_memory_store.reshape(
                1, episode_memory_store.shape[0], episode_memory_store.shape[1]
            )

            for episode_memory in episode_memory_store:
                X_batch, a_batch, y_batch = genreal_func.create_batch(
                    np.array(episode_memory), discount_rate, agent
                )
                x_for_train.append(X_batch)
                y_for_train.append(y_batch)
                a_for_train.append(a_batch)
        #         for item in memory:
        #             episode_memory_store.append(item)
        threadLock.release()


#         return list(episode_memory_store)


for e in range(number_of_experiment):
    #     if not os.path.exists('Experiments/'+ str(experiment)+ '/'+ str(e)):
    #         os.makedirs('Experiments/'+ str(experiment)+ '/'+ str(e))
    #     else:
    #         shutil.rmtree('Experiments/'+ str(experiment)+ '/'+ str(e))          #removes all the subdirectories!
    #         os.makedirs('Experiments/'+ str(experiment)+ '/'+ str(e))
    if os.path.exists("log.csv"):
        os.remove("log.csv")
    print("Experiments " + str(e) + " start")
    print("discount_factor:  " + str(discount_rate))

    current_lr = l_r_schedule[0][1]

    ##########################Experiment setup##########################
    start = timeit.default_timer()

    ## new  - for memory replay
    x_memory = deque()  # []
    y_memory = deque()  # []
    a_memory = deque()  # []
    env = Enviorement(location, train_file_name, test_file_name, learner_model)
    agent = Agent(internal_threshold, external_threshold, rnn_unit, env)
    # copy inference model initial weight
    #     agent.updae_model_for_infereance_weights()
    agent.complie(current_lr)

    for epi in range(episodes):
        x_for_train = []
        y_for_train = []
        a_for_train = []

        episode_start = timeit.default_timer()
        print("episode {} start".format(epi))
        l_r = genreal_func.learningRate(epi, l_r_schedule, episodes)
        if l_r != current_lr:
            print("complie model")
            agent.complie(l_r)
            current_lr = l_r

        eps = genreal_func.epsilon(epi, epsilon_schedule, episodes)

        threads = genreal_func.epsilon(epi, threads_schedule, episodes)
        print(f"threads: {threads}")
        n_memory = threads
        batch_size = threads

        threadLock = threading.Lock()
        jobs = []

        for i in range(0, threads):
            thread = actorthread(i)
            jobs.append(thread)

        for j in jobs:
            j.run(env, agent, episode_size, eps)

        x_for_train = np.array(x_for_train)
        x_for_train = x_for_train.reshape(
            x_for_train.shape[0], x_for_train.shape[2], x_for_train.shape[3]
        )
        y_for_train = np.array(y_for_train)
        #         y_for_train= y_for_train.reshape(y_for_train.shape[0], y_for_train.shape[2],y_for_train.shape[3])
        a_for_train = np.array(a_for_train)
        #         a_for_train= a_for_train.reshape(a_for_train.shape[0], a_for_train.shape[2],a_for_train.shape[3])

        ## new  - for memory replay
        for raw in x_for_train:
            x_memory.append(raw)
            if len(x_memory) > n_memory:
                x_memory.popleft()
        for raw in y_for_train:
            y_memory.append(raw)
            if len(y_memory) > n_memory:
                y_memory.popleft()
        for raw in a_for_train:
            a_memory.append(raw)
            if len(a_memory) > n_memory:
                a_memory.popleft()

        agent.train(
            [
                np.array(x_memory),
                np.array(a_memory).reshape(
                    np.array(a_memory).shape[0], np.array(a_memory).shape[1], 1
                ),
            ],
            np.array(y_memory).reshape(
                np.array(y_memory).shape[0],
                np.array(y_memory).shape[2],
                np.array(y_memory).shape[1],
            ),
            min(np.array(x_memory).shape[0], batch_size),
        )
        agent.updae_model_for_infereance_weights()

        if epi % copy == 0:
            print("update traget network")
            agent.updae_model_for_infereance_weights()

        ## policy results
        policy_s = env.action_space
        policy_selected_actions = env.action_space  # new
        policy_done = 0
        policy_order = []
        policy_a_Q_list = []

        agent.clear_inference_model_state()

        while not policy_done:
            policy_a, policy_a_Q = agent.act(policy_s, 1, eps, policy_selected_actions)
            policy_order.append(policy_a)
            policy_a_Q_list.append(policy_a_Q)
            if policy_a_Q >= agent.external_threshold:
                policy_s2, policy_selected_actions = env.s2(
                    policy_s, policy_a, policy_selected_actions
                )
                policy_done = agent.is_done(
                    policy_a_Q, policy_selected_actions, policy_calc=1
                )
                policy_s = policy_s2
            else:
                policy_done = 1

        policy_columns = np.where(policy_selected_actions == 1)[0]
        print("policy order: {}".format(policy_order))
        print("policy policy_a_Q_list: {}".format(policy_a_Q_list))

        # Calculate policy train accuracy
        if len(policy_columns) == 0:
            policy_accuracy_train = 0
            policy_accuracy_test = 0
        else:
            policy_episode_data = env.get_data(episode_size, 0, "train")
            policy_X, policy_y = env.data_separate(policy_episode_data)
            policy_X_train_main, policy_X_test_main, policy_y_train, policy_y_test = (
                env.data_split(policy_X, policy_y)
            )
            policy_accuracy_train = env.accuracy(
                policy_selected_actions,
                policy_X_train_main,
                policy_X_test_main,
                policy_y_train,
                policy_y_test,
            )
            #             policy_accuracy_train = env.accuracy(policy_selected_actions, policy_X, policy_X, policy_y, policy_y)

            # Calculate policy test accuracy
            test_episode_data = env.get_data(episode_size, 0, "test")
            test_X, test_y = env.data_separate(test_episode_data)
            #             test_X_train_main, test_X_test_main, test_y_train, test_y_test =  env.data_split(test_X,test_y)
            policy_accuracy_test = env.accuracy(
                policy_selected_actions, policy_X, test_X, policy_y, test_y
            )
        #             policy_accuracy_test = env.accuracy(policy_selected_actions, policy_X, test_X, policy_y, test_y)

        print("policy train accuracy: {} ".format(policy_accuracy_train))
        print("policy test accuracy: {} ".format(policy_accuracy_test))

        if sum(policy_selected_actions) > 0:
            # Append results to DataFrame
            # df = df.append(
            #     {
            #         "episode": str(epi + 1),
            #         "policy_columns": str(policy_columns),
            #         "policy_accuracy_train": policy_accuracy_train,
            #         "policy_accuracy_test": policy_accuracy_test,
            #     },
            #     ignore_index=True,
            # )
            new_row = pd.DataFrame([{
                "episode": str(epi + 1),
                "policy_columns": str(policy_columns),
                "policy_accuracy_train": policy_accuracy_train,
                "policy_accuracy_test": policy_accuracy_test,
            }])

            df = pd.concat([df, new_row], ignore_index=True)


        episode_stop3 = timeit.default_timer()
    #         print ("episode time: {}".format(episode_stop3 - episode_start))

    #    df.to_excel(writer, 'Experiment' + str(e))
    #    df_plot=df[['episode','policy_accuracy_train','policy_accuracy_test']]
    #     plot=df_plot.plot()
    #     fig = plot.get_figure()
    #     fig.savefig('Experiments/'+ str(experiment) + '/plot_experiment_' + str(e) +'.png')

    print(
        "episode policy: {}, number of features: {} ".format(
            policy_columns, len(policy_columns)
        )
    )
    print("policy train accuracy: {} ".format(policy_accuracy_train))
    print("policy test accuracy: {} ".format(policy_accuracy_test))
    stop = timeit.default_timer()

    if not os.path.exists("Experiments/" + str(experiment) + "/experiment_dict.pickle"):
        # Create the experiment dictionary file in a cross-platform way
        dict_filepath = os.path.join(
            "Experiments", str(experiment), "experiment_dict.pickle"
        )
        with open(dict_filepath, "w") as f:
            pass  # Just create the empty file
        # os.mknod("Experiments/" + str(experiment) + "/experiment_dict.pickle") # Removed os.mknod

        # Store experiment data
        experiment_dict = {}
        experiment_dict[e] = [policy_order, stop - start]

        with open(
            "Experiments/" + str(experiment) + "/experiment_dict.pickle", "wb"
        ) as handle:
            pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            "Experiments/" + str(experiment) + "/experiment_dict.pickle", "rb"
        ) as handle:
            experiment_dict = pickle.load(handle)
        experiment_dict[e] = [policy_order, stop - start]

        with open(
            "Experiments/" + str(experiment) + "/experiment_dict.pickle", "wb"
        ) as handle:
            pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #    experiment_dict[e]=[policy_order,stop - start]

    #    experiment_dict[e]=[policy_order,policy_a_Q_list]

    #     clear tensorflow session
    K.clear_session()

# Save results after the loop
df.to_excel(writer, sheet_name='Results', index=False)
writer.save()
print(f"Results saved to {excel_file_path}")

# Save pickles and experiment dict
print(
    "episode policy: {}, number of features: {} ".format(
        policy_columns, len(policy_columns)
    )
)
print("policy train accuracy: {} ".format(policy_accuracy_train))
print("policy test accuracy: {} ".format(policy_accuracy_test))
stop = timeit.default_timer()

if not os.path.exists("Experiments/" + str(experiment) + "/experiment_dict.pickle"):
    # Create the experiment dictionary file in a cross-platform way
    dict_filepath = os.path.join(
        "Experiments", str(experiment), "experiment_dict.pickle"
    )
    with open(dict_filepath, "w") as f:
        pass  # Just create the empty file
    # os.mknod("Experiments/" + str(experiment) + "/experiment_dict.pickle") # Removed os.mknod

    # Store experiment data
    experiment_dict = {}
    experiment_dict[0] = [policy_order, stop - start]

    with open(
        "Experiments/" + str(experiment) + "/experiment_dict.pickle", "wb"
    ) as handle:
        pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(
        "Experiments/" + str(experiment) + "/experiment_dict.pickle", "rb"
    ) as handle:
        experiment_dict = pickle.load(handle)
    experiment_dict[0] = [policy_order, stop - start]

    with open(
        "Experiments/" + str(experiment) + "/experiment_dict.pickle", "wb"
    ) as handle:
        pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#    experiment_dict[e]=[policy_order,policy_a_Q_list]

# writer.save()


# with open('Experiments/'+ str(experiment) + '/experiment_dict.pickle', 'wb') as handle:
#    pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

stop = timeit.default_timer()
print(stop - start)
