import pandas as pd
import numpy as np
import os
import shutil
import sys
import timeit
import threading
import pickle
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from collections import deque


import tensorflow as tf
from numpy.random import seed


from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Activation,
    BatchNormalization,
    MaxPooling2D,
    LSTM,
    Flatten,
    Lambda,
    Subtract,
    Multiply,
    Add,
    concatenate,
)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import relu, elu
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)

import warnings

warnings.filterwarnings("ignore")


# #### Define the parameters

file = sys.argv[1]
discount_rate = float(sys.argv[2])
iterations = int(sys.argv[3])
location = sys.argv[4]
experiment_name = sys.argv[5]

K.clear_session()
experiment = (
    str(experiment_name)
    + "_"
    + str(file)
    + "_DQN_"
    + str(discount_rate)
    + "_"
    + str(iterations)
)


number_of_experiment = 10


# Experiment:

number_of_experiment = 10

# Dataset parameters #


train_file_name = file + "_Train_Data.csv"
test_file_name = file + "_Val_Data.csv"


# l_r_schedule =[(0,0.05), (0.3,0.01), (0.6, 0.005), (0.9, 0.0001)]
# l_r_schedule = [(0,0.09), (0.25, 0.05), (0.5, 0.01), (0.75, 0.005)]
l_r_schedule = [(0, 0.05), (0.25, 0.01), (0.5, 0.005), (0.75, 0.001)]

# epsilon_schedule = [(0,0.9), (0.25, 0.5), (0.5, 0.3), (0.75, 0.1)]
epsilon_schedule = [(0, 0.9), (0.25, 0.5), (0.5, 0.1), (0.75, 0.01)]
# Learner and episode parameters

episode_size = 800
internal_threshold = -999
value_threshold = 0
learner_model = "DT"
batch_size = 5
threads = 5
update_target_net = 5
n_memory = 50  # 1000


# Experiments folder management:
if not os.path.exists("Experiments/" + str(experiment)):
    os.makedirs("Experiments/" + str(experiment))
else:
    shutil.rmtree("Experiments/" + str(experiment))
    os.makedirs("Experiments/" + str(experiment))
writer = pd.ExcelWriter("Experiments/" + str(experiment) + "/results.xlsx")

# experiment_dict = {}


class Enviorement:
    def __init__(self, location, train_file_name, test_file_name, learner_model):
        self.data = pd.read_csv(location + "/" + train_file_name)
        self.test_data = pd.read_csv(location + "/" + test_file_name)
        self.action_space = np.zeros(self.data.shape[1] - 1)
        self.data_size = len(self.data)
        self.learner_model = learner_model
        self.number_of_classes = self.data.iloc[:, -1].unique().shape[0]

    # Function that create the episode data - sample randomly
    def get_data(self, episode_size, for_episode, mode):
        # global dataset # Removed global as it's not best practice and return value is used
        if mode == "train":
            # Sample episode_size rows from the training data
            # Handle cases where episode_size is larger than available data
            n_samples = len(self.data)
            actual_sample_size = min(episode_size, n_samples)
            if actual_sample_size < episode_size:
                print(
                    f"Warning: Requested episode_size ({episode_size}) is larger than available training samples ({n_samples}). Using {actual_sample_size} samples."
                )

            # Use pandas sample method
            dataset = self.data.sample(
                n=actual_sample_size, random_state=np.random.randint(1, 10000)
            )  # Added random state seeding

        elif mode == "test":
            dataset = self.test_data  # Use the validation data as is
        else:
            raise ValueError(f"Invalid mode specified: {mode}")

        return dataset

    def data_separate(self, dataset):
        global X
        global y
        X = dataset.iloc[
            :, 0 : dataset.shape[1] - 1
        ]  # all rows, all the features and no labels
        y = dataset.iloc[:, -1]  # all rows, label only
        return X, y

    # Function that split the episode data into train and test
    def data_split(self, X, y):
        from sklearn.model_selection import train_test_split

        X_train_main, X_test_main, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=4
        )
        return X_train_main, X_test_main, y_train, y_test

    def s2(self, s, a, selected_actions):
        s2 = to_categorical(a, len(self.action_space))
        sel_actions = selected_actions.copy()
        sel_actions[a] = 1
        return s2, sel_actions

    def accuracy(self, s2, X_train_main, X_test_main, y_train, y_test):
        columns = np.where(s2 == 1)[0]
        X_train = X_train_main.iloc[:, columns]
        X_test = X_test_main.iloc[:, columns]
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        accuracy = self.leraner(X_train, X_test, y_train, y_test)
        return accuracy

    def leraner(self, X_train, X_test, y_train, y_test):
        if self.learner_model == "DT":
            learner = tree.DecisionTreeClassifier()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        elif self.learner_model == "NB":
            learner = MultinomialNB()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return accuracy


class Agent:
    def __init__(self, value_threshold):
        self.env = env
        self.input_dim = env.data.shape[1] - 1
        self.output_dim = env.data.shape[1] - 1
        self.target_model = self.create_Q_model()
        self.q_model = self.create_Q_model()
        self.full_model = self.create_full_model()
        self.threshold = value_threshold

    def create_Q_model(self):
        x = Input((self.input_dim,))
        output = Dense(
            self.output_dim,
            kernel_initializer="VarianceScaling",
            activation="linear",
            name="output",
        )(x)  # Ones
        model = Model(x, output)
        return model

    def create_full_model(self):
        input_state = Input(self.input_dim)
        action_id = Input(shape=[1], dtype="uint8")
        predicted_Qvalues = self.q_model(input_state)
        leave_only_action_value = tf.one_hot(action_id, self.output_dim)
        leave_only_action_value = tf.squeeze(leave_only_action_value, axis=1)
        predicted_q_for_action = tf.reduce_sum(
            predicted_Qvalues * leave_only_action_value, axis=1, keepdims=True
        )
        return Model([input_state, action_id], predicted_q_for_action)

    def complie(self, l_r):
        adam = Adam(lr=l_r)
        self.full_model.compile(loss="logcosh", metrics=["mse"], optimizer=adam)

    def act(self, state, policy_calc, eps, selected_actions):
        available_actions = np.ones(self.input_dim) - selected_actions

        state = state.reshape(-1, self.input_dim)
        Q_values = self.q_model.predict(state).flatten()
        avilable_actions_Q = available_actions * Q_values
        avilable_actions_Q_threshold = avilable_actions_Q - (selected_actions * 999)
        available_actions_unif_prob = available_actions / sum(available_actions)

        if policy_calc == 0:
            if np.random.rand() < eps:
                a = np.random.choice(self.output_dim, 1, p=available_actions_unif_prob)[
                    0
                ]
                a_Q = internal_threshold
            else:
                if (
                    len(
                        avilable_actions_Q_threshold[
                            avilable_actions_Q_threshold > internal_threshold
                        ]
                    )
                    > 0
                ):  # self.threshold
                    a = np.argmax(avilable_actions_Q_threshold)
                    a_Q = Q_values[a]
                else:
                    a = -999
                    a_Q = -999
        else:
            if (
                len(
                    avilable_actions_Q_threshold[
                        avilable_actions_Q_threshold > value_threshold
                    ]
                )
                > 0
            ):
                a = np.argmax(avilable_actions_Q_threshold)
                a_Q = Q_values[a]
            else:
                a = -999
                a_Q = -999

        return a, a_Q

    def is_done(self, a_Q, selected_actions):  ### need to fix
        done = 0
        if (
            a_Q < internal_threshold or sum(selected_actions == 1) == self.input_dim
        ):  # self.threshold
            done = 1
        return done

    def train(self, X_batch, y_batch, batch_size):
        print(batch_size)
        #         return self.model.train_on_batch(X_batch, y_batch)
        return self.full_model.fit(
            X_batch,
            y_batch,
            epochs=5,
            steps_per_epoch=10,
            batch_size=batch_size,
            verbose=0,
        )  # , callbacks=[csv_logger])

    def update_target_network(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def predict(self, X_batch):
        return self.target_model.predict(X_batch)


# def internal_reward(e, initial_error, discount_rate):
#     ## Compute the gamma-discounted rewards over an episode
#     r = np.zeros_like(e)
#     r[0] = initial_error - e[0]
#     for t in range(1, len(e)):
#         r[t] = e[t - 1]- e[t]
#     return r


def internal_reward(e, initial_error, discount_rate):
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


def create_batch(memory, discount_rate):
    sample = np.asarray(memory)

    s = sample[:, 0]
    a = sample[:, 1].astype(np.int8)
    r = sample[:, 2]
    s2 = sample[:, 3]
    d = sample[:, 4] * 1.0
    available_actions = sample[:, 5]
    available_actions = np.vstack(available_actions)

    X_batch = np.vstack(s)

    future_rewards = agent.predict(np.vstack(s2))
    # discount factor * expected future reward
    discount_future_rewards = discount_rate * tf.reduce_max(
        future_rewards * available_actions, axis=1
    )
    discount_future_rewards = 0 * tf.reduce_max(
        future_rewards * available_actions, axis=1
    )

    # if <0 set to 0
    discount_future_rewards = tf.maximum(discount_future_rewards, 0)
    # Q value = reward + discount_future_rewards
    updated_q_values = r + discount_future_rewards
    # If final state set the last value to -1
    updated_q_values = updated_q_values * (1 - d)

    return [X_batch, a], updated_q_values


def runprocess(thread_id):
    #     print('thread_id {} start'.format(thread_id))
    memory = deque()
    initial_error = 1 - (1 / env.number_of_classes)
    #     agent.complie(l_r)

    episode_data = env.get_data(episode_size, 1, "train")
    X, y = env.data_separate(episode_data)
    X_train_main, X_test_main, y_train, y_test = env.data_split(X, y)
    s = env.action_space
    selected_actions = env.action_space  # new
    done = 0

    while not done:
        #         print('thread_id {} , s {}'.format(thread_id, s))
        a, a_Q = agent.act(s, 0, eps, selected_actions)
        if a_Q >= internal_threshold:
            s2, selected_actions = env.s2(s, a, selected_actions)
            accuracy = env.accuracy(
                selected_actions, X_train_main, X_test_main, y_train, y_test
            )
            error = 1 - accuracy
            done = agent.is_done(a_Q, selected_actions)
            available_actions = np.ones(len(s)) - selected_actions
            memory.append([s, a, error, s2, done, available_actions])

            s = s2
        else:
            done = 1

    # Convert memory to numpy array, specifying dtype=object to handle mixed types/shapes
    episode_memory = np.asarray(memory, dtype=object)
    if len(episode_memory) > 0:
        # Use standard Python float instead of deprecated np.float
        errors = episode_memory[:, 2].astype(float)
        rewards = internal_reward(errors, initial_error, discount_rate)
        #     discounted_rewards = discount(rewards, discount_rate)
        episode_memory[:, 2] = rewards
        states = episode_memory[:, 0]
    return episode_memory


class actorthread(threading.Thread):
    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        global episode_memory_store
        threadLock.acquire()
        memory = runprocess(self.thread_id)
        for item in memory:
            episode_memory_store.append(item)
        threadLock.release()


#         return list(episode_memory_store)


def learningRate(epi, l_r_schedule):
    l_r = 0.0
    for episode_boundary, lr in l_r_schedule:
        if epi >= episodes * episode_boundary:
            l_r = lr
    print("epoch, lr: ", epi, l_r)
    return l_r


def epsilon(epi, epsilon_schedule):
    eps = 0.0
    for episode_boundary, epsilon in epsilon_schedule:
        if epi >= episodes * episode_boundary:
            eps = epsilon
    print("epoch, epsilon: ", epi, eps)
    return eps


# #### 4. Run all experiments


import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger("log.csv", append=True, separator=";")

for e in range(number_of_experiment):
    print("Experiments " + str(e) + " start")
    current_lr = l_r_schedule[0][1]
    ##########################Experiment setup##########################
    start = timeit.default_timer()

    env = Enviorement(location, train_file_name, test_file_name, learner_model)
    agent = Agent(value_threshold)
    agent.complie(current_lr)
    # Set the number of episodes to run:
    episodes = int(iterations / threads)
    print("number of episodes: {}".format(episodes))
    print("batch_sizes: {}".format(batch_size))

    # define data frame to save episode policies results
    df = pd.DataFrame(
        columns=(
            "episode",
            "policy_columns",
            "policy_accuracy_train",
            "policy_accuracy_test",
        )
    )

    for epi in range(episodes):
        print("episode {} start".format(epi))

        l_r = learningRate(epi, l_r_schedule)
        eps = epsilon(epi, epsilon_schedule)

        if l_r != current_lr:
            print("complie model")
            agent.complie(l_r)
            current_lr = l_r

        episode_memory_store = deque()
        memory = deque()

        threadLock = threading.Lock()
        jobs = []

        for i in range(0, threads):
            thread = actorthread(i)
            jobs.append(thread)

        for j in jobs:
            j.run()

        if epi % update_target_net == 0:
            print("update the the target network with new weights")
            agent.update_target_network()

        if len(episode_memory_store) > 0:
            for raw in episode_memory_store:
                memory.append(raw)
                if len(memory) > n_memory:
                    memory.popleft()

            X_batch, y_batch = create_batch(np.array(memory), discount_rate)
            agent.train(X_batch, y_batch, batch_size=threads)

            ## policy results
            policy_s = env.action_space
            policy_selected_actions = env.action_space  # new
            policy_done = 0
            policy_order = []
            policy_a_Q_list = []
            while not policy_done:
                policy_a, policy_a_Q = agent.act(
                    policy_s, 1, eps, policy_selected_actions
                )
                policy_order.append(policy_a)
                policy_a_Q_list.append(policy_a_Q)
                if policy_a_Q >= agent.threshold:
                    policy_s2, policy_selected_actions = env.s2(
                        policy_s, policy_a, policy_selected_actions
                    )
                    policy_done = agent.is_done(policy_a_Q, policy_selected_actions)
                    policy_s = policy_s2
                else:
                    policy_done = 1

            policy_columns = np.where(policy_selected_actions == 1)[0]

            if sum(policy_selected_actions) > 0:
                # Calculate policy train accuracy
                policy_episode_data = env.get_data(episode_size, 0, "train")
                policy_X, policy_y = env.data_separate(policy_episode_data)
                (
                    policy_X_train_main,
                    policy_X_test_main,
                    policy_y_train,
                    policy_y_test,
                ) = env.data_split(policy_X, policy_y)
                policy_accuracy_train = env.accuracy(
                    policy_selected_actions,
                    policy_X_train_main,
                    policy_X_test_main,
                    policy_y_train,
                    policy_y_test,
                )

                # Calculate policy test accuracy
                test_episode_data = env.get_data(episode_size, 0, "test")
                test_X, test_y = env.data_separate(test_episode_data)
                #                 test_X_train_main, test_X_test_main, test_y_train, test_y_test =  env.data_split(test_X,test_y)
                #                 policy_accuracy_test = env.accuracy(policy_selected_actions, test_X_train_main, test_X_test_main, test_y_train, test_y_test)
                policy_accuracy_test = env.accuracy(
                    policy_selected_actions, policy_X, test_X, policy_y, test_y
                )

                print(
                    "episode policy: {}, number of features: {} ".format(
                        policy_columns, len(policy_columns)
                    )
                )
                print("policy order: {}".format(policy_order))
                print('policy_a_Q_list: {}'.format(policy_a_Q_list))

                # Uncommented lines to append data to DataFrame and save to Excel sheet
                print('policy train accuracy: {} '.format(policy_accuracy_train))
                print('policy test accuracy: {} '.format(policy_accuracy_test))
                df.loc[len(df)] = {
                    'episode': str(epi + 1),
                    'policy_columns': str(policy_columns),
                    'policy_accuracy_train': policy_accuracy_train,
                    'policy_accuracy_test': policy_accuracy_test,
                }

        df.to_excel(writer, 'Experiment' + str(e))
        # df_plot=df[['episode','policy_accuracy_train','policy_accuracy_test']]
        # plot=df_plot.plot()
        # fig = plot.get_figure()
        # fig.savefig('Experiments/'+ str(experiment) + '/plot_experiment_' + str(e) +'.png')

        #     print('episode policy: {}, number of features: {} '.format(policy_columns, len(policy_columns)))
        #     print('policy train accuracy: {} '.format(policy_accuracy_train))
        #     print('policy test accuracy: {} '.format(policy_accuracy_test))
        #     df=df.append({'episode':str(epi+1), 'policy_columns':str(policy_columns),'policy_accuracy_train':policy_accuracy_train,'policy_accuracy_test':policy_accuracy_test}, ignore_index=True)

    #     df.to_excel(writer, 'Experiment' + str(e))
    #     df_plot=df[['episode','policy_accuracy_train','policy_accuracy_test']]
    #     plot=df_plot.plot()
    #     fig = plot.get_figure()
    #     fig.savefig('Experiments/'+ str(experiment) + '/plot_experiment_' + str(e) +'.png')

    #     print('episode policy: {}, number of features: {} '.format(policy_columns, len(policy_columns)))
    #     print('policy train accuracy: {} '.format(policy_accuracy_train))
    #     print('policy test accuracy: {} '.format(policy_accuracy_test))
    #     df=df.append({'episode':str(epi+1), 'policy_columns':str(policy_columns),'policy_accuracy_train':policy_accuracy_train,'policy_accuracy_test':policy_accuracy_test}, ignore_index=True)

    #     df.to_excel(writer, 'Experiment' + str(e))
    #     df_plot=df[['episode','policy_accuracy_train','policy_accuracy_test']]
    #     plot=df_plot.plot()
    #     fig = plot.get_figure()
    #     fig.savefig('Experiments/'+ str(experiment) + '/plot_experiment_' + str(e) +'.png')

    stop = timeit.default_timer()
    if not os.path.exists("Experiments/" + str(experiment) + "/experiment_dict.pickle"):
        # Create the experiment dictionary file in a cross-platform way
        dict_filepath = os.path.join("Experiments", str(experiment), "experiment_dict.pickle")
        with open(dict_filepath, 'w') as f:
            pass # Just create the empty file
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

# with open('Experiments/'+ str(experiment) + '/experiment_dict.pickle', 'wb') as handle:
#    pickle.dump(experiment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

writer.save()
stop = timeit.default_timer()
print(stop - start)
