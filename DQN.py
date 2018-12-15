# First we import some libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, Adam


class Movie(object):
    """
    record: 用户观看记录
    user_ids: record 对应的userId数组    
    user_movies: 用户观看电影dict{user_id：[movie one-hot]}
    index: 记录指针
    """
    def __init__(self, records=None, userIds=None, userMovies=None, num_movies=2623):
        self.record = records
        self.user_ids = userIds
        self.user_movies = userMovies
        self.num_actions = num_movies
        self.reset()

    def _update_state(self, reward, action):
        """
        Input: action
        """
        if reward > 0:
            s = action.astype(np.int) ^ self.state.astype(np.int)
            self.state = s
        elif not self._user_over():
            s = self.state
            self.index += 1
            s1 = s.astype(np.int) ^ self.state.astype(np.int)
            self.state = s1

    def _get_reward(self, action):
        reward = np.dot(self.record[self.index], action.T)
        return reward

    def _is_over(self):
        if self.index >= len(self.record) - 1:
            return True        
        return False
    
    def _user_over(self):
        if self._is_over():
            return True        
        if self.user_ids[self.index] == self.user_ids[self.index+1]:
            return False
        return True

    def observe(self):
        self.index += 1
        state = np.array(self.record[self.index])
        return state

    def act(self, action):
        all_over = self._is_over()
        user_over = self._user_over()
        reward = self._get_reward(action)
        self._update_state(reward, action)
        return self.state, reward, all_over, user_over

    def reset(self):
        self.index = -1
        self.state = self.observe()
        
    def get_user(self):
        return self.user_ids[self.index]
        
    def evaluate(self, pred, K):
        t = self.vector[self.user_index]
        p = np.dot(t, pred) / (1.0 * K)
#         r = np.dot(t, pred) / sum(t)
        # print(sum(t), sum(pred))
        # print('precison: ', p, 'recall: ', r)
        return p


class ExperienceReplay(object):
    """
    During recommending movies all the experiences < s, a, r, s’ > are stored in a replay memory.
    s: N-dim array. N is the number of movies
    a: N-dim array, in which there is k ones and others are zeros. 1 means movie indexes recommending
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=100, discount=.9, K=10):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.K = K

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # the number of actions is the number of all the movies can be recommended in the system
        # num_actions = model.output_shape[-1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), num_actions))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros(inputs.shape)

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]

            """
            If the recommend ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1.reshape((1, -1)))[0])

            # if the game ended, the reward is the final reward
            action_t_i = np.argmax(action_t)
            if game_over:  # if game_over is True
                targets[i, action_t_i] += reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t_i] += reward_t + self.discount * Q_sa
        return inputs, targets


def baseline_model(input_size, num_actions, hidden_size):
    # seting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.1), "mse")
    return model


def train(model, epochs, verbose=1):
    # Train
    # Reseting the win counter
    total_reward = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    reward_hist = []
    loss_hist = []
    # 训练轮数： epochs
    for e in range(epochs):
        loss = 0.
        # Resetting the recommender environment
        env.reset()
        all_over = False
        # 一个推荐回合将数据集跑一遍
        while not all_over:
            # get initial input
            input_t = env.observe()
            # 一个用户，K次推荐动作
            for i in range(K):
                # 
                input_tm1 = input_t.reshape((1,-1))
                
                action = np.zeros(num_actions)
                if np.random.rand() <= epsilon:
                    # Eat something random from the menu
                    action[np.random.randint(0, num_actions, size=1)] = 1
                else:
                    # Choose yourself
                    # q contains the expected rewards for the actions
                    q = model.predict(input_tm1)
                    # We pick the action with the highest expected reward
                    action[np.argmax(q[0])] = 1
                
                #recommend_movies.add(np.argmax(action))
                # predi = np.array(predi, dtype=np.int) | np.array(action, dtype=np.int)
                # apply action, get rewards and new state
                input_t, reward, all_over, k_over = env.act(action)
                # If we managed to catch the fruit we add 1 to our win counter
                total_reward += reward

                """
                The experiences < s, a, r, s’ > we make during gameplay are our training data.
                Here we first save the last experience, and then load a batch of experiences to train our model
                """
                # print("action: {:d} reward:{:f}".format(np.argmax(action), reward))
                # store experience
                exp_replay.remember([input_tm1, action, reward, input_t], all_over)

                # Load batch of experiences
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

                # train model on experiences
                batch_loss = model.train_on_batch(inputs, targets)

                # print(loss)
                loss += batch_loss
                
                if k_over:
                    break
            loss_hist.append(loss)
          
        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Reward: {}".format(e, epochs, loss, total_reward))
        reward_hist.append(total_reward)
        
    return reward_hist
def test(model, testset, movies):
    # Define recommender environment
    env = Movie(record, users, u_movies, num_actions)
    total_reward = 0

    # Resetting the recommender environment
    env.reset()
    all_over = False
    
    eval_res = []
    # 一个推荐回合将数据集跑一遍
    while not all_over:
        recommend_movies = set()
        # get initial input
        input_t = env.observe()
        u_id = env.get_user()
        # 一个用户，K次推荐动作
        for i in range(K):
            # 
            input_tm1 = input_t.reshape((1,-1))

            action = np.zeros(num_actions)
            if np.random.rand() <= epsilon:
                # Eat something random from the menu
                action[np.random.randint(0, num_actions, size=1)] = 1
            else:
                # Choose yourself
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                action[np.argmax(q[0])] = 1

            recommend_movies.add(np.argmax(action)+1)
#             if len(recommend_movies) >= K:
#                 break
            # predi = np.array(predi, dtype=np.int) | np.array(action, dtype=np.int)
            # apply action, get rewards and new state
            input_t, reward, all_over, k_over = env.act(action)

            # total_reward += reward
        # evaluate precision and diversity
        print(recommend_movies)
        precision, diversity = precision_diversity(testset, recommend_movies, u_id, movies)
        print(precision, diversity)
        eval_res.append([precision, diversity])
        
    return np.array(eval_res)

def user_movies_dic(data):
    dic = dict()
    for index, row in data.iterrows():
        if row['user'] not in dic:
            dic[row['user']] = [row['movie']]
        else:
            dic[row['user']].append(row['movie'])
    return dic

def get_onehot_dict(data, n_movies):
    dic = dict()
    for index, row in data.iterrows():
        if row['user'] not in dic:
            v = [0] * n_movies
            v[row['movie']-1] = 1
            dic[row['user']] = v
        else:
            dic[row['user']][row['movie']-1] = 1
    return dic

def get_onehot(df, n_movies):
    ret = []    
    for x in df['movie']:
        v = [0] * n_movies
        v[x-1] = 1
        ret.append(v)
    return ret
        
def get_traindata(df1, df2, n_movies):
    record = get_onehot(df1, n_movies)
    #   
    users = df1['user']
    
    #
    userMovies = get_onehot_dict(df2, n_movies)
    
    return record, users, userMovies

def gen_data(trainset, alpha):
    trainset.sort_values(by='user', inplace=True)
    users = trainset['user']

    grouped = trainset.groupby(trainset['user'])
    user_counts = dict(grouped.size())
    
    df1 = pd.DataFrame(columns=['user','movie','rating','timestamp'])
    df2 = pd.DataFrame(columns=['user','movie','rating','timestamp'])
    i = 0
    last_u = trainset['user'].values[0]
    for index, row in trainset.iterrows():
        u = row['user']
        if u != last_u:
            i = 0
            last_u = u
        n = user_counts[u]
        if i < n * alpha:
            df1 = df1.append(row, ignore_index=True)
        else:
            df2 = df2.append(row, ignore_index=True)
        i += 1
    return df1, df2


def get_dict(data):
    dic = dict()
    for index, row in data.iterrows():
        
        if row['user'] not in dic:
            dic[row['user']] = [row['movie']]
        dic[row['user']].append(row['movie'])
    return dic

def cos_similarity(movies, i, j):
    genres_cols = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    vi =movies[movies.movie_id==i][genres_cols]
    vj =movies[movies.movie_id==j][genres_cols]

    vi = np.array(vi)[0]
    vj = np.array(vj)[0]
    return np.dot(vi, vj) / (1.0 * np.sqrt(np.dot(vi, vi)) * np.sqrt(np.dot(vj, vj)))
    
def diversity(rank, movies):
    sim = 0
    rank = list(rank)
    l = len(rank)
    for i in rank:
        for j in rank:
            if i == j:
                continue
            sim += cos_similarity(movies, i, j)
    return 1 - (sim / (0.5 * l * (l-1)))

def precision_diversity(testset, recommend_movies, u_id, movies):
    hit, p = 0, 0
    if u_id not in testset:
        return 0, 0
    for m in recommend_movies:
        if m in testset[u_id]:
            hit += 1
    p = hit / (1.0 * len(recommend_movies))
    div = diversity(recommend_movies, movies)
    return p, div

dpath = 'G:/dataset/RecSys/MovieLens/ml-100k/'
cols = ['user','movie','rating','timestamp']
trainset = pd.read_csv(dpath + 'ua.base',delimiter='\t',names = cols)
# test = pd.read_csv(dpath + 'ua.test',delimiter='\t',names = cols)
# x_train, users = get_traindata(trainset)
mcols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
         'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv(dpath + 'u.item',delimiter='|',names = mcols)
n_movies = len(movies)

df1, df2 = gen_data(trainset, 0.7)
record, users, u_movies = get_traindata(df1, df2, n_movies)

# parameters
epsilon = .1  # exploration
num_actions = n_movies # [move_left, stay, move_right]
max_memory = 500 # Maximum number of experiences we are storing
hidden_size = 50 # Size of the hidden layers
batch_size = 1 # Number of experiences we use for training per batch
input_size = num_actions # Size of the playing field
K = 20 # number of movies recommended per time

# Define environment/recommender
env = Movie(record, users, user_movies_dic, num_actions)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

# Define model
model = baseline_model(input_size, num_actions,hidden_size)
model.summary()

# Number of games played in training, I found the model needs about 4,000 games till it plays well
epoch = 10
# Train the model
# For simplicity of the noteb
hist = train(model, epoch, verbose=1)
print("Training done")

testset = pd.read_csv(dpath + 'ua.test',delimiter='\t',names = ['user', 'movie', 'rating', 'timestamp'])
test_dic = get_dict(testset)

res = test(model, test_dic, movies)
