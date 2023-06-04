# coding = utf-8
# import torch.nn.functional as F
import torch.nn as nn
import torch
from Model import NetModel,DuelingDqn
from collections import deque


import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from newEnv import EightPuzzleEnv
import tqdm
from typing import Deque, Dict, List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
BATCH_SIZE = 64
EPOCH = 5000
EPSILON = 0.4  #
GAMMA = 0.9  # reward discount
TARGET_UPATE = 300  # target update frequency
Memory_capacity = 5000
env = EightPuzzleEnv(3, 3)  # (2, 3)
N_actions = 4
N_states = 9
N_STEP = 6
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
from typing import Dict, List, Tuple

# class ReplayBuffer:
#     """A simple numpy replay buffer."""

#     def __init__(self, obs_dim: int, size: int, batch_size: int = 64):
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.acts_buf = np.zeros([size], dtype=np.float32)
#         self.rews_buf = np.zeros([size], dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.max_size, self.batch_size = size, batch_size
#         self.ptr, self.size, = 0, 0

#     def store(
#         self,
#         obs: np.ndarray,
#         act: np.ndarray, 
#         rew: float, 
#         next_obs: np.ndarray, 
#         done: bool,
#     ):
#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample_batch(self) -> Dict[str, np.ndarray]:
#         idxs = np.random.choice(self.size, size=self.batch_size)
#         return dict(obs=self.obs_buf[idxs],
#                     next_obs=self.next_obs_buf[idxs],
#                     acts=self.acts_buf[idxs],
#                     rews=self.rews_buf[idxs],
#                     done=self.done_buf[idxs])

#     def __len__(self) -> int:
#         return self.size

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 3, 
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

class DQN_Agent(object):
    def __init__(self, lr=1e-4):
        self.eval_net, self.target_net = DuelingDqn(N_states, N_actions).to(device) \
            , DuelingDqn(N_states, N_actions).to(device)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # initialize memory
        self.memory = ReplayBuffer(9,Memory_capacity,BATCH_SIZE,n_step=1,gamma=GAMMA)
        self.memory_n = np.zeros((Memory_capacity, N_states * 2 + 2))  # initialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
                # memory for N-step Learning
        self.use_n_step = True if N_STEP > 1 else False
        if self.use_n_step:
            self.n_step = N_STEP
            self.memory_n = ReplayBuffer(
                9,Memory_capacity,BATCH_SIZE,n_step=self.n_step,gamma=GAMMA
            )

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_actions)
            action = action  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 竖向堆叠, shape: (10,)
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def calculate_loss(self,samples,gamma):
        b_s = torch.FloatTensor(samples['obs']).to(device)
        b_a = torch.LongTensor(samples['acts'].astype(int)).to(device).unsqueeze(dim=1)
        b_r = torch.FloatTensor(samples['rews']).to(device).unsqueeze(dim=1)
        b_s_ = torch.FloatTensor(samples['next_obs']).to(device)

        # import pdb
        # pdb.set_trace()
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        return loss
        
    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_UPATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # TODO
        # 抽取记忆库中的批数据
        # sample_index = np.random.choice(Memory_capacity, BATCH_SIZE)
        # b_memory = self.memory[sample_index, :]
        # b_s = Variable(torch.FloatTensor(b_memory[:, :N_states])).to(device)
        # b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int))).to(device)
        # b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2])).to(device)
        # b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:])).to(device)

        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self.calculate_loss(samples,GAMMA)
        
        if self.use_n_step:
            samples_n = self.memory_n.sample_batch_from_idxs(indices)
            gamma = GAMMA ** self.n_step
            n_loss = self.calculate_loss(samples_n, gamma)
            loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, section):
        torch.save({'model': self.eval_net.state_dict()}, r'torch_models/3last_step{}.pth'.format(section))
        # torch.save({'model': self.target_net.state_dict()}, r'torch_models\dueling_target2x3{}.pth')

    def load_model(self, section=1):
        state_dict = torch.load(r'torch_models/3last_step{}.pth'.format(section))
        self.eval_net.load_state_dict(state_dict['model'])

    def continue_train(self, section):
        state_dict = torch.load(r'torch_models/3last_step{}.pth'.format(section))
        self.eval_net.load_state_dict(state_dict['model'])
        self.target_net.load_state_dict(state_dict['model'])

    def predict(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        action_value = self.eval_net.forward(x).detach()
        action = torch.max(action_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


def train(d_step,dqn, env, episode, section="0"):
    global EPSILON# , delta_ep
    global difficulty_steps
    print('\nCollecting experience...')
    delta_ep = (0.8 - EPSILON) / EPOCH  # 1000个episode，每次给贪婪加delta_ep
    count = 0
    for i_episode in tqdm.tqdm(range(episode)):
        s = env.reset(step=d_step)
        s = s.flatten()
        step_inner = 0
        while True:
            # env.render()
            ep_r = 0
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            s_ = s_.flatten()
            # 存记忆, state, action, reward, next_state

            # dqn.store_transition(s, a, r, s_)
            # TODO
            # dqn.memory.store(s,a,r,s_,done)
            dqn.memory_counter += 1
            transition = [s,a,r,s_,done]
            # N-step transition
            if dqn.use_n_step:
                one_step_transition = dqn.memory_n.store(*transition)
            # 1-step transition
            else:
                one_step_transition = transition
            if one_step_transition:
                dqn.memory.store(*one_step_transition)

            ep_r += r
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                # print("train")
                if done: #and step_inner % 50 == 0:
                    # print('Ep: ', i_episode,
                    #       '| Ep_r: ', round(ep_r, 2), "Epsilon:", EPSILON, end=",,")
                    pass
                step_inner += 1
            if done:
                count += 1
            if done or env.max_episode_steps < step_inner:
                break
            s = s_
        EPSILON += delta_ep
    sucess = count/episode
    print("success:", count/episode)
    dqn.save_model(section)
    return sucess

# class 

def test(d_step,section=1):
    print("start test!")
    agent = DQN_Agent()
    agent.load_model(section)
    count = 0
    N = 500
    for i in range(N):
        step = 0
        s = env.reset(step=d_step)
        print(s)
        s = s.flatten()
        while True:
            a = agent.predict(s)
            s_, r, done, info = env.step(a)
            s_ = s_.flatten()
            step += 1
            if done:
                s = s_
                count += 1
                break
            elif step > 50:
                break
            s = s_
    print("success:", count/N)

def inference(board,section):
    agent = DQN_Agent()
    agent.load_model(section)
    count = 0
    step = 0
    status_sequence = []
    s = env.reset(board=np.array(board))
    status_sequence.append(s)
    s = s.flatten()
    while True:
        a = agent.predict(s)
        s_, r, done, info = env.step(a)
        status_sequence.append(s_)
        s_ = s_.flatten()
        step += 1
        if done:
            s = s_
            count += 1
            break
        elif step > 30:
            break
        s = s_
    return done, status_sequence



def main():
    dqn = DQN_Agent(1e-6)  # 0.0005 warm up????
    exp_name = 'init'
    dqn.continue_train(exp_name)
    test(exp_name)
    for i in range(5,15):
        print(f'=====Training Difficulties:{i} =====')
        dqn.optimizer.param_groups[0]['lr'] = 1e-5
        Epsilon = 0.4
        sucess_rate = train(dqn, env, EPOCH, exp_name)
        count = 0
        while(sucess_rate < 0.9):
            print(f'=====Training Difficulties(Additional):{i} Round-{count} =====')
            sucess_rate = train(dqn,env,1000,exp_name)
            count+=1
            if(count >=5):
                break
        difficulty_steps = i
        test(i,exp_name)
    

if __name__ == "__main__":
    main()
