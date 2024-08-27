import random
import math
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt


class Params:
    device = "cpu"
    env_name = "CartPole-v1"
    # 动作空间和状态空间以及隐藏层大小
    action_space_size = 0
    state_space_size = 0
    hidden_dim = 128
    # 学习率和折扣率, 很关键的参数!!! 调的时候一度怀疑自己的代码是否写错, 实际上就是参数问题
    lr = 0.0001
    gamma = 0.99

    # 训练次数以及每次最大步长
    epochs = 1000
    episode_len = 200

    # 经验回放池大小
    replay_buffer_size = 10000
    batch_size = 64

    # 更新target的频率以及保存频率
    save_freq = 100


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        """
        DQN贡献之一: 经验回访池, 打破数据时间相关性
        通用经验回放池，利用队列来进行维护，先入先出
        :param capacity: 经验回放池大小
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size: int) -> Tuple:
        """
        采样
        :param batch_size: 样本数
        :return: 采样结果
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, params: Params = None):
        """
        向量型环境状态表示, 使用全连接网络作为策略网络, 最后一层激活函数用softmax转换为概率
        :param params: 参数
        """
        super(MLP, self).__init__()
        self.input_linear = nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, params.action_space_size)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = nn.Softmax(dim=-1)(self.output_linear(x))
        return x


class PG:
    def __init__(self, params: Params = None, model: nn.Module = None, replay_buffer: ReplayBuffer = None):
        self.params = params

        self.policy = model.to(self.params.device)
        self.batch_size = params.batch_size
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.params.lr)

    def update(self, states: list, actions: list, rewards: list) -> None:
        """
        step 1. Value Update 计算t时刻的q(s,a)
        step 2. Policy Update
        step 3. 反向传播, 模型更新优化
        :param states: 状态
        :param actions: 动作
        :param rewards: 奖励
        :return: None
        """
        states = torch.tensor(np.array(states)).to(self.params.device)
        actions = torch.tensor(np.array(actions)).to(self.params.device)
        discounted_return = 0
        discounted_returns = []
        for reward in rewards:
            discounted_return = discounted_return * self.params.gamma + reward
            discounted_returns.insert(0, discounted_return)
        discounted_returns = torch.tensor(discounted_returns).to(self.params.device)

        self.optimizer.zero_grad()

        for t, (state, action) in enumerate(zip(states, actions)):
            action_log_prob = torch.distributions.Categorical(self.policy(state)).log_prob(action)
            loss = -action_log_prob * discounted_returns[t]
            loss.backward()
        self.optimizer.step()

    def sample_action(self, state: np.ndarray) -> int:
        """
        根据当前状态获得各个动作的概率，然后根据这个概率建立分类分布，再用这个分布进行采样获得动作
        :param state: 当前状态
        :return: 执行的动作
        """
        state = torch.tensor(state).to(self.params.device)
        dist = torch.distributions.Categorical(self.policy(state))
        action = dist.sample()
        return action.detach().item()


def smooth(data: np.ndarray, weight=0.9) -> list:
    """
    绘制平滑曲线
    :param data: 数据
    :param weight: 平滑程度
    :return: 平滑结果
    """
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def env_seed(seed: int = 1) -> None:
    """
    设定种子
    :param seed: 种子
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train(params: Params = None, env: gym.Env = None, agent: PG = None):
    immediately_rewards = []
    for epoch in range(params.epochs):
        state, info = env.reset()
        states, actions, rewards = [], [], []
        for step in range(params.episode_len):
            action = agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        agent.update(states, actions, rewards)
        immediately_rewards.append(sum(rewards))

        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 平均奖励:{np.array(immediately_rewards).mean():.2f}")
            np.save(f"./data/pg_immediately_rewards{epoch + 1}.npy", np.array(immediately_rewards))
            torch.save(agent.policy.state_dict(), f"./data/pg_policy_epoch{epoch + 1}.pt")
    env.close()


def pg():
    params = Params()

    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]

    model = MLP(params)
    agent = PG(params, model)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/pg_immediately_rewards1000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pg()
    plot_rewards()
