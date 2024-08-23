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
    hidden_dim = 256
    # 学习率和折扣率, 很关键的参数!!! 调的时候一度怀疑自己的代码是否写错, 实际上就是参数问题
    lr = 0.0005
    gamma = 0.99

    # 训练次数以及每次最大步长
    epochs = 2000
    episode_len = 100000

    batch_size = 64

    # 更新target的频率以及保存频率
    save_freq = 100


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
        x = nn.Softmax()(self.output_linear(x))
        return x


class DQN:
    def __init__(self, params: Params = None, model: nn.Module = None):
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
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        discounted_returns = []
        discounted_return = 0
        for reward in rewards[::-1]:
            discounted_return = reward + self.params.gamma * discounted_return
            discounted_returns.append(discounted_return)
        discounted_returns = torch.tensor(np.array(discounted_returns))

        self.optimizer.zero_grad()
        for t, (state, action) in enumerate(zip(states, actions)):
            action_probs = self.policy(torch.tensor(state))
            dist = torch.distributions.Categorical(action_probs)
            # 这里没那么神秘，转换为对数概率计算更稳定，也可以自己用torch.log手动转试试看值是否一样
            action_log_prob = dist.log_prob(torch.tensor(action))
            loss = -action_log_prob * discounted_returns[t]
            loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def sample_action(self, state: np.ndarray) -> int:
        """
        根据当前状态获得各个动作的概率，然后根据这个概率建立分类分布，再用这个分布进行采样获得动作
        :param state: 当前状态
        :return: 执行的动作
        """
        action_probs = self.policy(torch.tensor(state))
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item()


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


def train(params: Params = None, env: gym.Env = None, agent: DQN = None):
    for epoch in range(params.epochs):
        state, info = env.reset()
        states, actions, rewards = [], [], []
        for step in range(params.episode_len):
            action = agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                agent.update(states, actions, rewards)
                break
        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 奖励:{sum(rewards):.2f}")
            np.save(f"./data/pg_immediately_rewards{epoch + 1}.npy", np.array(rewards))
            torch.save(agent.policy.state_dict(), f"./data/pg_policy_epoch{epoch + 1}.pt")
    env.close()


def pg():
    params = Params()

    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]

    model = MLP(params)
    agent = DQN(params, model)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/pg_immediately_rewards2000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pg()
    # plot_rewards()
