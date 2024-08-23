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
    lr = 0.0001
    gamma = 0.99

    # 传统指数衰减epsilon = max_epsilon * exp(-gama * count)
    max_epsilon = 0.95
    min_epsilon = 0.01
    exp_decay_rate = 0.002

    # 训练次数以及每次最大步长
    epochs = 2000
    episode_len = 100000
    # 经验回放池大小
    replay_buffer_size = 100000

    batch_size = 64

    # 更新target的频率以及保存频率
    target_update_freq = 4
    save_freq = 10


class MLP(nn.Module):
    def __init__(self, params: Params = None):
        """
        向量型环境状态表示, 使用全连接网络作为Q网络, 最后一层不用ReLU
        :param params: 参数
        """
        super(MLP, self).__init__()
        self.input_linear = nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, params.action_space_size)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = self.output_linear(x)
        return x


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


class DQN:
    def __init__(self, params: Params = None, model: nn.Module = None, replay_buffer: ReplayBuffer = None):
        self.params = params

        self.epsilon = self.params.max_epsilon
        self.exp_decay_count = 0

        self.policy = model.to(self.params.device)
        self.target = model.to(self.params.device)

        self.batch_size = params.batch_size
        self.replay_buffer = replay_buffer

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.params.lr)

    def update(self):
        """
        step 1. 数据转换
        step 2. 优化函数
        step 3. 损失计算
        step 4. 反向传播, 模型更新优化
        :return: None
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(np.array(states))
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(np.array(next_states))
        dones = torch.tensor(dones)

        # DQN的目标函数的计算公式
        # 这里就是DQN的贡献之二:使用两个网络,只更新policy,不更新target
        state_action_values = self.policy(states)
        next_state_action_values = self.target(next_states).detach()
        # 计算预测值以及目标值
        q_values = state_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        max_action_values, indexes = torch.max(next_state_action_values, dim=1)
        q_targets = rewards + self.params.gamma * (max_action_values * (1 - dones))
        # 计算均方误差
        loss = nn.MSELoss()(q_values, q_targets)
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 限制梯度
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def sample_action(self, state: np.ndarray) -> int:
        """
        根据当前状态选择一个动作, epsilon greedy策略
        :param state: 当前状态
        :return: 执行的动作
        """
        self.exp_decay()
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_values = self.policy(torch.tensor(state))
                action = torch.argmax(action_values).item()
        else:
            action = random.randrange(self.params.action_space_size)
        return action

    def exp_decay(self) -> None:
        """
        指数衰减公式: epsilon = max_epsilon * exp(-gama * count)
        限制一下最小的探索率
        :return: None
        """
        self.exp_decay_count += 1
        self.epsilon = self.params.max_epsilon * math.exp(-self.params.exp_decay_rate * self.exp_decay_count)
        self.epsilon = max(self.epsilon, self.params.min_epsilon)


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
    immediately_rewards = np.zeros(shape=params.epochs, dtype=float)

    for epoch in range(params.epochs):
        immediately_reward = 0
        state, info = env.reset()

        for step in range(params.episode_len):
            action = agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.replay_buffer.push([state, action, reward, next_state, int(done)])

            state = next_state
            agent.update()
            immediately_reward += reward
            if done:
                immediately_rewards[epoch] = immediately_reward
                break

        if not ((epoch + 1) % params.target_update_freq):
            agent.target.load_state_dict(agent.policy.state_dict())

        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 奖励:{immediately_reward:.2f}, epsilon:{agent.epsilon:.3f}")
            np.save(f"./data/dqn_immediately_rewards{epoch + 1}.npy", immediately_rewards[:epoch])
            torch.save(agent.policy.state_dict(), f"./data/dqn_policy_epoch{epoch + 1}.pt")
    env.close()


def dqn():
    params = Params()

    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]

    model = MLP(params)
    replay_buffer = ReplayBuffer(params.replay_buffer_size)
    agent = DQN(params, model, replay_buffer)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/dqn_immediately_rewards200.npy")[:100]
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dqn()
    # plot_rewards()
