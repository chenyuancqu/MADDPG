import os.path
import openpyxl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import rl_utils
import multiagent.V2A_Envf_2 as env
import warnings
import datetime


warnings.filterwarnings("ignore")


# def make_env(scenario_name):
#     #从环境文件脚本中创建环境
#     scenario = scenarios.load(scenario_name+".py").Scenario()
#     world = scenario.make_world()
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
#     return env


"""让离散动作变得可导，使得可以使用反向传播等函数"""


def onehot_from_logits(logits, eps=0.01):
    # 输入示例的概率分布logits = torch.tensor([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
    # print('logits=',logits)
    # 输出tensor([[0., 1., 0.],[1., 0., 0.]]),也可能是其他值，因为贪婪算法有一定随机性
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y


# DDPG算法实现
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)           # 给定状态下预测动作
        self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)    # 用于执行软更新
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)            # 用于在给定状态和动作下预测Q值
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)     # 执行软更新
        self.target_critic.load_state_dict(self.critic.state_dict())              # 保证开始训练时两个目标网络于实际网络参数一致
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)    # 网络优化器，更新网络参数

    def take_action(self, state, explore=False):
        action = self.actor(state)  # 输入状态，得到输出的action
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)     # 只有最大概率对应的位置为1，其余位置都为0。这就是概率分布的独热表示。[0. 0. 1.]
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(env.agents):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(env.agents)
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()
    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


num_episodes = 2000
episode_length = 50
buffer_size = 100000
minimal_size = 4000
hidden_dim = 64
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.98
tau = 1e-2
batch_size = 1024
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(device)
update_interval = 100   # 每隔100步进行一次网络更新


env_id = "V2A_Envf_2"
env = env.V2A_Envf_2()
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n)
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0])
critic_input_dim = sum(state_dims) + sum(action_dims)
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau)


def evaluate(env_id, maddpg, n_episode=50, episode_length=25):
    returns = np.zeros(env.agents)
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info, _, _ = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()


return_list = []
total_step = 0
total_reward = []
q_loss = []
qloss = 0

average_returns = []
average_latency = []
complete_rate_n = []
loss = []

for i_episode in range(num_episodes):
    state = env.reset()
    ep_returns = np.zeros(env.agents)
    ep_latency = np.zeros(env.agents)
    for e_i in range(episode_length):
        actions = maddpg.take_action(state, explore=True)
        next_state, reward, done, _, latency, complete_rate = env.step(actions)

        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state
        ep_returns += np.array(reward)
        ep_latency += np.array(latency)
        complete_rate_n.append(complete_rate)

        if e_i >= episode_length:
            done = [True] * env.agents

        total_step += 1
        if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]

            sample = [stack_array(x) for x in sample]

            for a_i in range(env.agents):
                qloss = maddpg.update(sample, a_i)
                q_loss.append(qloss)

            maddpg.update_all_targets()

    print(f"Episode {i_episode + 1}, Total Return: {np.mean(ep_returns)}")
    average_returns.append(ep_returns)
    average_latency.append(ep_latency)
    average_returns_every_5 = [np.mean(average_returns[i:i + 5])
    for i in range(0, num_episodes, 5)]


    if (i_episode + 1) % 50 == 0:
        ep_returns = evaluate(env_id, maddpg, n_episode=50)
        return_list.append(ep_returns)
        total_reward.append(sum(ep_returns) / env.agents)
        print(f"Episode: {i_episode+1}, {ep_returns}")  # 输出当前轮次的训练和评估结果


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fold_path = './savedata'
"""保存reward"""
if os.path.exists(fold_path):
    df = pd.DataFrame(average_returns)  # 将average_returns转换为dataframe
    df.index = [f'Episode_{i + 1}' for i in range(df.shape[0])]
    df.columns = [f'vehicle_{i + 1}' for i in range(df.shape[1])]
    file_path = f'./savedata/{env.agents}_reward_{current_time}.xlsx'
    try:
        df.to_excel(file_path)
        print("data saved successfully.")
    except Exception as e:
        print(f"Error while saving data: {e}")
else:
    print("Folder does not exits")

"""保存时间延迟"""
fold_path2 = "./savelatency"
if os.path.exists(fold_path2):
    df = pd.DataFrame(average_latency)
    df.index = [f'Episode_{i + 1}' for i in range(df.shape[0])]
    df.columns = [f'vehicle_{i + 1}' for i in range(df.shape[1])]
    file_path = f"./savelatency/{env.agents}_latency_{current_time}.xlsx"
    try:
        df.to_excel(file_path)
        print("data saved successfully.")
    except Exception as e:
        print(f"Error while saving data: {e}")
else:
    print("Folder does not exits")


"""保存任务完成率"""
fold_path3 = "./saverate"
if os.path.exists(fold_path3):
    df = pd.DataFrame(complete_rate_n)
    df.index = [f'Episode_{i + 1}' for i in range(df.shape[0])]
    df.columns = [f'vehicle_{i + 1}' for i in range(df.shape[1])]
    file_path = f"./saverate/{env.agents}_rate_{current_time}.xlsx"
    try:
        df.to_excel(file_path)
        print("data saved successfully.")
    except Exception as e:
        print(f"Error while saving data: {e}")
else:
    print("Folder does not exits")

x_axis = list(range(1, num_episodes + 1, 5))
plt.plot(x_axis, average_returns_every_5)
plt.xlabel('Training Episodes')
plt.ylabel('Mean Episode Rewards')
file_path1 = f'./savefig/training_fig_{current_time}.jpg'
plt.savefig(file_path1)
plt.show()

