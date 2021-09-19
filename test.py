import numpy as np
import torch
import datetime
import time
import torch.optim as optim
from model import DRL4TSP
from tasks import tbpp
from trainer import StateCritic



def BestFit(s, e, c, Cap=100):
    n = len(s)
    f = max(e)
    sol = []
    usage = [[0 for i in range(f)] for j in range(n)]

    for i in range(n):
        b = -1
        for j in range(i + 1):
            if usage[j][s[i]] + c[i] <= Cap:
                if b == -1 or usage[j][s[i]] + c[i] > usage[b][s[i]] + c[i]:
                    b = j

        sol.append(b)
        for k in range(s[i], e[i]):
            usage[b][k] += c[i]

    return sol, len(set(sol))


def valid():
    # actor = torch.load('tbpp_item/50/2021-09-18-19/checkpoints/2/actor.pt')
    points = np.load('valid50.npy').transpose((0,2,1))
    dynamic = torch.zeros(points.shape[0],1,points.shape[2])
    points = torch.tensor(points)
    actor.eval()
    with torch.no_grad():
        # dynamic = torch.zeros(static.size()[0], 1, static.size()[2])
        tour_indices, tour_logp = actor(points, dynamic, None)
        reward = tbpp.reward(points, tour_indices)
    print(reward.mean().item())
    reward_bf = []
    for i in range(10):
        reward_bf.append(BestFit(points[i, 0, :], points[i, 1, :], points[i, 2, :])[1])
        # fea = fea and check(points_test[i], tour_indices[i])
    print(np.mean(reward_bf))



def judge(d=None):
    from tasks import tbpp
    from tasks.tbpp import TBPPDataset
    points_test = np.load('test50.npy').transpose((0, 2, 1))
    static = torch.tensor(points_test[:, :2, :])
    dataset = TBPPDataset(size=50, training=False)
    static = dataset.dataset
    dynamic = dataset.dynamic
    # train_sample(actor, critic, t, torch.zeros(1,1,test_size), None)
    actor.eval()
    critic.eval()
    with torch.no_grad():
        # dynamic = torch.zeros(static.size()[0], 1, static.size()[2])
        tour_indices, tour_logp = actor(static, dynamic, None)
        reward = tbpp.reward(static, tour_indices)
    print(reward.mean().item())
    fea = True
    reward_bf = []
    for i in range(90):
        reward_bf.append(BestFit(points_test[i, 0, :], points_test[i, 1, :], points_test[i, 2, :])[1])
        fea = fea and check(points_test[i], tour_indices[i])
    print(np.mean(reward_bf))
    print(fea)
    # print('est',est)
    # feasible = check(s,e,c,tour_indices[0])
    # print(tour_indices)
    # print(BestFit(s,e,c))
    # print(tour_logp)
    # print(feasible)
    # print(len(set(BestFit(s,e,c))))
    # print(tbpp.reward(t, tour_indices)[0].item())
    # print(best_r)


def judge_item():
    actor = torch.load('tbpp_item/50/2021-09-18-19/checkpoints/2/actor.pt')
    from tasks import tbpp_item
    from tasks.tbpp_item import TiDataset
    points_test = np.load('test50.npy').transpose((0, 2, 1))
    static = torch.tensor(points_test[:, :2, :])
    dataset = TiDataset(size=50, training=False)
    static = dataset.dataset
    dynamic = dataset.dynamic
    # train_sample(actor, critic, t, torch.zeros(1,1,test_size), None)
    actor.eval()
    # critic.eval()
    with torch.no_grad():
        # dynamic = torch.zeros(static.size()[0], 1, static.size()[2])
        tour_indices, tour_logp = actor(static, dynamic, None)
        reward = tbpp_item.reward(static, tour_indices)
    print(reward.mean().item())

    fea = True
    reward_bf = []
    for i in range(static.size()[0]):
        reward_bf.append(BestFit(points_test[i, 0, :], points_test[i, 1, :], points_test[i, 2, :])[1])
        fea = fea and check_item(points_test[i], tour_indices[i])
    print(np.mean(reward_bf))
    print(fea)
    
    points = np.load('valid50.npy').transpose((0,2,1))
    points = torch.tensor(points)
    points = torch.cat((torch.zeros(10,3,1),points),2)
    dynamic = torch.ones(points.shape[0],1,points.shape[2])
    dynamic[:,:,0] = 0
    
    actor.eval()
    with torch.no_grad():
        # dynamic = torch.zeros(static.size()[0], 1, static.size()[2])
        tour_indices, tour_logp = actor(points, dynamic, None)
        reward = tbpp_item.reward(points, tour_indices)
    print(reward.mean().item())
    reward_bf = []
    for i in range(10):
        reward_bf.append(BestFit(points[i, 0, :], points[i, 1, :], points[i, 2, :])[1])
        # fea = fea and check(points_test[i], tour_indices[i])
    print(np.mean(reward_bf))


def check(points, sol, Cap=100):
    s= points[0,:]
    e= points[1,:]
    c=points[2,:]
    for i in range(len(s)):
        b = sol[i]
        x = 0
        for j in range(len(sol)):
            if sol[j] == b:
                if s[j] <= s[i] < e[j]:
                    x += c[j]
        if x > Cap:
            print(i)
            return False
    return True


def check_item(points, tour_index):
    s = points[0, :]
    e = points[1, :]
    c = points[2, :]
    l = points.shape[1]
    u = [0]*l
    cap = [0]*(max(e)+1)
    for i in range(len(tour_index)):
        if tour_index[i] == 0:
            cap = [0] * (max(e) + 1)
            continue
        t = tour_index[i]-1
        u[t]+=1
        for j in range(s[t],e[t]):
            cap[j]+=c[t]
            if cap[j]>100:
                return False
    for i in range(l):
        if u[i]!=1:
            return False
    return True




def train_sample(actor, critic, static, dynamic, x0):
    actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-5)
    for epoch in range(1):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx in range(20):
            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = tbpp.reward(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 2)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 2)
            critic_optim.step()
            print('reward', reward[0].item())
            print('critic_est', critic_est[0].item())


judge_item()

