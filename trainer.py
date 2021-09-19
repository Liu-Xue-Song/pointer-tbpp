"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import DRL4TSP, Encoder
from tasks.tbpp import TBPPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        # self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        # self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        # self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        self.fc1 = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        # self.fc3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.fc4 = nn.Conv1d(hidden_size, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        # output = F.relu(self.fc3(output))
        output = self.fc4(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output1 = F.relu(self.fc2(output)).squeeze(2)
        output2 = self.fc3(output1).sum(dim=2)
        return output2


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H")
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    use_baseline = True
    for epoch in range(3):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):
            start = time.time()
            static, dynamic, x0, baseline= batch
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)
            # from tasks.tbpp_item import check
            # print(check(static, dynamic, tour_indices))
            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            if use_baseline:
                critic_est = baseline
            else:
                critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))


            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()
            if not use_baseline:
                critic_loss = torch.mean(advantage ** 2)
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                critic_rewards.append(torch.mean(critic_est.detach()).item())
            
            rewards.append(torch.mean(reward.detach()).item())
            
            losses.append(torch.mean(actor_loss.detach()).item())

            # if (batch_idx + 1) % 100 == 0:
            #     end = time.time()
            #     times.append(end - start)
            #     start = end
            #
            #     mean_loss = np.mean(losses[-100:])
            #     mean_reward = np.mean(rewards[-100:])
            #
            #     print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
            #           (batch_idx, len(train_loader), mean_reward, mean_loss,
            #            times[-1]))
            from tasks.tbpp_item import check
            print(epoch, 'batch_idx', batch_idx, 'time', int(time.time()-start),rewards[-1])
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        print('loss for', epoch, ':', mean_loss)
        print('reward', mean_reward, 'estimate', np.mean(critic_rewards))
        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor, save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic, save_path)

        # Save rendering of validation set tours
        # valid_dir = os.path.join(save_dir, '%s' % epoch)

        # mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
        #                       valid_dir, num_plot=5)
        #
        # # Save best model parameters
        # if mean_valid < best_reward:
        #
        #     best_reward = mean_valid
        #
        #     save_path = os.path.join(save_dir, 'actor.pt')
        #     torch.save(actor.state_dict(), save_path)
        #
        #     save_path = os.path.join(save_dir, 'critic.pt')
        #     torch.save(critic.state_dict(), save_path)
        #
        # print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
        #       '(%2.4fs / 100 batches)\n' % \
        #       (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
        #       np.mean(times)))



def train_tsp(args):

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    from tasks import tbpp
    from tasks.tbpp import TBPPDataset

    STATIC_SIZE = 3 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility

    train_data = TBPPDataset(args.num_nodes, args.train_size)
    valid_data = TBPPDataset(args.num_nodes, args.valid_size, training=False)

    update_fn = None

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tbpp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tbpp.reward
    kwargs['render_fn'] = None

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = TBPPDataset(args.num_nodes, args.train_size, args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, tbpp.reward, None, test_dir, num_plot=5)

    print('Average tour length: ', out)


def train_tbpp_dynamic(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import tbpp
    from tasks.tbpp import TBPPDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    STATIC_SIZE = 3 # (x, y)
    DYNAMIC_SIZE = 1 # (rest_cap)

    train_data = TBPPDataset(args.num_nodes, args.train_size)
    valid_data = TBPPDataset(args.num_nodes, args.valid_size, training=False)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    tbpp.update_dynamic,
                    tbpp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    # actor = torch.load('tbpp_dynamic/50/2021-09-13-14/checkpoints/0/actor.pt')
    # critic = torch.load('tbpp_dynamic/50/2021-09-13-14/checkpoints/0/critic.pt')
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tbpp.reward
    kwargs['render_fn'] = None

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = TBPPDataset(args.num_nodes, args.train_size, args.seed + 2, training=False)
    #
    test_dir = 'test'
    test_loader = DataLoader(test_data, 90, False, num_workers=0)
    # out = validate(test_loader, actor, tbpp.reward, None, test_dir, num_plot=5)
    #
    # print('Average tour length: ', out)


def train_tbpp_item(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import tbpp_item
    from tasks.tbpp_item import TiDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    STATIC_SIZE = 3 # (x, y)
    DYNAMIC_SIZE = 1 # (rest_cap)

    train_data = TiDataset(args.num_nodes, args.train_size)
    valid_data = TiDataset(args.num_nodes, args.valid_size, training=False)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    tbpp_item.update_dynamic,
                    tbpp_item.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    actor = torch.load('tbpp_item/50/2021-09-18-19/checkpoints/2/actor.pt')
    # critic = torch.load('tbpp_dynamic/50/2021-09-13-14/checkpoints/0/critic.pt')
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tbpp_item.reward
    kwargs['render_fn'] = None

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)


    #
    # print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tbpp_item')
    parser.add_argument('--nodes', dest='num_nodes', default=50, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=1e-6, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()

    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    if args.task == 'tsp':
        train_tsp(args)
    elif args.task == 'tbpp_item':
        train_tbpp_item(args)
    elif args.task == 'tbpp_dynamic':
        train_tbpp_dynamic(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)
