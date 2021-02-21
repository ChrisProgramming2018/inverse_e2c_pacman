import os
import gym
import random
import torch
import numpy as np
from collections import deque
import gym.wrappers
from models.vae import VAE
from models.mdrnn import MDRNNCell
from os.path import join, exists



def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def time_format(sec):
    """
    
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')


def eval_policy(env, agent, writer, steps, config, episodes=2): 
    print("Eval policy at {} steps ".format(steps))
    score = 0 
    average_score = 0
    average_steps = 0
    agent.eval()
    for i in range(episodes):
        # env = gym.wrappers.Monitor(env,str(config["locexp"])+"/vid/{}/{}".format(steps, i), video_callable=lambda episode_id: True,force=True)
        env.seed(i)
        state = env.reset("mediumClassic")
        episode_reward = 0
        for t in range(125):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            score += reward
            episode_reward += reward
            if done or t == 124:
                print(episode_reward)
                break
        average_score += score
        average_steps += t
    average_score = average_score / episodes
    average_steps = average_steps / episodes
    print("Evaluate policy on {} Episodes".format(episodes))
    agent.train()
    writer.add_scalar('Eval_ave_score', average_score, steps)
    writer.add_scalar('Eval_ave_steps ', average_steps, steps)


def create_memory(env, agent, memory, steps, config, episodes=100): 
    print("Create buffer with size {} steps ".format(steps))
    score = 0 
    average_score = 0
    average_steps = 0
    agent.eval()
    
    LSIZE = 200
    ASIZE = 1
    RSIZE = 256
    mdir = "15.02_l200_RGB"
    m = ""
    vae_file, rnn_file = [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn']]
    vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(config["device"])})
            for fname in (vae_file, rnn_file)]
    for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
        print("Loading {} at epoch {} "
                "with test loss {}".format(
                    m, s['epoch'], s['precision']))

    vae = VAE(3, LSIZE).to(config["device"])
    vae.load_state_dict(vae_state['state_dict'])
    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(config["device"])
    mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
    for i in range(episodes):
        # env = gym.wrappers.Monitor(env,str(config["locexp"])+"/vid/{}/{}".format(steps, i), video_callable=lambda episode_id: True,force=True)
        env.seed(i)
        state, obs = env.reset("mediumClassic")
        print(obs)
        episode_reward = 0
        index = memory.idx
        hidden = [torch.zeros(1, RSIZE).to(config["device"]) for _ in range(2)]
        for t in range(125):
            state_tensor = state.clone().detach().type(torch.cuda.FloatTensor).div_(255)
            action = agent.act(state_tensor)
            action_rnn = torch.as_tensor(action, device=config["device"]).type(torch.int).unsqueeze(0).unsqueeze(0)
            # print(action_rnn)
            # print(action_rnn.shape)
            states = obs
            states = torch.as_tensor(states, device=config["device"]).unsqueeze(0)
            states = states.type(torch.float32).div_(255)
            _, latent_mu, _ = vae(states)
            # print(latent_mu.shape)
            _, _, _, _, _, next_hidden = mdrnn(action_rnn, latent_mu, hidden)
            # print(next_hidden[0].shape)
            # print(next_hidden[1].shape)
            # sys.exit()
            next_state, reward, done, next_obs = env.step(action)
            if t != 124:
                done_no_max = done
            else:
                done_no_max = False
            memory.add(obs, hidden[0], hidden[1], action, next_obs, next_hidden[0], next_hidden[1], done, done_no_max)
            if memory.idx % 500 == 0:
                path = "pacman_expert_memory-{}".format(memory.idx)
                print("save memory to ",path)
                memory.save_memory(path)
                if memory.idx >= steps:
                    return
            hidden = next_hidden
            state = next_state
            obs = next_obs
            score += reward
            episode_reward += reward
            if done or t == 124:
                if episode_reward < 600:
                    memory.idx = index

                print("Episode_reward {} and memory idx {}".format(episode_reward, memory.idx))
                break
