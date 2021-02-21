import sys
import time
from replay_buffer import ReplayBuffer
from agent_iql import Agent
from helper import time_format


def train(env, config):
    """

    """
    t0 = time.time()
    save_models_path =  str(config["locexp"])
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["buffer_size"], config["seed"], config["device"]) 
    memory.load_memory(config["buffer_path"])
    # memory.idx = 50
    agent = Agent(state_size=32, action_size=4,  config=config) 
    #if config["idx"] < memory.idx:
    #    memory.idx = config["idx"] 
    print("memory idx ",memory.idx)  
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory)
        if t % int(config["eval"]) == 0:
            print(text)
            agent.save(save_models_path + "/models/{}-".format(t))
            agent.test_predicter(memory)
            agent.test_q_value(memory)
            # agent.eval_policy()
            # agent.eval_policy(True, 1)


def eval_policy(env, config):
    """

    """
    t0 = time.time()
    save_models_path =  str(config["locexp"])
    agent = Agent(state_size=32, action_size=4,  config=config) 
    for t in range(1000,30001, 1000):
        print(t)
        #t = 10000
        path = "20.02_50k/models/{}-".format(t)
        # path = "hypersearch/models/{}-".format(t)
        agent.load(path)
        agent.eval_policy(eval_episodes=10, eval_policy=True, steps=t)


def create_vid(env, config):
    """

    """
    t0 = time.time()
    save_models_path =  str(config["locexp"])
    agent = Agent(state_size=32, action_size=4,  config=config) 
    path = "20.02_50k/models/{}-".format(25000)
    agent.load(path)
    agent.eval_policy(record=True, eval_episodes=100, steps=0)
