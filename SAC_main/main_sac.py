import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import os
import torch as T

if __name__ == '__main__':
    # env = gym.make('InvertedPendulumBulletEnv-v0')
    # env = gym.make('Hopper-v2')
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 300
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    # env = Recorder(env, 'tmp/video')
    filename = 'SAC_300_LunarLanderContinuous-v2.png'
    figure_file = 'plots/' + filename
    
    ## اضافه کردم
    checkpoint_file_epoch = os.path.join('tmp/sac', 'epoch_sac')

    best_score = env.reward_range[0]
    epoch = 0
    score_history = []
    ## مهم
    load_checkpoint = False
    ##

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')
        epoch_score = T.load(checkpoint_file_epoch)
        epoch = epoch_score['n_epoch']
        score_history = epoch_score['score_hist']

    for i in range(epoch,n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            # if not load_checkpoint:
            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #if not load_checkpoint:
        agent.save_models()
        T.save({'n_epoch':i, 'score_hist':score_history}, checkpoint_file_epoch)


        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    # if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)