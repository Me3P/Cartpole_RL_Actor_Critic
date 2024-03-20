import gymnasium as gym
import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve
from gym import wrappers
import tensorflow as tf
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1500
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'cartpole_1e-5_1024x512_1500games.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True
    training_flag = False

    if load_checkpoint:
        # agent = load_model(agent.actor_critic.checkpoint_file)
        agent.load_models()

    if training_flag:
        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                observation_ = tf.convert_to_tensor([observation_], dtype=tf.float32)
                score += reward
                if training_flag:
                    agent.learn(observation, reward, observation_, done)
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if training_flag:
                    agent.save_models()

            print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        if not load_checkpoint:
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file)

    # test
    for i in range(15):
        env = gym.make('CartPole-v1', render_mode="human")
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            observation = tf.convert_to_tensor([observation_], dtype=tf.float32)
            score += reward
            env.render()
        print (score)
        env.close()