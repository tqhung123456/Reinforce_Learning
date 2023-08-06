import random

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm


def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    return np.argmax(Qtable[state])


def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        # Exploitation
        action = greedy_policy(Qtable, state)
    else:
        # Exploration
        action = env.action_space.sample()
    return action


def train(Qtable, train_params, env_params):
    n_training_episodes = train_params["n_training_episodes"]
    learning_rate = train_params["learning_rate"]
    min_epsilon = train_params["min_epsilon"]
    max_epsilon = train_params["max_epsilon"]
    decay_rate = train_params["decay_rate"]
    max_steps = env_params["max_steps"]
    gamma = env_params["gamma"]

    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        print(epsilon)
        state, _ = env.reset()
        terminated = False
        truncated = False

        for _ in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            Qtable[state][action] += learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            if terminated or truncated:
                break
            state = new_state

    return Qtable


def evaluate_agent(env, Q, eval_params):
    n_eval_episodes = eval_params["n_eval_episodes"]
    max_steps = env_params["max_steps"]
    eval_seed = eval_params["eval_seed"]

    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if eval_seed:
            state, _ = env.reset(seed=eval_seed[episode])
        else:
            state, _ = env.reset()
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for _ in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, _, terminated, truncated, _ = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )


# Training parameters
train_params = {
    "n_training_episodes": 100000,
    "learning_rate": 0.5,
    "min_epsilon": 0,
    "max_epsilon": 1.0,
    "decay_rate": 0.00001,
}

# Evaluation parameters
eval_params = {
    "n_eval_episodes": 100,
    "eval_seed": [],
}

# Environment parameters
env_params = {"env_id": "FrozenLake-v1", "max_steps": 999, "gamma": 0.95}

env = gym.make(
    env_params["env_id"], map_name="8x8", is_slippery=False, render_mode="rgb_array"
)

training = False
if training:
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Qtable_frozenlake = initialize_q_table(state_space, action_space)

    Qtable_frozenlake = train(Qtable_frozenlake, train_params, env_params)
    print(Qtable_frozenlake)
    np.save("qtable.npy", Qtable_frozenlake)

    mean_reward, std_reward = evaluate_agent(env, Qtable_frozenlake, eval_params)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
else:
    Qtable_frozenlake = np.load("qtable.npy")
    video_path = "replay.mp4"
    video_fps = 1
    record_video(env, Qtable_frozenlake, video_path, video_fps)
