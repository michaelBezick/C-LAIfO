import matplotlib.pyplot as plt
import numpy as np
import os

folder = "./depth,less_mismatch,seed2/"

eval_path = folder + "eval.csv"
train_path = folder + "train.csv"

eval_reward = []
eval_step = []

with open(eval_path, "r") as file:
    for i, line in enumerate(file.readlines()):
        if i == 0:
            continue
        reward = line.split(",")[2]
        step = line.split(",")[3]

        eval_reward.append(float(reward))
        eval_step.append(float(step))


train_reward = []
train_step = []

with open(train_path, "r") as file:
    for i, line in enumerate(file.readlines()):
        if i == 0:
            continue
        reward = line.split(",")[3]
        step = line.split(",")[5]

        train_reward.append(float(reward))
        train_step.append(float(step))

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
window_size = 10
train_reward = moving_average(train_reward, window_size)
train_step = train_step[:len(train_reward)]  # Adjust x-axis accordingly

eval_reward = moving_average(eval_reward, window_size)
eval_step = eval_step[:len(eval_reward)]

plt.figure()
plt.title("Depth Image Small Mismatch - Seed 2")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.plot(train_step, train_reward, label="Train")
plt.plot(eval_step, eval_reward, label="Eval")
plt.legend()
plt.savefig("./plots/depth_image_small_mismatch_seed2.pdf")
plt.savefig("./plots/depth_image_small_mismatch_seed2.jpg",dpi=600)
