import matplotlib.pyplot as plt

filename = "./experiments/exp_walker_walk_walker_walk_difficulty=easy_delta_source=0.2_delta_target=-0.25/lail_cl_multiset_apply_aug=CL-Q_aug_type=brightness_CL_data_type=agent_2025.02.21/0854_CL_data_type=agent,agent=lail_cl_multiset,apply_aug=CL-Q,aug_type=brightness,batch_size=64,delta_source=0.2,delta_target=-0.25,depth_flag=True,device=cuda:1,difficulty=easy,task_agent=walker_walk,task_expert=walker_walk/eval.csv"

rewards = []
steps = []

with open(filename, "r") as file:

    for i, line in enumerate(file.readlines()):

        line = line.split(",")

        if i == 0:
            continue
        reward = line[2]
        step = line[4]

        rewards.append(float(reward))
        steps.append(float(step))

plt.figure()
plt.plot(steps, rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Walker Walk from 3D Point Cloud")
plt.savefig("plot.jpg")
