
import matplotlib.pyplot as plt

eval_path = "./medium_point_net,less_mismatch,rotation_aug/1934_CL_data_type=agent,agent=lail_cl_multiset,apply_aug=CL-Q,aug_type=brightness,batch_size=256,delta_source=0.2,delta_target=-0.25,depth_flag=True,device=cuda:1,difficulty=easy,lr=1e-4,task_agent=walker_walk,task_expert=walker_walk/eval.csv"
train_path = "./medium_point_net,less_mismatch,rotation_aug/1934_CL_data_type=agent,agent=lail_cl_multiset,apply_aug=CL-Q,aug_type=brightness,batch_size=256,delta_source=0.2,delta_target=-0.25,depth_flag=True,device=cuda:1,difficulty=easy,lr=1e-4,task_agent=walker_walk,task_expert=walker_walk/train.csv"

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

plt.figure()
plt.title("Point Cloud Small Mismatch Train vs. Eval on Walker Walk")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.plot(train_step, train_reward, label="Train")
plt.plot(eval_step, eval_reward, label="Eval")
plt.legend()
plt.savefig("./plots/point_cloud_large_no_mismatch.pdf")
