import random
import numpy as np
import matplotlib.pyplot as plt

params = np.zeros(2, dtype=float)
evs = np.zeros(2, dtype=float)
params_hist = []
reward_hist = []
evs_hist = []
lr = 0.01
reward_avg = 0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for i in range(10000):
    r1 = random.choices([0, 1], weights=[0.25, 0.75])
    r2 = random.choices([0, 1], weights=[0.75, 0.25])

    action = random.choices([1, 2], weights=softmax(params))[0]

    reward = (r1 if action == 1 else r2)[0]

    reward_hist.append(reward)

    grad = np.zeros(2, dtype=float)

    grad[action - 1] = 1

    evs[action - 1] += lr * (reward - evs[action - 1])

    evs_hist.append(evs.copy())

    reward_avg += lr * (reward - reward_avg)

    params += lr * (grad - softmax(params)) * evs[action - 1]

    params_hist.append(params.copy())

params_hist = np.array(params_hist)
reward_hist = np.array(reward_hist)
evs_hist = np.array(evs_hist)

plt.plot(params_hist[:, 0], label='R1')
plt.plot(params_hist[:, 1], label='R2')

plt.title('Preference Values Over Time')
plt.xlabel('Iterations')
plt.ylabel('Preference Value')
plt.show()

plt.title('Rewards Over Time')
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.plot(reward_hist)
plt.show()

plt.plot(evs_hist[:, 0], label='Estimated Value R1')
plt.plot(evs_hist[:, 1], label='Estimated Value R2')
plt.title('Estimated Values Over Time')
plt.xlabel('Iterations')
plt.ylabel('Estimated Value')
plt.show()