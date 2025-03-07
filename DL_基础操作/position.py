import numpy as np
import matplotlib.pyplot as plt


# 生成不同位置的编码
pos = np.arange(100)  # 位置从 0 到 99
i = np.array([0, 1, 2, 3])  # 4 维的 sin/cos
d = 512  # 假设 embedding 维度为 512
freq = np.exp(-np.log(10000) * (2 * i / d))

sin_values = np.sin(pos[:, None] * freq)
cos_values = np.cos(pos[:, None] * freq)

plt.plot(pos, sin_values[:, 0], label="sin(0th dim)")
plt.plot(pos, sin_values[:, 1], label="sin(1st dim)")
plt.plot(pos, cos_values[:, 2], label="cos(2nd dim)")
plt.plot(pos, cos_values[:, 3], label="cos(3rd dim)")
plt.legend()
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.title("Sinusoidal Positional Encoding")
plt.show(block=True)
plt.savefig('/root/autodl-tmp/动手学深度学习-练习/pic.png')