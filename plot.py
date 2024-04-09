import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--mem_input", type=str, default=None, help="The scale of noise offset.")
parser.add_argument("--nomem_input", type=str, default=None, help="The scale of noise offset.")
args = parser.parse_args()

save_name = 'test'

mem_name = f'{args.mem_input}.npy'
nomem_name = f'{args.nomem_input}.npy'

attn_mem = np.load(mem_name).mean((2,3))
attn_nomem = np.load(nomem_name).mean((2,3))

entropy_every_step_mem = (- attn_mem * np.log(attn_mem)).sum(2)
entropy_every_step_nomem = (- attn_nomem * np.log(attn_nomem)).sum(2)
vector1 = entropy_every_step_mem.mean(1)
vector2 = entropy_every_step_nomem.mean(1)

x = np.arange(0, len(vector1))[::-1]

plt.figure(figsize=(6.7, 5))

# Plotting the vectors
plt.plot(x, vector1, label='Memorization')
plt.plot(x, vector2, label='Non-memorization')

plt.xticks([0, 49 / 2, 49], ['0', '$T/2$', '$T$'], fontsize=22)
plt.yticks(fontsize=16)

plt.ylabel('Entropy', fontsize=22)
plt.legend(loc='upper left', fontsize=20)
plt.ylim(0.4, 1.3)
plt.tight_layout()
plt.savefig(f"./plot/{save_name}.png")
print(f"saved at ./plot/{save_name}.png")

plt.close()
