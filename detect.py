import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--mode", type=str, default='entropy', help="The scale of noise offset.")
parser.add_argument("--mem_input", type=str, default=None, help="The scale of noise offset.")
parser.add_argument("--nomem_input", type=str, default=None, help="The scale of noise offset.")
args = parser.parse_args()

if args.mode == "D":
    use_entropy = True
    use_delta_padding = True
    use_special_layer = False
    step_id = np.arange(40, 50)
elif args.mode == "E":
    use_entropy = True
    use_delta_padding = False
    use_special_layer = True
    step_id = np.array([0])
    layer_id = np.array([3])
else:
    raise("Incorrect mode.")

mem_name = f'{args.mem_input}.npy'
nomem_name = f'{args.nomem_input}.npy'
mem_length_name = f'{args.mem_input}_length.npy'
nomem_length_name = f'{args.nomem_input}_length.npy'

if use_special_layer:
    attn_mem = np.load(mem_name)[:, :, layer_id].mean((2, 3))
    attn_nomem = np.load(nomem_name)[:, :, layer_id].mean((2, 3))
else:
    attn_mem = np.load(mem_name).mean((2, 3))
    attn_nomem = np.load(nomem_name).mean((2, 3))

# import pdb ; pdb.set_trace()

entropy_every_step_mem = (- attn_mem * np.log(attn_mem)).sum(2)
entropy_every_step_nomem = (- attn_nomem * np.log(attn_nomem)).sum(2)

score_mem = 0
score_nomem = 0

if use_entropy:
    score_mem += entropy_every_step_mem[step_id].mean(0)
    score_nomem += entropy_every_step_nomem[step_id].mean(0)

if use_delta_padding:

    length_mem = np.load(f'./{mem_length_name}')
    length_nomem = np.load(f'./{nomem_length_name}')

    padding_entropy_mem_list = []
    padding_entropy_nomem_list = []

    prompt_entropy_mem_list = []
    prompt_entropy_nomem_list = []

    for i in range(attn_mem.shape[1]):
        padding_entropy = attn_mem[:, i, length_mem[i]-1:]
        padding_entropy = (- padding_entropy * np.log(padding_entropy)).sum(1)
        padding_entropy_mem_list.append(padding_entropy)

        prompt_entropy = attn_mem[:, i, :length_mem[i]-1]
        prompt_entropy = (- prompt_entropy * np.log(prompt_entropy)).sum(1)
        prompt_entropy_mem_list.append(prompt_entropy)
        
    padding_entropy_mem_list = np.stack(padding_entropy_mem_list, axis=1)
    prompt_entropy_mem_list = np.stack(prompt_entropy_mem_list, axis=1)

    for i in range(attn_nomem.shape[1]):
        padding_entropy = attn_nomem[:, i, length_nomem[i]-1:]
        padding_entropy = (- padding_entropy * np.log(padding_entropy)).sum(1)
        padding_entropy_nomem_list.append(padding_entropy)

        prompt_entropy = attn_nomem[:, i, :length_nomem[i]-1]
        prompt_entropy = (- prompt_entropy * np.log(prompt_entropy)).sum(1)
        prompt_entropy_nomem_list.append(prompt_entropy)

    padding_entropy_nomem_list = np.stack(padding_entropy_nomem_list, axis=1)
    prompt_entropy_nomem_list = np.stack(prompt_entropy_nomem_list, axis=1)


    score_mem += (padding_entropy_mem_list[step_id].mean(0) - padding_entropy_mem_list[0])
    score_nomem += (padding_entropy_nomem_list[step_id].mean(0) - padding_entropy_nomem_list[0])

scores = np.concatenate([score_mem, score_nomem], axis=0)
labels = np.array([1] * len(entropy_every_step_mem[40]) + [0] * len(entropy_every_step_nomem[40]))

# import pdb ; pdb.set_trace()
auroc = roc_auc_score(labels, scores)
floats_list = [auroc]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)

# Find the closest FPR to 0.01 (1% FPR)
thre_fpr = [0.01, 0.03, 0.05, 0.1]
for i in range(len(thre_fpr)):
    target_fpr = thre_fpr[i]
    closest_fpr_index = np.argmin(np.abs(fpr - target_fpr))
    closest_fpr = fpr[closest_fpr_index]
    tpr_at_target_fpr = tpr[closest_fpr_index]

    # print(f"True Positive Rate (TPR) at 1% False Positive Rate (FPR): {tpr_at_target_fpr}")
    floats_list.append(tpr_at_target_fpr)

# Specify the output file name
output_file_name = 'results.txt'

heads = ['AUROC', "TPR@0.01FPR", "TPR@0.03FPR", "TPR@0.05FPR", "TPR@0.1FPR"]

# Open the output file in write mode
with open(output_file_name, 'w') as file:
    # Join the list of floats converted to strings with '\t' as the separator
    # and write to the file
    file.write('\t'.join(map(str, heads)) + '\n')
    file.write('\t'.join(map(str, floats_list)))

print(f"List of floats has been saved to '{output_file_name}'.")
