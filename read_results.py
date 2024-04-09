import argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--n_mem", type=int, default=295, help="The scale of noise offset.")
parser.add_argument("--n_step", type=int, default=51, help="The scale of noise offset.")
parser.add_argument("--hidden_dim", type=int, default=4096, help="The scale of noise offset.")
parser.add_argument("--n_head", type=int, default=8, help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--mode", type=str, default='entropy', help="The scale of noise offset.")
parser.add_argument("--merge_input", type=str, default=None, help="The scale of noise offset.")
parser.add_argument("--merge_output", type=str, default=None, help="The scale of noise offset.")
args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import numpy as np
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
import glob

def merge_save(n_count, path, save_path):

    entropy_results = []
    length_results = []

    mem_bar = tqdm(range(n_count), total=n_mem)

    for i in mem_bar:
        file_path = f"{path}/{i}_pos.npz"

        # Load the data
        data = np.load(file_path)
        all_arrays = []
        for array_name in data.files:
            all_arrays.append(data[array_name])
        data.close()

        length_results.append(all_arrays.pop())

        step_attn_list = [[] for _ in range(args.n_step)]
        step_entropy_list = [[] for _ in range(args.n_step)]

        for attn_id, array in enumerate(all_arrays):
            step_id = attn_id // 16
            array_tensor = torch.tensor(array, device='cuda')
            repeated_head = array_tensor.repeat(args.n_head // array_tensor.shape[0], args.hidden_dim // array_tensor.shape[1], 1)
            step_attn_list[step_id].append(repeated_head.detach().cpu().numpy())

        for step_i in range(len(step_attn_list)):
            step_attn_all_repeated = np.stack(step_attn_list[step_i], axis=0)
            step_entropy_list[step_i].append(step_attn_all_repeated.mean((1,2)))

        entropy_results.append(np.array(step_entropy_list))

    result = np.concatenate(entropy_results, axis=1)
    length_results = np.array(length_results)
    np.save(f'{save_path}_length.npy', length_results)
    np.save(f'{save_path}.npy', result)

def merge_save_small(n_count, path, save_path):

    entropy_results = []
    length_results = []

    mem_bar = tqdm(range(n_count), total=n_mem)

    for i in mem_bar:
        file_path = f"{path}/{i}_pos.npz"

        # Load the data
        data = np.load(file_path)
        all_arrays = []
        for array_name in data.files:
            all_arrays.append(data[array_name])
        data.close()

        length_results.append(all_arrays.pop())

        # import pdb ; pdb.set_trace()

        step_attn_list = [[] for _ in range(args.n_step)]
        step_entropy_list = [[] for _ in range(args.n_step)]

        for attn_id, array in enumerate(all_arrays):
            step_id = attn_id // 16
            step_attn_list[step_id].append(array)

        for step_i in range(len(step_attn_list)):
            step_attn_all_repeated = np.stack(step_attn_list[step_i], axis=0)
            step_entropy_list[step_i].append(step_attn_all_repeated)

        entropy_results.append(np.array(step_entropy_list))

    result = np.concatenate(entropy_results, axis=1)
    length_results = np.array(length_results)
    np.save(f'{save_path}.npy', result)
    np.save(f'{save_path}_length.npy', length_results)
    print(result.shape)

n_mem = args.n_mem

if args.mode == 'merge':
    entropy_every_step_mem = merge_save(n_mem, path=args.merge_input, save_path=args.merge_output)

elif args.mode == 'merge_small':
    entropy_every_step_mem = merge_save_small(n_mem, path=args.merge_input, save_path=args.merge_output)
