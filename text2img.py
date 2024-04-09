import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="an integer to be processed")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--save_prefix", type=str, default="heatmap/test", help="The scale of noise offset.")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--output_name", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--save_prefix_numpy", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--miti_mem", action="store_true", default=False, help="display the square of the number")
parser.add_argument("--save_numpy", action="store_true", default=False, help="display the square of the number")
parser.add_argument("--mask_length_minis1", action="store_true", default=False, help="display the square of the number")
parser.add_argument("--cross_attn_mask", action="store_true", default=False, help="display the square of the number")
parser.add_argument('--c1', type=float, default=1, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=0, help='an integer for the accumulator')

# Parse the arguments
args = parser.parse_args()

# import pdb ; pdb.set_trace()

import os

if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from refactored_classes.MemAttn import MemStableDiffusionPipeline as StableDiffusionPipeline
from refactored_classes.refactored_unet_2d_condition import UNet2DConditionModel

import numpy as np
import random

torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_id = args.model_name
device = "cuda"

unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir="/localscratch/renjie/cache/", torch_dtype=torch.float16
    )

if args.model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, unet=unet, cache_dir="/egr/research-dselab/renjie3/.cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, unet=unet, cache_dir="/egr/research-dselab/renjie3/.cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}_seed{args.seed}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
from time import time

time_counter = 0

args.save_prefix_numpy = save_dir

counter = 0
with open(f"{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):
        prompt = line.strip()
        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')

        print(prompt)
    
        args.prompt_id = counter
        save_prefix = f"{save_dir}/{args.prompt_id}_{save_name}"

        set_seed(line_id + args.seed)
        num_images_per_prompt = 1
        start_time = time()
        images = pipe(prompt, num_images_per_prompt=num_images_per_prompt, save_prefix=save_prefix, args=args).images
        image = images[0]
        end_time = time()
        print(end_time - start_time)
        try:
            image.save(f"{save_prefix}.png")
            print("image saved at: ", f"{save_prefix}.png")
        except:
            print(f"save at {save_prefix} failed")
            continue
        
        time_counter += end_time - start_time
        counter += 1

print(time_counter / counter)
