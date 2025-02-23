# Generating images and save attention entropy
# MY_CMD="python text2img.py --prompt prompt/group_nomem500 --output_name debug --save_numpy"

# Merging results
# MY_CMD="python read_results.py --mode merge_small --n_mem 500 --n_step 50 --merge_input ./results/local_group_nomem500_detect_small_seed0 --merge_output ./results/local_group_nomem500_detect_small_seed0"

# Detection results
# MY_CMD="python detect.py --mode E --mem_input ./results/local_sd1_mem_not_n_detect_small_seed0 --nomem_input ./results/local_group_nomem500_detect_small_seed0"

# Plot results
# MY_CMD="python plot.py --mem_input ./results/local_sd1_mem_not_n_detect_small_seed0 --nomem_input ./results/local_group_nomem500_detect_small_seed0"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='4' $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
