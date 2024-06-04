# Generating images and save attention entropy
MY_CMD="python text2img.py --prompt prompt/group_nomem500 --float16 --output_name debug --c1 1.25 --cross_attn_mask --miti_mem"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='4' $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
