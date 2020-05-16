#!/bin/bash

# Extracts the machine name for the server:
export host="`squeue -u oteytaud | grep -iv pd | grep -i pg | grep ser | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`"

# Checked: this is ok.
echo "host=<${host}>"


# Launching three arrays.
for k in `seq 3`
do
sbatch --array=0-279%20 --comment=notenough --partition=learnfair --time=72:00:00 --mem=150Go --job-name=polytrain --gres=gpu:8 --cpus-per-task=80 --
wrap="python -u -m pypolygames train   --max_time=259200 --saving_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1
cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_period 256 --num_rollouts 999   --replay_capacity 20000 --r
eplay_warmup 2000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir \"exps/yaclient_\$SLURM_JOB_ID\" --out_feature --game_name havannah10pie
--model_name ResConvConvLogitPoolModel  --turn_features --bn --nnks 3 --nnsize 5 --history 2 --nb_layers_per_net 6 --nb_nets 19   --bn --server_conne
ct_hostname tcp://$host:10023 --num_game 20  " &
done
