# Launching clients.

me=`whoami`
export host="`squeue -u $me | grep -iv pd | grep -i pg | grep ser | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`"
echo "host=<${host}>"
for k in `seq 5`
do
sbatch --array=0-279%20 --comment=notenough --partition=learnfair --time=72:00:00 --mem=150Go --job-name=polytrain --gres=gpu:8 --cpus-per-task=80 --wrap="python -u -m pypolygames train   --max_time=259200 --saving_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_period 256 --num_rollouts 600   --replay_capacity 20000 --replay_warmup 2000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir \"exps/yaclient_\$SLURM_JOB_ID\" --out_feature --game_name minishogi --model_name ResConvConvLogitPoolModel  --turn_features --bn --nnks 3 --nnsize 8 --history 2 --nb_layers_per_net 6 --nb_nets 31   --bn --server_connect_hostname tcp://$host:10023 --num_game 20  " &
done
sbatch --array=0-279%20 --comment=notenough --partition=uninterrupted --time=72:00:00 --mem=150Go --job-name=polytrain --gres=gpu:8 --cpus-per-task=80 --wrap="python -u -m pypolygames train   --max_time=259200 --saving_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_period 256 --num_rollouts 600   --replay_capacity 20000 --replay_warmup 2000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir \"exps/yaclient_\$SLURM_JOB_ID\" --out_feature --game_name minishogi --model_name ResConvConvLogitPoolModel  --turn_features --bn --nnks 3 --nnsize 8 --history 2 --nb_layers_per_net 6 --nb_nets 31   --bn --server_connect_hostname tcp://$host:10023 --num_game 20  " &

