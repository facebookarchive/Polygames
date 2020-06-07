export MASTER_ADDR=127.0.0.1
export MASTER_PORT=10022
export MASTER_PORT=10023
export MASTER_PORT=10022
export RANK=0
export WORLD_SIZE=1
#for k in `seq 12`
#do
export host="`squeue -u oteytaud | grep -iv pd | grep -i pg | grep ser | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`"
echo "host=<${host}>"
for k in `seq 3`
do
sbatch --array=0-279%20 --comment=notenough --partition=learnfair --time=72:00:00 --mem=150Go --job-name=polytrain --gres=gpu:8 --cpus-per-task=80 --wrap="python -u -m
 pypolygames train   --max_time=259200 --saving_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 c
uda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_period 256 --num_rollouts 100   --replay_capacity 20000 --replay_warmup 2000 --do_not_save_replay_buffer --ddp tr
ue   --checkpoint_dir \"exps/yaclient_\$SLURM_JOB_ID\" --out_feature --game_name havannah10pie --model_name ResConvConvLogitPoolModel  --turn_features --bn --nnks 3 --
nnsize 5 --history 2 --nb_layers_per_net 6 --nb_nets 31   --bn --server_connect_hostname tcp://$host:10023 --num_game 20  " &
done
sbatch --array=0-279%20 --comment=notenough --partition=uninterrupted --time=72:00:00 --mem=150Go --job-name=polytrain --gres=gpu:8 --cpus-per-task=80 --wrap="python -
u -m pypolygames train   --max_time=259200 --saving_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda
:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_period 256 --num_rollouts 100   --replay_capacity 20000 --replay_warmup 2000 --do_not_save_replay_buffer --dd
p true   --checkpoint_dir \"exps/yaclient_\$SLURM_JOB_ID\" --out_feature --game_name havannah10pie --model_name ResConvConvLogitPoolModel  --turn_features --bn --nnks
3 --nnsize 5 --history 2 --nb_layers_per_net 6 --nb_nets 31   --bn --server_connect_hostname tcp://$host:10023 --num_game 20  " &
#export host=learnfair0932; sbatch --array=0-179%25 --comment=icganote_dec19 --partition=priority --time=72:00:00 --mem=100Go --job-name=polytrain  --gres=gpu:8 --cpus
-per-task=50 --wrap="singularity exec --nv --overlay overlay.img /checkpoint/polygames/polygames_190927.simg python -u -m pypolygames train   --max_time=259200 --savin
g_period=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396
 --sync_period 256 --num_rollouts 600   --replay_capacity 2000000 --replay_warmup 200000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir "exps/yo`date | sed
's/ /_/g'`_$i" --out_feature --game_name havannah10pie --model_name ResConvConvLogitModel  --turn_features --bn --nnks 3 --nnsize 8 --nb_layers_per_net 6 --nb_nets 19
  --bn --server_connect_hostname tcp://$host:10023 --num_game 32  "
#export host=learnfair0932; sbatch --array=0-179%30 --comment=havperf --partition=learnfair --time=72:00:00 --mem=100Go --job-name=polytrain  --gres=gpu:8 --cpus-per-t
ask=50 --wrap="singularity exec --nv --overlay overlay.img /checkpoint/polygames/polygames_190927.simg python -u -m pypolygames train   --max_time=259200 --saving_peri
od=4 --num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --syn
c_period 256 --num_rollouts 600   --replay_capacity 2000000 --replay_warmup 200000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir "exps/yo`date | sed 's/ /_
/g'`_$i" --out_feature --game_name havannah10pie --model_name ResConvConvLogitModel  --turn_features --bn --nnks 3 --nnsize 8 --nb_layers_per_net 6 --nb_nets 19   --bn
 --server_connect_hostname tcp://$host:10023 --num_game 32  "
#export host=learnfair0932; sbatch --array=0-40%1 --comment=havperf  --partition=dev --time=72:00:00 --mem=100Go --job-name=polytrain  --gres=gpu:8 --cpus-per-task=50
--wrap="singularity exec --nv --overlay overlay.img /checkpoint/polygames/polygames_190927.simg python -u -m pypolygames train   --max_time=259200 --saving_period=4 --
num_game 120 --per_thread_batchsize 192   --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --nnks 3 --epoch_len 256 --batchsize 396 --sync_perio
d 256 --num_rollouts 600   --replay_capacity 2000000 --replay_warmup 200000 --do_not_save_replay_buffer --ddp true   --checkpoint_dir "exps/yo`date | sed 's/ /_/g'`_$i
" --out_feature --game_name havannah10pie --model_name ResConvConvLogitModel  --turn_features --bn --nnks 3 --nnsize 8 --nb_layers_per_net 6 --nb_nets 19   --bn --serv
er_connect_hostname tcp://$host:10023 --num_game 32  "
#sleep 7200
#done
