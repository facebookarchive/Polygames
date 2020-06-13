for i in 4096 2048 1024 512 256 128 64
do                                                                                                                                                                     |
# LD_PRELOAD=/private/home/vegardmella/libjemalloc.so.1 \       # uncomment on H2
python -u -m pypolygames train --max_time=259200 --saving_period=4 --num_game 40 --per_thread_batchsize 12 --device cuda:0 cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 --epoch_len 256 --batchsize $i --sync_period 32 --num_rollouts 600 --replay_capacity 100000 --replay_warmup 9000 --do_not_save_replay_buffer --ddp true --checkpoint_dir exps/minishogi --out_feature --game_name minishogi --model_name ResConvConvLogitPoolModel --turn_features --bn --nnks 3 --history 2 --nnsize 5 --nb_nets 31 --nb_layers_per_net 6 --nnks 3 --server_listen_endpoint tcp://*:10023 --num_game 0 --lr 1e-5
done
