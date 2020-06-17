# Launch this script, wait 10 days, and you should get an excellent minishogi model.

export me=`whoami`                                                                                                                                                            |
sbatch --partition=dev --time=72:00:00 --mem=150Go --job-name=pgserver  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_minishogi_server.sh"                          |
sleep 200
export host=`squeue -u $me| grep -i pgserver | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`
echo "host=<${host}>"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser2  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_minishogi_server.sh"
sleep 60
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser3  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_minishogi_server.sh"
sleep 60
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser4  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_minishogi_server.sh"
sleep 60
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser5  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_minishogi_server.sh"
./launch_minishogi_clients.sh
sleep 86400
./launch_minishogi_clients.sh
sleep 86400
./launch_minishogi_clients.sh
