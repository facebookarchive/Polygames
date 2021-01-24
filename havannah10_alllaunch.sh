sbatch --partition=dev --time=72:00:00 --mem=150Go --job-name=pgserver  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh"
sleep 200
export host=`squeue -u oteytaud | grep -i pgserver | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`
echo "host=<${host}>"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser2  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh
"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser3  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh
"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser3  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh
"
./launch_havannah10_clients.sh
sleep 86400
./launch_havannah10_clients.sh
sleep 86400
./launch_havannah10_clients.sh
sleep 86400
./launch_havannah10_clients.sh
