# Launch a server on dev.
sbatch --partition=dev --time=72:00:00 --mem=150Go --job-name=pgserver  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh"

# Wait a bit for launching the clients
sleep 200
./launch_havannah10_clients.sh
sleep 86400

# We want the same machine for the next server.
export host=`squeue -u oteytaud | grep -i pgserver | sed 's/.*learnfair/learnfair/g' | sed 's/ //g'`
echo "host=<${host}>"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser2  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh"
sbatch -w $host --partition=dev --time=72:00:00 --mem=150Go --job-name=pgser3  --gres=gpu:8 --cpus-per-task=80 --wrap="./launch_havannah10_server.sh"

# And more clients.
./launch_havannah10_clients.sh
sleep 86400

sleep 86400
./launch_havannah10_clients.sh

sleep 86400
./launch_havannah10_clients.sh
