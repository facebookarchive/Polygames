mkdir results || exit


for i in $(cat models.txt); do
  ni=$(basename $(dirname $i))__$(basename $i)
  mkdir results/$ni || exit
  for i2 in $(cat models.txt); do
    ni2=$(basename $(dirname $i2))__$(basename $i2)
    mkdir results/$ni/$ni2 || exit
  done
done

x=$(cat models.txt | wc -l)

#bash subeval.sh $i 0
#sbatch -p uninterrupted -c 40 --gres=gpu:4 --mem 200G --time 30 -a 0-$x subeval.sh $i
sbatch -p dev,uninterrupted,learnfair -c 40 --gres=gpu:4 --mem 200G -a 0-$x --time 900 subeval.sh $i
#break


