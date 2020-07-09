#!/bin/bash

index=$SLURM_ARRAY_TASK_ID

counter=0
for i in $(cat models.txt); do
  if [[ $counter == $index ]]; then
    ni=$(basename $(dirname $i))__$(basename $i)
    for i2 in $(cat models.txt); do
      ni2=$(basename $(dirname $i2))__$(basename $i2)
      echo "subeval $ni / $ni2"
      for n in $(seq 0 5); do
        c=$(expr $n % 4)
        python eval.py $i $i2 cuda:$c > results/$ni/$ni2/$n.txt &
      done
      time wait
    done
  fi
  counter=$(expr $counter + 1)
done

