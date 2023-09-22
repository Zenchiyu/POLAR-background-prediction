#!/bin/bash

RANDOM=42  # init
for training_num in {1..10}
do
    seed=$RANDOM
    echo "Starting training n°: ${training_num} with seed: ${seed}"
    python src/main.py common.seed=$seed
done
