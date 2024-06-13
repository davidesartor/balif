#!/bin/bash

# Define different sets of parameters to run runsim.py with
# Each line represents a different set of parameters
comand="/home/davide_sartor/.conda/envs/torch_env/bin/python /home/davide_sartor/balif/runsim.py" 
params=(
  "--hyperplane_components 1 --p_normal_idx uniform --query_strategy margin"
  "--hyperplane_components 1 --p_normal_idx range --query_strategy margin"
  "--hyperplane_components -1 --p_normal_idx uniform --query_strategy margin"
  "--hyperplane_components -1 --p_normal_idx range --query_strategy margin"
  "--hyperplane_components 1 --p_normal_idx uniform --query_strategy random"
  "--hyperplane_components 1 --p_normal_idx range --query_strategy random"
  "--hyperplane_components -1 --p_normal_idx uniform --query_strategy random"
  "--hyperplane_components -1 --p_normal_idx range --query_strategy random"
)

# Loop through each set of parameters and run runsim.py with them
for p in "${params[@]}"; do
  CUDA_VISIBLE_DEVICES=$1 $comand $p &
done

wait