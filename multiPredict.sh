#!/bin/bash
# "CondA" "CondB" "CondC"
for VARIABLE in "CondA" "CondB" "CondC"
do
    sbatch Predict.sh $VARIABLE
done
