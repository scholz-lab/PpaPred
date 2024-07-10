#!/bin/bash
# Eren Boeger 2024
# Exp1:"Exp1_WT_larvae" "Exp1_WT_OP50"      
# Exp2: "Exp2_WT_larvae" "Exp2_tph1_larvae" "Exp2_cat2_larvae" "Exp2_tdc1_larvae" "Exp2_tbh1_larvae" 
# Exp3b: "Exp3_tdc1_larvae" "Exp3_tyra2_larvae" "Exp3_ser2_larvae" "Exp3_tyra3_larvae" "Exp3_lgc55_larvae"  "Exp3_tbh1tdc1_larvae"
# Exp3a:"Exp3_WT_larvae" "Exp3_tbh1_larvae" "Exp3_ser3_larvae" "Exp3_ser6_larvae" "Exp3_octr1_larvae" 
# Supp:  "Supp5_tyramine_larvae""Supp5_octopamine_larvae" "Supp7_ser2tyra2tyra3_larvae"

# roca: "L147" "L157" "L176" "L118" "L119" "L156"

for VARIABLE in "Exp1_WT_larvae" "Exp1_WT_OP50"
do
    sbatch Predict.sh $VARIABLE
done
