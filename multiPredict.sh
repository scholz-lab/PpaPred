#!/bin/bash
# "Exp1_WT_OP50" "Exp1_WT_larvae" 
#"Exp2_tph1_larvae" "Exp2_tdc1_larvae" "Exp2_cat2_larvae" "Exp2_tbh1_larvae"
#"Exp3_octr1_larvae" "Exp3_ser3_larvae" "Exp3_ser6_larvae" "Exp3_tyra2_larvae" "Exp3_ser2_larvae" "Exp3_lgc55_larvae" "Exp3_tyra3_larvae" "Exp3_tdc1_larvae" "Exp3_tbh1_larvae" "Exp3_tbh1tdc1_larvae"
#"Exp2_WT_larvae" "Exp3_WT_larvae"
# "L147" "L157" "L176" "L118" "L119" "L156"

for VARIABLE in "Exp2_WT_larvae"
do
    sbatch Predict.sh $VARIABLE
done