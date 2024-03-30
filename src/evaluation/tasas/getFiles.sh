#!/bin/bash

rm hyp_*
rm pred_*

modality=$1
for system in $2; do

if [[ "$system" == *"smbr"* || "$system" == *"mmi"* || "$system" == *"mpe"* ]]; then
   best_iter=1
   best_wer=500.0
   for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        out=$(cat ../exp/${modality}/${system}/decode_test_${i}/scoring_kaldi/best_wer)
        wer=$(echo $out | cut -f 2 -d ' ')
        if [[ $(echo "$wer < $best_wer" | bc -l) == 1 ]]; then best_iter=$i; best_wer=$wer; fi;

   done;
   output=$(cat ../exp/${modality}/${system}/decode_test_${best_iter}/scoring_kaldi/best_wer)
   lmwt=$(echo $output | cut -f5 -d_)
   penalty=$(echo $output | cut -f6 -d_)
   sort ../exp/${modality}/${system}/decode_test_${best_iter}/scoring_kaldi/penalty_${penalty}/${lmwt}.txt | cut -f 2- -d ' ' > ./pred_${lmwt}_${penalty}.txt
else
    output=$(cat ../exp/${modality}/${system}/decode_test/scoring_kaldi/best_wer)
    lmwt=$(echo $output | cut -f3 -d_)
    penalty=$(echo $output | cut -f4 -d_)
    sort ../exp/${modality}/${system}/decode_test/scoring_kaldi/penalty_${penalty}/${lmwt}.txt | cut -f 2- -d ' ' > ./pred_${lmwt}_${penalty}.txt
fi;
#for lmwt in 15 16 17 18; do
#for penalty in 1.0 2.0 3.0; do

# Generamos el fichero que contiene las predicciones de la mejor decodificacion del sistema
#cut -f 2- -d ' ' ../exp/${modality}/${system}/decode_test/scoring_kaldi/penalty_${penalty}/${lmwt}.txt > ./pred_${lmwt}_${penalty}.txt
paste -d '#' groundtruth.txt pred_${lmwt}_${penalty}.txt > hyp_${lmwt}_${penalty}.txt

./tasas -s " " -f "#" -ie hyp_${lmwt}_${penalty}.txt
echo ""
echo "--------------------------------------------------------------------------------"
echo ""
./tasasIntervalo -s " " -f "#" -ie hyp_${lmwt}_${penalty}.txt

done;
#done;
#done;

