# in fact, math model is in another repo
MATH_MODEL="meta-math/MetaMath-7B-V1.0"
LAW_MODEL="AdaptLLM/law-chat"
MED_MODEL="AdaptLLM/medicine-chat"
FIN_MODEL="AdaptLLM/finance-chat"
OOD_MODEL="meta-llama/Llama-2-7b-chat-hf"

MATH_IN_DIST=("GSM8K" "MATH")
MATH_OUT_DIST=("boolq" "PIQA" "SQUAD")

LAW_IN_DIST=("casehold")
LAW_OUT_DIST=("boolq" "PIQA" "SQUAD" "FPB" "PubMedQA")

FIN_IN_DIST=("FPB")
FIN_OUT_DIST=("boolq" "PIQA" "SQUAD" "casehold" "PubMedQA")

MED_IN_DIST=("PubMedQA")
MED_OUT_DIST=("boolq" "PIQA" "SQUAD" "casehold" "FPB")

SAMPLE_SUFFIX="--tensor_parallel_size 1 --end 50"
# SAMPLE_SUFFIX="--tensor_parallel_size 1"

# $1: model, $2: dataset
function sample_and_get_likelihood {
  python sample.py --model $1 --dataset $3 $SAMPLE_SUFFIX
  for form in "q" "a" "qa"
  do
    python get_likelihood.py --in_model $1 --out_model $2 --dataset $3 --form $form --mode "in"
    python get_likelihood.py --in_model $1 --out_model $2 --dataset $3 --form $form --mode "out"
  done
}

# Math model
# for dataset in "${MATH_IN_DIST[@]}" "${MATH_OUT_DIST[@]}"
# do
#   sample_and_get_likelihood $MATH_MODEL $dataset
# done
# # evaluation MATH_IN_DIST MATH_OUT_DIST $MATH_MODEL
# for datasetin in "${MATH_IN_DIST[@]}"
# do
#   for datasetout in "${MATH_OUT_DIST[@]}"
#   do
#     for mode in "q" "qa" "a" "agivenq"
#     do
#       python eval_ood.py --in_model $MATH_MODEL --out_model $OOD_MODEL --in_dist $datasetin --out_dist $datasetout --mode $mode
#     done
#   done
# done

# Law model
for dataset in "${LAW_IN_DIST[@]}" "${LAW_OUT_DIST[@]}"
do
  sample_and_get_likelihood $LAW_MODEL $OOD_MODEL $dataset
done
# evaluation LAW_IN_DIST LAW_OUT_DIST $LAW_MODEL
for datasetin in "${LAW_IN_DIST[@]}"
do
  for datasetout in "${LAW_OUT_DIST[@]}"
  do
    for mode in "q" "qa" "a" "agivenq"
    do
      python eval_ood.py --in_model $LAW_MODEL --out_model $OOD_MODEL --in_dist $datasetin --out_dist $datasetout --mode $mode
    done
  done
done

# Finance model
for dataset in "${FIN_IN_DIST[@]}" "${FIN_OUT_DIST[@]}"
do
  sample_and_get_likelihood $FIN_MODEL $OOD_MODEL $dataset
done
for datasetin in "${FIN_IN_DIST[@]}"
do
  for datasetout in "${FIN_OUT_DIST[@]}"
  do
    for mode in "q" "qa" "a" "agivenq"
    do
      python eval_ood.py --in_model $FIN_MODEL --out_model $OOD_MODEL --in_dist $datasetin --out_dist $datasetout --mode $mode
    done
  done
done

# Medicine model
for dataset in "${MED_IN_DIST[@]}" "${MED_OUT_DIST[@]}"
do
  sample_and_get_likelihood $MED_MODEL $OOD_MODEL $dataset
done
for datasetin in "${MED_IN_DIST[@]}"
do
  for datasetout in "${MED_OUT_DIST[@]}"
  do
    for mode in "q" "qa" "a" "agivenq"
    do
      python eval_ood.py --in_model $MED_MODEL --out_model $OOD_MODEL --in_dist $datasetin --out_dist $datasetout --mode $mode
    done
  done
done

