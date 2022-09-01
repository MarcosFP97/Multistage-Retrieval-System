# ### VALIDATION
# python evaluation/compat_evaluator.py eval-transformer-model --fold 3 --partition valid
# printf "\n\n"
CSV="runs/trec-2022/transformers/fusion_trec_2022/bert-large-uncased_f$1_6_2020+2021_6_1.00e-07_0.111_valid.csv"
MANUEL="-manuel --fold $1"
#MANUEL=""

# ## SCORE FUSION
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.99 --weights 0.01
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.95 --weights 0.05
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.90 --weights 0.10
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.85 --weights 0.15
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.80 --weights 0.20
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion$MANUEL --input-csv $CSV --weights 0.75 --weights 0.25
printf "\n\n"

# ## TEST
# python evaluation/compat_evaluator.py eval-transformer-model --fold 3 --partition test
# printf "\n\n"
### SCORE FUSION
# python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_3.csv --weights 0.95 --weights 0.05
# printf "\n\n"


