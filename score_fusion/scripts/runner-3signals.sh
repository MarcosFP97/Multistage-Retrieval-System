### VALIDATION
# python evaluation/compat_evaluator.py eval-transformer-model --fold 3 --partition valid
# printf "\n\n"
## SCORE FUSION
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.95 --weights 0.01 --weights 0.04
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.95 --weights 0.04 --weights 0.04
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.95 --weights 0.025 --weights 0.025
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.90 --weights 0.03 --weights 0.07
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.90 --weights 0.07 --weights 0.03
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.90 --weights 0.05 --weights 0.05
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.85 --weights 0.04 --weights 0.11
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.85 --weights 0.11 --weights 0.04
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.85 --weights 0.075 --weights 0.075
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.85 --weights 0.05 --weights 0.10
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.85 --weights 0.10 --weights 0.05
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.80 --weights 0.06 --weights 0.14
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.80 --weights 0.14 --weights 0.06
printf "\n\n"
python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_4.csv --second-csv ./runs/credibility_TEST_4.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.80 --weights 0.10 --weights 0.10
printf "\n\n"

# ## TEST
# python evaluation/compat_evaluator.py eval-transformer-model --fold 3 --partition test
# printf "\n\n"
### SCORE FUSION
# python evaluation/compat_evaluator.py score-fusion --input-csv ./runs/readability_TEST_3.csv --second-csv ./runs/credibility_TEST_3.csv --signals score_monoT5 --signals score_readability --signals score_credibility --weights 0.95 --weights 0.01 --weights 0.04
# printf "\n\n"


