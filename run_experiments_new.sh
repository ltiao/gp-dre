#!/usr/bin/env bash

set -v
set -e   # fail if any command fails

# Number of runs (default: 10).
NUM_SEEDS=${1:-20}

# declare -a weights=("logreg" "vgp-mode" "rulsif" "kliep")
# declare -a weights=("kliep")

DATASET_PATH="logs/covariate_shift.h5"

# # Make dataset
# docker run -it --rm --gpus all \
#            -v "$PWD/scripts":/usr/src/app/scripts \
#            -v "$PWD/models":/usr/src/app/models \
#            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
#            python scripts/make_covariate_shift_example_2d.py ${DATASET_PATH} \
#            --importance-weights-filename=logs/exact.h5

for (( seed=0; seed<$NUM_SEEDS; seed++ ))
do
    # # Uniform importance weights
    # docker run -it --rm --gpus all \
    #            -v "$PWD/scripts":/usr/src/app/scripts \
    #            -v "$PWD/models":/usr/src/app/models \
    #            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #            python scripts/logistic_regression_covariate_shift_example_2d.py uniform ${DATASET_PATH} --seed=${seed}

    # # Exact density ratio as importance weights
    # docker run -it --rm --gpus all \
    #            -v "$PWD/scripts":/usr/src/app/scripts \
    #            -v "$PWD/models":/usr/src/app/models \
    #            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #            python scripts/logistic_regression_covariate_shift_example_2d.py exact ${DATASET_PATH} --sample-weights-filename=logs/exact.h5 --seed=${seed}

    # Logistic regression importance weights
    docker run -it --rm --gpus all \
               -v "$PWD/scripts":/usr/src/app/scripts \
               -v "$PWD/models":/usr/src/app/models \
               -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
               python scripts/generate_logreg_importance_weights.py logs/logreg-deep.${seed}.h5 ${DATASET_PATH} --num-layers=2 --num-units=20
    docker run -it --rm --gpus all \
               -v "$PWD/scripts":/usr/src/app/scripts \
               -v "$PWD/models":/usr/src/app/models \
               -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
               python scripts/logistic_regression_covariate_shift_example_2d.py logreg-deep ${DATASET_PATH} --sample-weights-filename=logs/logreg-deep.${seed}.h5 --seed=${seed}

    # # VGP importance weights
    # docker run -it --rm --gpus all \
    #            -v "$PWD/scripts":/usr/src/app/scripts \
    #            -v "$PWD/models":/usr/src/app/models \
    #            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #            python scripts/generate_vgp_importance_weights.py logs/vgp-mode.${seed}.h5 ${DATASET_PATH} --num-epochs=1000

    # # RuLSIF importance weights
    # docker run -it --rm --gpus all \
    #            -v "$PWD/scripts":/usr/src/app/scripts \
    #            -v "$PWD/models":/usr/src/app/models \
    #            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #            python scripts/generate_rulsif_importance_weights.py logs/rulsif.${seed}.h5 ${DATASET_PATH}

    # # KLIEP importance weights
    # docker run -it --rm --gpus all \
    #            -v "$PWD/scripts":/usr/src/app/scripts \
    #            -v "$PWD/models":/usr/src/app/models \
    #            -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #            python scripts/generate_kliep_importance_weights.py logs/kliep.${seed}.h5 ${DATASET_PATH}

    # for w in "${weights[@]}"
    # do
    #     docker run -it --rm --gpus all \
    #                -v "$PWD/scripts":/usr/src/app/scripts \
    #                -v "$PWD/models":/usr/src/app/models \
    #                -v "$PWD/logs":/usr/src/app/logs tiao/gp-dre \
    #                python scripts/logistic_regression_covariate_shift_example_2d.py ${w} ${DATASET_PATH} --sample-weights-filename=logs/${w}.${seed}.h5 --seed=${seed}
    # done
done
