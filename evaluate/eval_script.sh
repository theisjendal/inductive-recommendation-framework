#!/bin/bash

# all methods: toppop bpr ppr inmo bert4rec idcf | simplerec2 pinsage graphsage
settings="standard cold_user cold_item warm_user cold_item_limit cold_user_item cold_no_item_ratings"
settings="cold_no_item_ratings"

## ------- METHOD USING FEATURES
models="ginrec pinsage graphsage"

datasets=( "ml_1m_temporal" "ab_temporal" "yk_temporal" )

parameters=( "" "ml_1m_temporal" "ml_1m_temporal" )
RESULT_PATH="../results"

FEATURE_CONF="comsage"

for i in "${!datasets[@]}"; do
  set -e
  dataset=${datasets[i]}
  parameter=${parameters[i]}
  ext="features_${FEATURE_CONF}${parameter:+_parameter_${parameter}}"
  if [ -n "$other_models" ]; then
    for model in $models; do
      for other_model in $other_models; do
        if  [[ $model == ${other_model}* ]]; then
          inner_ext="${ext}_model_${other_model}"
          cmd="python3 cold_eval_converter.py --result_path ${RESULT_PATH} --models ${model} --experiment ${dataset} --ext ${inner_ext} --settings ${settings}"
          echo $cmd
          eval $cmd
        fi
      done
    done
  else
    cmd="python3 cold_eval_converter.py --result_path ${RESULT_PATH} --models ${models} --experiment ${dataset} --ext ${ext} --settings ${settings}"
    echo $cmd
    eval $cmd
  fi

  set +e
  for setting in $settings; do
    if [[ -n $other_models ]]; then
      for model in $models; do
        for om in $other_models; do
          if [[ $model == $om* ]]; then
            other_model=$om
            cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${model} --folds fold_0 --experiment_setting ${setting} --feature_configuration ${FEATURE_CONF} ${parameter:+--parameter ${parameter}} ${other_model:+--other_model ${other_model}}"
            echo $cmd;
            eval $cmd;
          fi
        done
      done
    else
        cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${models} --folds fold_0 --experiment_setting ${setting} --feature_configuration ${FEATURE_CONF} ${parameter:+--parameter ${parameter}}"
        echo $cmd;
        eval $cmd;
    fi
  done
done

models="random toppop bpr ppr ppr-cf inmo idcf"
for i in "${!datasets[@]}"; do
  set -e
  dataset=${datasets[i]}
  parameter=${parameters[i]}
  ext="${parameter:+parameter_${parameter}}"
  if [ -n "$other_models" ]; then
    for model in $models; do
      for other_model in $other_models; do
        if  [[ $model == ${other_model}* ]]; then
          inner_ext="${ext:+${ext}_}model_${other_model}"
          cmd="python3 cold_eval_converter.py --result_path ${RESULT_PATH} --models ${model} --experiment ${dataset} --ext ${inner_ext} --settings ${settings}"
          echo $cmd
          eval $cmd
        fi
      done
    done
  else
    cmd="python3 cold_eval_converter.py --result_path ${RESULT_PATH} --models ${models} --experiment ${dataset} ${ext:+--ext ${ext}} --settings ${settings}"
    echo $cmd
    eval $cmd
  fi

  set +e
  for setting in $settings; do
    if [[ -n $other_models ]]; then
      for model in $models; do
        for om in $other_models; do
          if [[ $model == $om* ]]; then
            other_model=$om
            cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${model} --folds fold_0 --experiment_setting ${setting} ${parameter:+--parameter ${parameter}} ${other_model:+--other_model ${other_model}}"
            echo $cmd;
            eval $cmd;
          fi
        done
      done
    else
        cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${models} --folds fold_0 --experiment_setting ${setting} ${parameter:+--parameter ${parameter}}"
        echo $cmd;
        eval $cmd;
    fi
  done
done
