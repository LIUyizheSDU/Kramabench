#!/bin/bash
for workload in "archeology" "astronomy" "biomedical" "legal" "environment" "wildfire"; do
for sut_name in "BaselineLLMSystemDeepseekR1" "BaselineLLMSystemGPTo3" "BaselineLLMSystemGPT4o" "BaselineLLMSystemQwen2_5Coder" BaselineLLMSystemLlama3_3Instruct;  do
for config in "Naive" "OneShot" "FewShot"; do
    # echo "Running workload: $workload with SUT: $sut_name$config"
    screen -d -m python evaluate.py --sut $sut_name$config --workload $workload.json --dataset_name $workload
done
done
done
