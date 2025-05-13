#!/bin/bash
for domain in "astronomy" "biomedical"
do
    for idx in {0..13}
    do
        python workload/messy_subtasks/annotate_subtask_script.py --idx $idx --domain $domain
    done
done