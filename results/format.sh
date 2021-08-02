#!/bin/bash

for m in "F1 (macro averaged by example)" "Hamming loss" "Jaccard index" "Log Loss (lim. L)"
do
    echo $m
    echo "---"
    echo

    echo "|          |     BR    |    ECC    |   MLkNN   |   NJE-BR  |  NJE-ECC  | NJE-MLkNN |"
    echo "| -------- | --------- | --------- | --------- | --------- | --------- | --------- |"

    for d in "yeast" "enron" "medical" "emotions" "genbase" "scene"
    do
        printf "| %-8s |" $d

        ls * | grep $d | \
            xargs -I% sh -c "cat % | grep '$m' | tr -s ' ' | sed 's/ *$//g' | rev | cut -d ' ' -f 1 | rev" | \
            xargs -I{} sh -c "printf ' %-9s |' {}"

        echo
    done

    echo
    echo
done
