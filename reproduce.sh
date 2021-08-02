#!/bin/bash

CLASSPATH=build/classes/java/main:$MEKA_JAR_PATH
DATA_DIR=$1
RESULTS_DIR=$2

for d in "yeast" "enron" "medical" "emotions" "genbase" "scene"
do
    TRAIN=$DATA_DIR/$d/$d-train.arff
    TEST=$DATA_DIR/$d/$d-test.arff

    java -cp $CLASSPATH meka.classifiers.multilabel.BR -verbosity 2 -t $TRAIN -T $TEST -W weka.classifiers.lazy.IBk -- -K 10 | tee $RESULTS_DIR/br-$d.log
    java -cp $CLASSPATH meka.classifiers.multilabel.MULAN -verbosity 2 -t $TRAIN -T $TEST -S MLkNN | tee $RESULTS_DIR/mlknn-$d.log
    java -cp $CLASSPATH meka.classifiers.multilabel.meta.BaggingML -verbosity 2 -t $TRAIN -T $TEST -P 100 -I 50 -W meka.classifiers.multilabel.CC -- -W weka.classifiers.lazy.IBk -- -K 10 | tee $RESULTS_DIR/ecc-$d.log
    java -cp $CLASSPATH meka.classifiers.multilabel.meta.FilteredClassifier -verbosity 2 -t $TRAIN -T $TEST -F "meka.filters.multilabel.JaccardEmbedder -D 16 -W \"weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1\"" -W meka.classifiers.multilabel.BR -- -W weka.classifiers.lazy.IBk -- -K 10 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" | tee $RESULTS_DIR/nje-br-$d.log
    java -cp $CLASSPATH meka.classifiers.multilabel.meta.FilteredClassifier -verbosity 2 -t $TRAIN -T $TEST -F "meka.filters.multilabel.JaccardEmbedder -D 16 -W \"weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1\"" -W meka.classifiers.multilabel.MULAN -- -S MLkNN | tee $RESULTS_DIR/nje-mlknn-$d.log
    java -cp $CLASSPATH meka.classifiers.multilabel.meta.FilteredClassifier -verbosity 2 -t $TRAIN -T $TEST -F "meka.filters.multilabel.JaccardEmbedder -D 16 -W \"weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1\"" -W meka.classifiers.multilabel.CC -- -W weka.classifiers.lazy.IBk -- -K 10 | tee $RESULTS_DIR/nje-ecc-$d.log
done
