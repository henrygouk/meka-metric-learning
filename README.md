Multi-Label Metric Learning
===========================

This repository contains an implementation of the method described in [Learning Distance Metrics for Multi-Label Classification](http://proceedings.mlr.press/v63/Gouk8.html). The implementation is written in Java and makes use of the [MEKA framework](https://waikato.github.io/meka). If you use this in a research context, please consider citing:

```
@InProceedings{gouk2016,
    title = {Learning Distance Metrics for Multi-Label Classification},
    author = {Henry Gouk and Bernhard Pfahringer and Michael Cree},
    booktitle = {Proceedings of The 8th Asian Conference on Machine Learning},
    year = {2016},
    month = {16--18 Nov},
    publisher = {PMLR}
} 
```

Usage
=====

The project uses the Gradle build system. To get started you will first need to compile the Java code using either the `gradlew` script (on UNIX-like systems) or the `gradlew.bat` script (on Windows). On Linux, this can be done by running the following command from the root project directory:

```
    ./gradlew build
```

To open the MEKA Explorer GUI with the metric learning classes pre-loaded, you can run the following Gradle task

```
    ./gradlew runGUI
```

The methods are implemented as filters that can be used with the `meka.classifiers.meta.FilteredClassifier` method.

Results
=======

Due to using a customised version of MEKA when writing the original paper, there have been issues reproducing the exact results. I've rerun the experiments used to generate Table 3 in the paper with MEKA 1.9.5:

F1 (macro averaged by example)
---

|          |     BR    |    ECC    |   MLkNN   |   NJE-BR  |  NJE-ECC  | NJE-MLkNN |
| -------- | --------- | --------- | --------- | --------- | --------- | --------- |
| yeast    | 0.65      | 0.649     | 0.645     | 0.655     | 0.649     | 0.645     |
| enron    | 0.464     | 0.43      | 0.501     | 0.589     | 0.561     | 0.58      |
| medical  | 0.507     | 0.484     | 0.595     | 0.681     | 0.677     | 0.624     |
| emotions | 0.668     | 0.642     | 0.657     | 0.639     | 0.671     | 0.634     |
| genbase  | 0.96      | 0.961     | 0.98      | 0.985     | 0.982     | 0.984     |
| scene    | 0.7       | 0.697     | 0.703     | 0.777     | 0.765     | 0.744     |


Hamming loss
---

|          |     BR    |    ECC    |   MLkNN   |   NJE-BR  |  NJE-ECC  | NJE-MLkNN |
| -------- | --------- | --------- | --------- | --------- | --------- | --------- |
| yeast    | 0.225     | 0.207     | 0.207     | 0.208     | 0.201     | 0.209     |
| enron    | 0.07      | 0.064     | 0.06      | 0.059     | 0.053     | 0.055     |
| medical  | 0.025     | 0.03      | 0.021     | 0.019     | 0.017     | 0.02      |
| emotions | 0.19      | 0.196     | 0.195     | 0.21      | 0.209     | 0.206     |
| genbase  | 0.005     | 0.007     | 0.003     | 0.003     | 0.003     | 0.003     |
| scene    | 0.114     | 0.109     | 0.104     | 0.087     | 0.084     | 0.091     |


Jaccard index
---

|          |     BR    |    ECC    |   MLkNN   |   NJE-BR  |  NJE-ECC  | NJE-MLkNN |
| -------- | --------- | --------- | --------- | --------- | --------- | --------- |
| yeast    | 0.54      | 0.546     | 0.537     | 0.555     | 0.553     | 0.545     |
| enron    | 0.356     | 0.34      | 0.368     | 0.472     | 0.454     | 0.469     |
| medical  | 0.459     | 0.422     | 0.547     | 0.638     | 0.644     | 0.58      |
| emotions | 0.586     | 0.573     | 0.571     | 0.558     | 0.587     | 0.557     |
| genbase  | 0.945     | 0.948     | 0.971     | 0.977     | 0.974     | 0.977     |
| scene    | 0.664     | 0.672     | 0.675     | 0.753     | 0.751     | 0.721     |


Log Loss (lim. L)
---

|          |     BR    |    ECC    |   MLkNN   |   NJE-BR  |  NJE-ECC  | NJE-MLkNN |
| -------- | --------- | --------- | --------- | --------- | --------- | --------- |
| yeast    | 0.431     | 0.446     | 0.438     | 0.48      | 0.529     | 0.504     |
| enron    | 0.155     | 0.209     | 0.144     | 0.155     | 0.21      | 0.167     |
| medical  | 0.06      | 0.068     | 0.056     | 0.054     | 0.065     | 0.053     |
| emotions | 0.403     | 0.385     | 0.417     | 0.374     | 0.374     | 0.391     |
| genbase  | 0.01      | 0.013     | 0.011     | 0.006     | 0.01      | 0.009     |
| scene    | 0.222     | 0.192     | 0.207     | 0.155     | 0.151     | 0.186     |