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