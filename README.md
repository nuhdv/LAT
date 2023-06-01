# FLAD
SMC23 - Anomaly Detection

This is the source code of the paper  **"Local-Adaptive Transformer for Multivariate Time Series Anomaly Detection and Diagnosis"** accepted by SMC23 (to appear).

[//]: # "## Citation"

[//]: # "Please cite our paper if you find this code is useful.  "

[//]: # "Zhou Xiaohui, Wang Yijie, Xu Hongzuo, Liu Mingyu. Local-Adaptive Transformer for Multivariate Time Series Anomaly Detection and Diagnosis"

## Usage
1. run main.py for sample usage. 
2. Data set: You may want to find the sample input data set in the "datasets" folder.
3. The input path can be an individual data set or just a folder.  
4. The performance might have slight differences between two independent runs. In our paper, we report the average auc with std over 5 runs. 


## Dependencies
```
Python 3.6
Troch == 1.7.0+cu110
pandas == 1.1.5
scikit-learn == 1.0.1
numpy == 1.21.6
```