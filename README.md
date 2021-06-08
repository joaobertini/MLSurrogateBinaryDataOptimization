# Machine Learning Surrogate Models
> Codes from paper "A comparison of machine learning surrogate models for net present value prediction from well placement binary data"

## Task
- Well placement optimization

## Objective 
- Build a surrogate model for NPV forcasting taking binary well placement data as input
- Perform well placement optimization using a surrogate model as objective function in a optimization method

## Capabilities
- Evaluate surrogate models either built by:
  - A machine learning regression model (RM)
  - Or a dimensionality reduction algorithm (DR) and a machine learning regression model
- Implemented RMs:
  - Support Vector Regression (SVR)
  - Kernel Ridge Regression (KRR)
  - Muilt-layer Perceptron (MLP)
  - Elastic Net (ENET)
  - Gradient Tree Boosting (GTB)
  - K-Nearest Neighbor (KNN)

## Methodology
- Best model found by nested cross-validation
  - Inner loop does model selection (parameter adjustment)
  - Outter loop evaluates model in validation set
 - **Output:** best model RMSE calculated using an out-of-sample test data
 - **Output:** scatter plot of the best model predictions against the simulator output - consider whole data set

## Dependencies

```bash
pip3 install numpy pandas matplotlib scikit-learn
```

## Running

```bash
python3 run.py DATASET_NAME MODEL_NAME REDUCER_NAME REDUCER_DIMENSION
```

Examples:
```bash
python3 run.py dataUNISIM1 GTB PCA 5
```

```bash
python3 run.py dataUNISIM1 KRR NONE 0
```

