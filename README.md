# Machine Learning Surrogate Models
> Codes from paper "A comparison of machine learning surrogate models for net present value prediction from well placement binary data"

## Task
- Well placement optimization

## Objective 
- Build a surrogate model for NPV forcasting taking binary well placement data as input
- Perform well placement optimization using a surrogate model as objective function in a optimization method

## Capabilities
- Evaluate surrogate models built by a machine learning regression model (RM)
- Implemented RMs:
  - Support Vector Regression (SVR)
  - Kernel Ridge Regression (KRR)
  - Muilt-layer Perceptron (MLP)
  - Elastic Net (ENET)
  - Gradient Tree Boosting (GTB)
  - K-Nearest Neighbor (KNN)
- Use Genetic Algorithm (GA) to search for a new production strategy

## Methodology
- Best model found by nested cross-validation
  - Inner loop does model selection (parameter adjustment)
  - Outter loop evaluates model in validation set
  - Genetic algorithm searches for a new production strategy using the model
- **Output:** best model RMSE calculated using an out-of-sample test data
- **Output:** scatter plot of the best model predictions against the simulator output - consider whole data set

## Dependencies

```bash
pip3 install numpy scikit-learn pandas matplotlib xgboost
```

## Running

```bash
python3 run.py MODEL_NAME
```

Where `MODEL_NAME` must be one of `GTB`, `KRR`, `MLP`, `ENET`, `KNN`, `GPR`, `XGBOOST` or `SVR`

Examples:
```bash
python3 train.py GTB
```

```bash
python3 train.py KRR
```

