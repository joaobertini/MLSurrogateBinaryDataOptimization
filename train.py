
import os
import sys
import datetime
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

from utils import savePlot, saveCsv
from genetic_algorithm import genetic_algorithm

testing = False

if testing:
  print('TESTING')

single_processor = True

trials = 10 if not testing else 3
genetic_iterations = 5000 if not testing else 20 # 2000
genetic_populations = 500 if not testing else 50 # 200
dataset_path = './datasets/dataExp1.txt' if not testing else './datasets/test.txt'
output_directory = './output' if not testing else './output_test'
cross_validations = 10 if not testing else 3
num_splits = 10 if not testing else 4

assert len(sys.argv) >= 2, 'Please specify all the required arguments when running this script'

regressor_name = sys.argv[1].upper()


regressors = {
  'GTB': {
    'function': GradientBoostingRegressor,
    'params': {},
    'search_params': {
      'min_samples_split': [0.05, 0.1, 0.2, 0.3],
      'n_estimators': [50, 100, 150],
      'learning_rate': [0.01, 0.1, 0.5],
      'loss': ['ls', 'lad', 'huber']
    }
  },
  'KRR': {
    'function': KernelRidge,
    'params': {},
    'search_params': [
      {'kernel': ['poly'], 'degree': [2,3,4], 'alpha': [1e0, 0.1, 1e-2, 1e-3]},
      {'kernel': ['rbf'], 'gamma':  np.logspace(-3, 3, 7), 'alpha': [1e0, 0.1, 1e-2, 1e-3]}  #[1e-3, 1e-1, 1e1]
    ]
  },
  'GPR': {
    'function': GaussianProcessRegressor,
    'params': {},
    'search_params': [
      {'kernel': [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))], 'alpha': np.logspace(-2, 0, 3)},
      {'kernel': [1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)],'alpha': np.logspace(-2, 0, 3)},
      {'kernel': [1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0))],'alpha': np.logspace(-2, 0, 3)}
    ]
  },
  'MLP': {
    'function': MLPRegressor,
    'params': { 'max_iter': 400, 'verbose': 0 },
    'search_params': {
      'learning_rate': ["invscaling"],
      'learning_rate_init': [0.001, 0.01, 0.1],
      'hidden_layer_sizes': [(25,), (50), (100,), (150,), (50,25), (50,50), (100,50), (100, 100), (150, 100)],
      'activation': ["logistic", "relu", "tanh"]
    }
  },
  'SVR': {
    'function': SVR,
    'params': {},
    'search_params': {
      'kernel': ['rbf'],
      'gamma': np.logspace(-3, 3, 7), # [1e-5, 1e-3, 1e-1, 1e1],
      'C': [1e1, 1e3, 1e5, 1e7]
    }
  },
  'ENET': {
    'function': ElasticNet,
    'params': { 'max_iter': 100000 },
    'search_params': {
      'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
      'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
    }
  },
  'KNN': {
    'function': KNeighborsRegressor,
    'params': { 'n_jobs': 1 if single_processor else -1 },
    'search_params': {
      'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      'weights': ('uniform', 'distance')
    }
  },
  'XGBOOST': {
    'function': XGBRegressor,
    'params': { 'objective': 'reg:squarederror', 'n_jobs': 1 if single_processor else -1 },
    'search_params': {
      'n_estimators': [50, 100, 150],
      'learning_rate': [0.01, 0.1, 0.5],
       # 'booster': ['gbtree', 'gblinear', 'dart']
    },
  },
}


def load_csv(file_path):
  return np.loadtxt(file_path, delimiter='\t', dtype=np.float64)

def r2_adj(observation, prediction):
    r2 = r2_score(observation, prediction)
    (n, p) = observation.shape
    return 1 - (1-r2) * (n-1) / (n-p-1)

train_durations = []
test_durations = []

dataset = np.loadtxt(dataset_path, delimiter='\t', dtype=np.float64)

for trial in range(1, trials + 1):
  output_path = output_directory + '/' + regressor_name + '/TRIAL_' + str(trial) + '/'
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  with open(output_path + '/output.txt', 'w') as output_file:
    if testing:
      print('TESTING', file=output_file, flush=True)
      print(file=output_file)

    print('Command line arguments: ' + str(sys.argv), file=output_file, flush=True)

    print('Trial ' + str(trial), file=output_file, flush=True)

    print(file=output_file)
    print(regressor_name, flush=True)
    print(regressor_name, file=output_file, flush=True)
    print(file=output_file)

    RMSEs = []
    R2s = []

    best_regressor = None
    best_RMSE = None
    best_R2 = None
    best_y_scaler = None
    current_test_features = None
    current_test_y = None

    k_fold = ShuffleSplit(n_splits=num_splits, random_state=trial, test_size=0.1)

    for train_indices, test_indices in k_fold.split(dataset):
      train_data = dataset[train_indices, :]
      test_data = dataset[test_indices, :]

      regressor_info = regressors[regressor_name]
      regressor_function = regressor_info['function'](**regressor_info['params'])

      # Train

      train_start = datetime.datetime.now()

      train_features = train_data[:, :-1]
      train_y = train_data[:, -1].reshape(-1, 1)
      y_scaler = MinMaxScaler().fit(train_y)
      train_y = y_scaler.transform(train_y)

      regressor = GridSearchCV(
        estimator=regressor_function,
        param_grid=regressor_info['search_params'],
        cv=cross_validations,
        n_jobs=1 if single_processor else -1,
        verbose=0
      )

      regressor.fit(train_features, train_y.ravel())

      train_end = datetime.datetime.now()
      train_durations.append((train_end - train_start).total_seconds())

      # Test

      test_start = datetime.datetime.now()

      test_features = test_data[:, :-1]
      test_y = test_data[:, -1].reshape(-1, 1)
      test_y = y_scaler.transform(test_y)

      predicted_y = regressor.predict(test_features).reshape(-1, 1)

      test_end = datetime.datetime.now()
      test_durations.append((test_end - test_start).total_seconds())

      RMSE = np.sqrt(mean_squared_error(test_y, predicted_y))
      R2 = r2_adj(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))

      RMSEs.append(RMSE)
      R2s.append(R2)

      if best_RMSE is None or RMSE < best_RMSE:
        print('- RMSE: ' + str(RMSE) + ' R2: ' + str(R2), file=output_file, flush=True)

        best_regressor = regressor
        best_RMSE = RMSE
        best_R2 = R2
        best_y_scaler = y_scaler
        current_test_features = test_features
        current_test_y = test_y

    print(file=output_file)
    print("Best RMSE %f" % (best_RMSE), file=output_file)
    print("Best R2 %f" % (best_R2), file=output_file)

    # Save predictions from best iteration

    prediction = best_regressor.predict(current_test_features).reshape(-1, 1)

    denormalized_test_y = best_y_scaler.inverse_transform(current_test_y)
    denormalized_prediction = best_y_scaler.inverse_transform(prediction)

    saveCsv(
      filename=output_path + '/csv',
      prediction=denormalized_prediction,
      original=denormalized_test_y.ravel()
    )
    savePlot(
      filename=output_path + '/english',
      x=denormalized_prediction,
      y=denormalized_test_y.ravel(),
      english=True
    )
    savePlot(
      filename=output_path + '/portuguese',
      x=denormalized_prediction,
      y=denormalized_test_y.ravel(),
      english=False
    )

    # # Find a new combination with genetic algorithm

    print(file=output_file, flush=True)
    print("Genetic algorithm", file=output_file, flush=True)
    print(file=output_file, flush=True)

    genetic_start = datetime.datetime.now()

    best_combination, beval = genetic_algorithm(
      objective=best_regressor.predict,
      n_bits=train_features.shape[1],
      n_iter=genetic_iterations,
      n_pop=genetic_populations,
      r_cross=0.90,
      r_mut=0.05,
      logger=lambda *x : print(*x, file=output_file, flush=True)
    )
    # best, beval = genetic_algorithm(best_regressor.predict, 117, 2000, 200, 0.90, 0.05)

    genetic_end = datetime.datetime.now()
    genetic_duration = (test_end - test_start).total_seconds()

    # Save summary

    print(file=output_file)
    print(
      "Best combination: " + ', '.join(str(v) for v in best_combination),
      file=output_file, flush=True
    )
    print("Normalized beval: %f" % (beval), file=output_file, flush=True)

    denormalized_beval = best_y_scaler.inverse_transform(np.array([beval]).reshape(-1, 1))
    print("Denormalized beval: %f" % (denormalized_beval), file=output_file, flush=True)
    print(file=output_file, flush=True)

    print(file=output_file)
    print("RMSE mean %f std %f " % (np.mean(RMSEs), np.std(RMSEs)), file=output_file)
    print("R2 mean %f std %f " % (np.mean(R2s), np.std(R2s)), file=output_file)

    print(file=output_file)
    print("Regressor training seconds: %f" % (np.sum(train_durations)), file=output_file, flush=True)
    print("Regressor testing seconds: %f" % (np.sum(test_durations)), file=output_file, flush=True)
    # print("Genetic algorithm seconds: %f" %(genetic_duration), file=output_file, flush=True)
    print(file=output_file, flush=True)

