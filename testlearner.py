

import numpy as np
import sys
import RTLearner as rtl
import DTLearner as dtl
import BagLearner as bl
import InsaneLearner as it
import matplotlib.pyplot as plt
import time

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def evaluate_models(train_x, train_y, test_x, test_y):
    # Initialize learners
    dt_learner = dtl.DTLearner(leaf_size=5)
    rt_learner = rtl.RTLearner(leaf_size=5)

    start_time = time.time()
    dt_learner.add_evidence(train_x, train_y)
    dt_training_time = time.time() - start_time

    start_time = time.time()
    rt_learner.add_evidence(train_x, train_y)
    rt_training_time = time.time() - start_time

    dt_predictions_train = dt_learner.query(train_x)
    dt_predictions_test = dt_learner.query(test_x)

    dt_mae_train = mean_absolute_error(train_y, dt_predictions_train)
    dt_r2_train = r_squared(train_y, dt_predictions_train)

    dt_mae_test = mean_absolute_error(test_y, dt_predictions_test)
    dt_r2_test = r_squared(test_y, dt_predictions_test)

    rt_predictions_train = rt_learner.query(train_x)
    rt_predictions_test = rt_learner.query(test_x)

    rt_mae_train = mean_absolute_error(train_y, rt_predictions_train)
    rt_r2_train = r_squared(train_y, rt_predictions_train)

    rt_mae_test = mean_absolute_error(test_y, rt_predictions_test)
    rt_r2_test = r_squared(test_y, rt_predictions_test)

    return {
        "DT": {
            "train_mae": dt_mae_train,
            "test_mae": dt_mae_test,
            "train_r2": dt_r2_train,
            "test_r2": dt_r2_test,
            "training_time": dt_training_time
        },
        "RT": {
            "train_mae": rt_mae_train,
            "test_mae": rt_mae_test,
            "train_r2": rt_r2_train,
            "test_r2": rt_r2_test,
            "training_time": rt_training_time
        }
    }


def plot_comparison(results):
    labels = ["DTLearner", "RTLearner"]

    train_mae = [results["DT"]["train_mae"], results["RT"]["train_mae"]]
    test_mae = [results["DT"]["test_mae"], results["RT"]["test_mae"]]
    train_r2 = [results["DT"]["train_r2"], results["RT"]["train_r2"]]
    test_r2 = [results["DT"]["test_r2"], results["RT"]["test_r2"]]

    # Plot MAE
    plt.figure(figsize=(12, 6))
    plt.bar(labels, train_mae, color='b', alpha=0.7, label='Train MAE')
    plt.bar(labels, test_mae, color='r', alpha=0.7, label='Test MAE', bottom=train_mae)
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error Comparison')
    plt.legend()
    plt.savefig('images/exp31')

    # Plot R-Squared
    plt.figure(figsize=(12, 6))
    plt.bar(labels, train_r2, color='b', alpha=0.7, label='Train R-Squared')
    plt.bar(labels, test_r2, color='r', alpha=0.7, label='Test R-Squared', bottom=train_r2)
    plt.ylabel('R-Squared')
    plt.title('R-Squared Comparison')
    plt.legend()
    plt.savefig('images/exp32')

def evaluate_rmse(learner, train_x, train_y, test_x, test_y):
    pred_train_y = learner.query(train_x)
    pred_test_y = learner.query(test_x)

    in_sample_rmse = np.sqrt(((train_y - pred_train_y) ** 2).sum() / train_y.shape[0])
    out_sample_rmse = np.sqrt(((test_y - pred_test_y) ** 2).sum() / test_y.shape[0])

    return in_sample_rmse, out_sample_rmse

def experiment(data_x, data_y, bags=20, leaf_sizes=np.arange(1, 21), verbose=False):
    train_size = int(0.6 * data_x.shape[0])
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_y = data_y[train_size:]

    in_sample_rmses_baseline = []
    out_sample_rmses_baseline = []
    in_sample_rmses_bagging = []
    out_sample_rmses_bagging = []

    for leaf_size in leaf_sizes:
        learner = dtl.DTLearner(leaf_size=leaf_size, verbose=verbose)
        learner.add_evidence(train_x, train_y)
        in_rmse, out_rmse = evaluate_rmse(learner, train_x, train_y, test_x, test_y)
        in_sample_rmses_baseline.append(in_rmse)
        out_sample_rmses_baseline.append(out_rmse)

        bag_learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, verbose=verbose)
        bag_learner.add_evidence(train_x, train_y)
        in_rmse_bag, out_rmse_bag = evaluate_rmse(bag_learner, train_x, train_y, test_x, test_y)
        in_sample_rmses_bagging.append(in_rmse_bag)
        out_sample_rmses_bagging.append(out_rmse_bag)

    plt.figure(figsize=(12, 6))
    plt.plot(leaf_sizes, in_sample_rmses_baseline, marker='o', label='In-Sample RMSE (No Bagging)')
    plt.plot(leaf_sizes, out_sample_rmses_baseline, marker='x', label='Out-Sample RMSE (No Bagging)')
    plt.plot(leaf_sizes, in_sample_rmses_bagging, marker='o', linestyle='--', label='In-Sample RMSE (With Bagging)')
    plt.plot(leaf_sizes, out_sample_rmses_bagging, marker='x', linestyle='--', label='Out-Sample RMSE (With Bagging)')

    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Leaf Size (With and Without Bagging)')
    plt.legend()
    plt.grid(True)
    plt.savefig("images/exp2")


def clean_data(X, Y):
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    Y_clean = Y[mask]
    return X_clean, Y_clean

def plot_rmse_vs_leaf_size(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 21)
    in_sample_rmses = []
    out_of_sample_rmses = []

    for size in leaf_sizes:
        temp_learner = dtl.DTLearner(leaf_size=size)
        temp_learner.add_evidence(train_x, train_y)

        pred_train_y = temp_learner.query(train_x)
        in_sample_rmse = np.sqrt(((train_y - pred_train_y) ** 2).sum() / train_y.shape[0])
        in_sample_rmses.append(in_sample_rmse)

        pred_test_y = temp_learner.query(test_x)
        out_of_sample_rmse = np.sqrt(((test_y - pred_test_y) ** 2).sum() / test_y.shape[0])
        out_of_sample_rmses.append(out_of_sample_rmse)

    plt.figure(figsize=(12, 6))
    plt.plot(leaf_sizes, in_sample_rmses, marker='o', label='In-Sample RMSE')
    plt.plot(leaf_sizes, out_of_sample_rmses, marker='x', label='Out-of-Sample RMSE')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/exp1')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    train_x = data[:train_rows, 1:-1]  # Skip date column
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 1:-1]  # Skip date column
    test_y = data[train_rows:, -1]

    train_x, train_y = clean_data(train_x, train_y)

    learner = dtl.DTLearner(leaf_size=1, verbose=True)
    learner.add_evidence(train_x, train_y)

    pred_y = learner.query(train_x)
    rmse = np.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])


    pred_y = learner.query(test_x)
    rmse = np.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    if learner.verbose:
        plot_rmse_vs_leaf_size(train_x, train_y, test_x, test_y)

    learner = rtl.RTLearner(leaf_size=1, verbose=True)
    learner.add_evidence(train_x, train_y)


    pred_y = learner.query(train_x)
    rmse = np.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])


    pred_y = learner.query(test_x)
    rmse = np.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])


    learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    learner.add_evidence(train_x, train_y)
    pred_y = learner.query(train_x)

    learner = it.InsaneLearner(verbose=False)
    learner.add_evidence(train_x, train_y)
    pred_y = learner.query(train_x)
    data_x = data[:, 1:-1]  # Skip the first column (date) and the last column (target variable)
    data_y = data[:, -1]
    experiment(data_x, data_y, bags=20, verbose=True)
    results = evaluate_models(train_x, train_y, test_x, test_y)
    plot_comparison(results)

