
import argparse
import json
import logging
import os
import pathlib
import pickle as pkl
import tarfile


import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO)

TRAIN_VALIDATION_FRACTION = 0.2
RANDOM_STATE_SAMPLING = 200

logging.basicConfig(level=logging.INFO)


def prepare_data(train_dir, validation_dir):
    """Read data from train and validation channel, and return predicting features and target variables.

    Args:
        data_dir (str): directory which saves the training data.

    Returns:
        Tuple of training features, training target, validation features, validation target.
    """
    df_train = pd.read_csv(
        os.path.join(train_dir, "train.csv"),
        header=None,
    )
    df_train = df_train.iloc[np.random.permutation(len(df_train))]
    df_train.columns = ["target"] + [f"feature_{x}" for x in range(df_train.shape[1] - 1)]

    try:
        df_validation = pd.read_csv(
            os.path.join(validation_dir, "validation.csv"),
            header=None,
        )
        df_validation.columns = ["target"] + [
            f"feature_{x}" for x in range(df_validation.shape[1] - 1)
        ]

    except FileNotFoundError:  # when validation data is not available in the directory
        logging.info(
            f"Validation data is not found. {TRAIN_VALIDATION_FRACTION * 100}% of training data is "
            f"randomly selected as validation data. The seed for random sampling is {RANDOM_STATE_SAMPLING}."
        )
        df_validation = df_train.sample(
            frac=TRAIN_VALIDATION_FRACTION,
            random_state=RANDOM_STATE_SAMPLING,
        )
        df_train.drop(df_validation.index, inplace=True)
        df_validation.reset_index(drop=True, inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, :1]
    X_val, y_val = df_validation.iloc[:, 1:], df_validation.iloc[:, :1]

    return X_train.values, y_train.values, X_val.values, y_val.values


def main():
    """Run training."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_depth",
        type=int,
    )
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--verbosity", type=int)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument("--learning_rate", type=str, default="auto")
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--sm_hosts", type=str, default=os.environ.get("SM_HOSTS"))
    parser.add_argument("--sm_current_host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    args, _ = parser.parse_known_args()

    X_train, y_train, X_val, y_val = prepare_data(args.train, args.validation)

    # create dataset for lightgbm
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "validation")]

    # specify your configurations as a dict
    params = {
        "booster": "gbtree",
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": 1,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "eval_metric": "rmse",
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        xgb_model=None,
    )

    model_location = args.model_dir + "/xgboost-model"
    pkl.dump(bst, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    main()
