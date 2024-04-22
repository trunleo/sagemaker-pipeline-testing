
import json
import os
import pickle as pkl

import numpy as np
import pandas as pd
import sagemaker_xgboost_container.encoder as xgb_encoders
import xgboost as xgb
import io
import logging

logging.basicConfig(level=logging.INFO)


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster


def transform_fn(model, request_body, request_content_type, accept):
    """ """
    if request_content_type == "text/libsvm":
        input_data = xgb_encoders.libsvm_to_dmatrix(request_body)
    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body.strip("\n")), header=None)
        df.drop(0, axis=1, inplace=True)
        input_data = xgb.DMatrix(data=df)

    else:
        raise ValueError("Content type {} is not supported.".format(request_content_type))

    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))

    logging.info("Successfully completed transform job!")

    return ",".join(str(x) for x in output[0])
