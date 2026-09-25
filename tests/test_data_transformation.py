import os

import numpy as np

from src.components.data_transformation import DataTransformation
from tests.conftest import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN


def to_dense(array):
    return array.toarray() if hasattr(array, "toarray") else array


def expected_n_features(df):
    # numerical columns are kept as is, each category becomes one one-hot column
    return len(NUMERICAL_COLUMNS) + sum(df[col].nunique() for col in CATEGORICAL_COLUMNS)


def test_preprocessor_output_shape(raw_df):
    X = raw_df.drop(columns=[TARGET_COLUMN])

    output = DataTransformation().get_data_transformer_object().fit_transform(X)

    assert output.shape == (len(X), expected_n_features(raw_df))


def test_preprocessor_imputes_missing_values(raw_df):
    X = raw_df.drop(columns=[TARGET_COLUMN])
    X.loc[0, "reading_score"] = np.nan
    X.loc[1, "gender"] = np.nan

    output = to_dense(DataTransformation().get_data_transformer_object().fit_transform(X))

    assert not np.isnan(output).any()


def test_initiate_data_transformation(raw_df, tmp_path):
    train_df, test_df = raw_df.iloc[:800], raw_df.iloc[800:]
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    transformation = DataTransformation()
    # never overwrite the real artifacts/preprocessor.pkl from a test
    transformation.data_transformation_config.preprocessor_obj_file_path = str(tmp_path / "preprocessor.pkl")

    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        str(train_path), str(test_path)
    )

    # features + target as last column
    assert train_arr.shape == (800, expected_n_features(raw_df) + 1)
    assert test_arr.shape == (200, expected_n_features(raw_df) + 1)
    np.testing.assert_array_equal(train_arr[:, -1], train_df[TARGET_COLUMN].to_numpy())
    np.testing.assert_array_equal(test_arr[:, -1], test_df[TARGET_COLUMN].to_numpy())
    assert os.path.exists(preprocessor_path)
