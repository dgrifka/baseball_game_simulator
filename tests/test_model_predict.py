"""
Smoke test proving the shipped pickle actually predicts.

Nothing in the suite called ``predict_proba`` before this, so a pickle that
unpickles but whose ColumnTransformer no longer matches
``create_features_for_prediction`` would pass CI and fail in production. Loads
the artifact exactly as ``Simulator/game_simulator.py`` does.
"""
import json
import os

import joblib
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_REPO_ROOT, "Model", "batted_ball_model.pkl")
_METADATA_PATH = os.path.join(_REPO_ROOT, "Model", "model_metadata.json")


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(_MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    with open(_METADATA_PATH) as fh:
        return json.load(fh)


def test_predict_proba_shape_and_normalization(pipeline, golden_frame):
    proba = pipeline.predict_proba(golden_frame)
    assert proba.shape == (1, 5)
    assert proba[0].sum() == pytest.approx(1.0, abs=1e-9)
    assert (proba >= 0).all()


def test_classes_match_metadata(pipeline, metadata):
    """Class order is the contract: downstream code indexes proba by position.

    ``model_metadata.json`` stores ``classes`` as an index -> label map (the
    pipeline itself is fit on the integer labels), so the keys are what must
    line up with ``classes_``, and their order fixes which column of
    ``predict_proba`` means "home_run".
    """
    expected_labels = sorted(metadata["classes"], key=int)
    assert [str(c) for c in pipeline.classes_] == expected_labels
    assert [metadata["classes"][k] for k in expected_labels] == [
        "out",
        "single",
        "double",
        "triple",
        "home_run",
    ]


def test_metadata_feature_count_matches_pipeline_input(metadata, golden_frame):
    """model_metadata.json is the ground truth for what shipped; keep it honest."""
    features = metadata["features"]
    n_features = len(features["numeric"]) + len(features["categorical"])
    assert n_features == 28
    emitted = set(golden_frame.columns)
    missing = [
        name
        for name in features["numeric"] + features["categorical"]
        if name not in emitted
    ]
    assert missing == []
