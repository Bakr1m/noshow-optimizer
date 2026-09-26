"""SHAP smoke test: TreeExplainer output geometry on a tiny model."""
import lightgbm as lgb
import numpy as np
import shap


def test_tree_explainer_output_shape():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 5)
    y = (X[:, 0] + rng.rand(40) * 0.5 > 0.7).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, verbose=-1)
    clf.fit(X, y)
    values = shap.TreeExplainer(clf).shap_values(X[:10])
    values = np.asarray(values)
    assert values.shape == (10, 5)
    assert np.isfinite(values).all()
    assert (np.abs(values).mean(axis=0) > 0).any()  # something drives output
