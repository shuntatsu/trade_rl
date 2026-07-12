from mars_lite.pipeline.gates import evaluate_residual_alpha_gate


def test_selected_model_gate_requires_ic_stability_and_positive_folds() -> None:
    report = {
        "model": "gbm",
        "mean_oos_ic": 0.025,
        "positive_fold_ratio": 0.8,
        "t_stat": 1.5,
        "n_folds": 5,
    }

    result = evaluate_residual_alpha_gate(report)

    assert result["passed"] is True
    assert result["model"] == "gbm"


def test_mean_ic_alone_is_not_enough() -> None:
    result = evaluate_residual_alpha_gate(
        {
            "model": "gbm",
            "mean_oos_ic": 0.03,
            "positive_fold_ratio": 0.4,
            "t_stat": 0.5,
            "n_folds": 5,
        }
    )

    assert result["passed"] is False
    assert result["checks"]["positive_fold_ratio"] is False
    assert result["checks"]["stability"] is False


def test_ridge_and_gbm_reports_remain_distinct() -> None:
    ridge = evaluate_residual_alpha_gate(
        {
            "model": "ridge",
            "mean_oos_ic": 0.015,
            "positive_fold_ratio": 0.8,
            "t_stat": 1.2,
            "n_folds": 5,
        }
    )
    gbm = evaluate_residual_alpha_gate(
        {
            "model": "gbm",
            "mean_oos_ic": 0.022,
            "positive_fold_ratio": 0.8,
            "t_stat": 1.2,
            "n_folds": 5,
        }
    )

    assert ridge["passed"] is False
    assert gbm["passed"] is True
