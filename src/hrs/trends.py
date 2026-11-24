"""Generalised least-squares trends in the three standardised risks, with Rubin pooling."""

import numpy as np
import pandas as pd
from scipy.stats import t


def pooled_trends(draws):
    design = np.column_stack([np.ones(3), np.arange(3)])
    rows = []
    for outcome, subset in draws.groupby("endpoint"):
        risks = subset.pivot(index=["imp", "bootstrap"], columns="group", values="risk").mul(100)
        if risks.shape != (510, 3) or list(risks.columns) != [0, 1, 2] or risks.isna().any().any():
            raise ValueError(f"Incomplete risk estimates for {outcome}")
        points = risks.xs(0, level="bootstrap")
        covariances = []
        for _, batch in risks.groupby(level="imp", sort=True):
            boot = batch.droplevel("imp").drop(0)
            if list(boot.index) != list(range(1, 51)):
                raise ValueError("Expected 50 bootstrap resamples in each imputation")
            covariances.append(boot.cov().to_numpy())
        nimp = len(points)
        total_cov = np.mean(covariances, axis=0) + (1 + 1 / nimp) * points.cov().to_numpy()
        weighted_design = np.linalg.solve(total_cov, design).T
        slope_contrast = np.linalg.solve(weighted_design @ design, weighted_design)[1]
        np.testing.assert_allclose(slope_contrast @ design, [0, 1], atol=1e-10)
        slopes = points.to_numpy() @ slope_contrast
        within = np.mean([slope_contrast @ matrix @ slope_contrast for matrix in covariances])
        between = (1 + 1 / nimp) * slopes.var(ddof=1)
        se = np.sqrt(within + between)
        df = (nimp - 1) * (1 + within / between) ** 2 if between > 0 else np.inf
        estimate = slopes.mean()
        margin = t.ppf(0.975, df) * se
        rows.append(
            dict(
                endpoint=outcome,
                trend_pp=estimate,
                lo=estimate - margin,
                hi=estimate + margin,
                p_trend=2 * t.sf(abs(estimate) / se, df),
                se=se,
                df=df,
            )
        )
    return pd.DataFrame(rows)
