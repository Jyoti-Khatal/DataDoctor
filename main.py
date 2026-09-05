#importing the required libraries

import base64
import math
from pathlib import Path
import io

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="DataDoctor API",
    version="1.0.0"
)

def make_json_safe(obj):

    # Dictionary
    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    # Lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return [
            make_json_safe(value)
            for value in obj.tolist()
        ]

    # NumPy scalar types
    # np.bool_, np.int64, np.float64, etc.
    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())

    # Pandas Series
    if isinstance(obj, pd.Series):
        return [
            make_json_safe(value)
            for value in obj.tolist()
        ]

    # Pandas Index
    if isinstance(obj, pd.Index):
        return [
            make_json_safe(value)
            for value in obj.tolist()
        ]

    # Pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # Pandas missing values
    if obj is pd.NA or obj is pd.NaT:
        return None

    # Python float NaN / infinity
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj


def profile_dataset(
    df,
    target_column=None,
    problem_type=None
):
    """
    DataDoctor Dataset Profiler

    Profiles a Pandas DataFrame and returns deterministic
    dataset-quality and ML-readiness evidence.

    Main responsibilities:
    - Dataset-level statistics
    - Column-level statistics
    - Missing values
    - Duplicates
    - Constant / near-constant columns
    - Possible identifier detection
    - Numeric values stored as text
    - Possible date columns
    - Numeric outliers
    - Skewness
    - High correlations
    - Target analysis
    - ML context

    Important:
    Statistics are calculated by Python.
    The AI layer should only interpret this output.
    """

    import pandas as pd
    import numpy as np

    # ==========================================================
    # 1. BASIC VALIDATION
    # ==========================================================

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    # Clean target value
    if target_column is not None:
        target_column = str(target_column).strip()

        if target_column == "":
            target_column = None

    # Clean problem type
    if problem_type is not None:
        problem_type = str(problem_type).strip().lower()

        if problem_type == "":
            problem_type = None

    # Validate target
    if (
        target_column is not None
        and target_column not in df.columns
    ):
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    # ==========================================================
    # 2. DETERMINE ML CONTEXT
    # ==========================================================

    if problem_type is None:

        if target_column is None:
            detected_problem_type = (
                "unknown_unsupervised_possible"
            )

        else:
            target_series = df[target_column]

            if pd.api.types.is_numeric_dtype(
                target_series
            ):
                detected_problem_type = "regression"

            else:
                detected_problem_type = "classification"

    else:

        detected_problem_type = problem_type

    # ==========================================================
    # 3. DATASET-LEVEL INFORMATION
    # ==========================================================

    rows = len(df)

    column_count = len(df.columns)

    total_cells = rows * column_count

    missing_cells = int(
        df.isna().sum().sum()
    )

    if total_cells > 0:

        overall_missing_percentage = round(
            (
                missing_cells
                / total_cells
            ) * 100,
            2
        )

    else:

        overall_missing_percentage = 0.0

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if rows > 0:

        duplicate_percentage = round(
            (
                duplicate_rows
                / rows
            ) * 100,
            2
        )

    else:

        duplicate_percentage = 0.0

    # ==========================================================
    # 4. STORAGE FOR RESULTS
    # ==========================================================

    columns_profile = {}

    constant_columns = []

    near_constant_columns = []

    possible_identifiers = []

    possible_numeric_columns = []

    possible_date_columns = []

    numeric_columns = []

    categorical_columns = []

    # ==========================================================
    # 5. IDENTIFIER NAME HELPER
    # ==========================================================

    def name_looks_like_identifier(column_name):
        """
        Detect common identifier naming patterns.

        Examples:
        id
        customer_id
        customerId
        transaction_id
        user_key
        uuid
        guid
        """

        name = str(column_name).strip().lower()

        normalized = (
            name
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        # Exact identifier names
        exact_identifier_names = {
            "id",
            "identifier",
            "uuid",
            "guid",
            "key"
        }

        if name in exact_identifier_names:
            return True

        # Common snake-case / separated forms
        if (
            name.endswith("_id")
            or name.startswith("id_")
            or name.endswith("_key")
            or name.startswith("key_")
        ):
            return True

        # customerId / transactionId / recordId
        if normalized.endswith("id"):
            return True

        if normalized.endswith("uuid"):
            return True

        if normalized.endswith("guid"):
            return True

        return False

    # ==========================================================
    # 6. PROFILE EACH COLUMN
    # ==========================================================

    for column in df.columns:

        series = df[column]

        non_missing = series.dropna()

        non_missing_count = len(
            non_missing
        )

        missing_count = int(
            series.isna().sum()
        )

        if rows > 0:

            missing_percentage = round(
                (
                    missing_count
                    / rows
                ) * 100,
                2
            )

        else:

            missing_percentage = 0.0

        unique_count = int(
            non_missing.nunique()
        )

        if non_missing_count > 0:

            unique_percentage = round(
                (
                    unique_count
                    / non_missing_count
                ) * 100,
                2
            )

        else:

            unique_percentage = 0.0

        # ------------------------------------------------------
        # BASIC COLUMN PROFILE
        # ------------------------------------------------------

        column_info = {
            "dtype": str(series.dtype),

            "missing_count": missing_count,

            "missing_percentage": (
                missing_percentage
            ),

            "non_missing_count": (
                non_missing_count
            ),

            "unique_count": unique_count,

            "unique_percentage": (
                unique_percentage
            )
        }

        # ======================================================
        # 7. CONSTANT COLUMN
        # ======================================================

        if (
            non_missing_count > 0
            and unique_count <= 1
        ):

            constant_columns.append(
                column
            )

            column_info[
                "constant"
            ] = True

        else:

            column_info[
                "constant"
            ] = False

        # ======================================================
        # 8. NEAR-CONSTANT COLUMN
        # ======================================================

        near_constant = False

        dominant_value_percentage = None

        if (
            non_missing_count > 0
            and unique_count > 1
        ):

            value_counts = (
                non_missing
                .value_counts(
                    normalize=True
                )
            )

            if not value_counts.empty:

                dominant_value_percentage = round(
                    float(
                        value_counts.iloc[0]
                        * 100
                    ),
                    2
                )

                if (
                    dominant_value_percentage
                    >= 99
                ):

                    near_constant = True

                    near_constant_columns.append(
                        column
                    )

        column_info[
            "near_constant"
        ] = near_constant

        column_info[
            "dominant_value_percentage"
        ] = dominant_value_percentage

        # ======================================================
        # 9. NUMERIC COLUMN
        # ======================================================

        if pd.api.types.is_numeric_dtype(
            series
        ):

            numeric_columns.append(
                column
            )

            column_info[
                "column_type"
            ] = "numeric"

            numeric_series = pd.to_numeric(
                non_missing,
                errors="coerce"
            ).dropna()

            if len(numeric_series) > 0:

                # ----------------------------------------------
                # DESCRIPTIVE STATISTICS
                # ----------------------------------------------

                column_info[
                    "mean"
                ] = round(
                    float(
                        numeric_series.mean()
                    ),
                    3
                )

                column_info[
                    "median"
                ] = round(
                    float(
                        numeric_series.median()
                    ),
                    3
                )

                std_value = (
                    numeric_series.std()
                )

                if pd.notna(std_value):

                    column_info[
                        "std"
                    ] = round(
                        float(std_value),
                        3
                    )

                else:

                    column_info[
                        "std"
                    ] = None

                column_info[
                    "min"
                ] = float(
                    numeric_series.min()
                )

                column_info[
                    "max"
                ] = float(
                    numeric_series.max()
                )

                # ----------------------------------------------
                # QUARTILES
                # ----------------------------------------------

                q1 = float(
                    numeric_series.quantile(
                        0.25
                    )
                )

                q3 = float(
                    numeric_series.quantile(
                        0.75
                    )
                )

                iqr = q3 - q1

                column_info[
                    "q1"
                ] = round(
                    q1,
                    3
                )

                column_info[
                    "q3"
                ] = round(
                    q3,
                    3
                )

                column_info[
                    "iqr"
                ] = round(
                    iqr,
                    3
                )

                # ----------------------------------------------
                # OUTLIER DETECTION USING IQR
                # ----------------------------------------------

                if iqr > 0:

                    lower_bound = (
                        q1
                        - 1.5 * iqr
                    )

                    upper_bound = (
                        q3
                        + 1.5 * iqr
                    )

                    outlier_mask = (
                        (
                            numeric_series
                            < lower_bound
                        )
                        |
                        (
                            numeric_series
                            > upper_bound
                        )
                    )

                    outlier_count = int(
                        outlier_mask.sum()
                    )

                else:

                    lower_bound = None

                    upper_bound = None

                    outlier_count = 0

                if len(numeric_series) > 0:

                    outlier_percentage = round(
                        (
                            outlier_count
                            / len(numeric_series)
                        ) * 100,
                        2
                    )

                else:

                    outlier_percentage = 0.0

                column_info[
                    "outlier_count"
                ] = outlier_count

                column_info[
                    "outlier_percentage"
                ] = outlier_percentage

                if lower_bound is not None:

                    column_info[
                        "outlier_lower_bound"
                    ] = round(
                        float(lower_bound),
                        3
                    )

                    column_info[
                        "outlier_upper_bound"
                    ] = round(
                        float(upper_bound),
                        3
                    )

                else:

                    column_info[
                        "outlier_lower_bound"
                    ] = None

                    column_info[
                        "outlier_upper_bound"
                    ] = None

                # ----------------------------------------------
                # SKEWNESS
                # ----------------------------------------------

                if (
                    len(numeric_series) >= 3
                    and numeric_series.nunique() > 1
                ):

                    skewness = (
                        numeric_series.skew()
                    )

                    if pd.notna(skewness):

                        skewness = round(
                            float(skewness),
                            3
                        )

                    else:

                        skewness = None

                else:

                    skewness = None

                column_info[
                    "skewness"
                ] = skewness

                column_info[
                    "highly_skewed"
                ] = (
                    skewness is not None
                    and abs(skewness) >= 2
                )

            else:

                column_info[
                    "min"
                ] = None

                column_info[
                    "max"
                ] = None

                column_info[
                    "outlier_count"
                ] = 0

                column_info[
                    "outlier_percentage"
                ] = 0.0

                column_info[
                    "skewness"
                ] = None

                column_info[
                    "highly_skewed"
                ] = False

        # ======================================================
        # 10. NON-NUMERIC COLUMN
        # ======================================================

        else:

            categorical_columns.append(
                column
            )

            column_info[
                "column_type"
            ] = "categorical"

            # --------------------------------------------------
            # NUMERIC-LIKE TEXT DETECTION
            # --------------------------------------------------

            if non_missing_count > 0:

                converted_numeric = pd.to_numeric(
                    non_missing,
                    errors="coerce"
                )

                numeric_like_count = int(
                    converted_numeric
                    .notna()
                    .sum()
                )

                numeric_like_percentage = round(
                    (
                        numeric_like_count
                        / non_missing_count
                    ) * 100,
                    2
                )

            else:

                numeric_like_percentage = 0.0

            column_info[
                "numeric_like_percentage"
            ] = numeric_like_percentage

            # Avoid target auto conversion
            if column != target_column:

                if (
                    numeric_like_percentage
                    >= 80
                ):

                    possible_numeric_columns.append(
                        column
                    )

            # --------------------------------------------------
            # POSSIBLE DATE DETECTION
            # --------------------------------------------------

            date_like_percentage = 0.0

            # Do not run expensive date parsing when almost all
            # values already look numeric.
            if (
                non_missing_count > 0
                and numeric_like_percentage < 80
            ):

                try:

                    parsed_dates = pd.to_datetime(
                        non_missing.astype(str),
                        format="mixed",
                        errors="coerce"
                    )

                    date_like_count = int(
                        parsed_dates
                        .notna()
                        .sum()
                    )

                    date_like_percentage = round(
                        (
                            date_like_count
                            / non_missing_count
                        ) * 100,
                        2
                    )

                except Exception:

                    date_like_percentage = 0.0

            column_info[
                "date_like_percentage"
            ] = date_like_percentage

            if (
                date_like_percentage
                >= 80
            ):

                possible_date_columns.append(
                    column
                )

        # ======================================================
        # 11. IMPROVED IDENTIFIER DETECTION
        # ======================================================

        #
        # IMPORTANT CHANGE:
        #
        # High uniqueness alone does NOT make a column an ID.
        #
        # account_balance = unique values, but measurement
        # customer_id     = unique values + ID-like name
        #

        if column != target_column:

            name_identifier = (
                name_looks_like_identifier(
                    column
                )
            )

            if (
                non_missing_count > 0
            ):

                uniqueness_ratio = (
                    unique_count
                    / non_missing_count
                )

            else:

                uniqueness_ratio = 0.0

            # ----------------------------------------------
            # TEXT IDENTIFIERS
            # ----------------------------------------------

            if (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype
                )
            ):

                if (
                    name_identifier
                    and uniqueness_ratio >= 0.95
                ):

                    possible_identifiers.append(
                        column
                    )

            # ----------------------------------------------
            # NUMERIC IDENTIFIERS
            # ----------------------------------------------

            elif pd.api.types.is_numeric_dtype(
                series
            ):

                #
                # Numeric measurements are not identifiers
                # just because values are unique.
                #
                # We require:
                #
                # 1. ID-like column name
                # 2. >=95% uniqueness
                # 3. Integer-like values
                #

                if (
                    name_identifier
                    and uniqueness_ratio >= 0.95
                ):

                    numeric_values = pd.to_numeric(
                        non_missing,
                        errors="coerce"
                    ).dropna()

                    if len(numeric_values) > 0:

                        integer_like = bool(
                            np.all(
                                np.isclose(
                                    numeric_values,
                                    np.round(
                                        numeric_values
                                    )
                                )
                            )
                        )

                        if integer_like:

                            possible_identifiers.append(
                                column
                            )

        # ======================================================
        # STORE COLUMN PROFILE
        # ======================================================

        columns_profile[
            column
        ] = column_info

    # ==========================================================
    # 12. HIGH CORRELATION DETECTION
    # ==========================================================

    high_correlation_pairs = []

    usable_numeric_columns = [
        column
        for column in numeric_columns
        if column != target_column
    ]

    if len(usable_numeric_columns) >= 2:

        numeric_df = df[
            usable_numeric_columns
        ].select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] >= 2:

            correlation_matrix = (
                numeric_df.corr()
            )

            correlation_columns = (
                correlation_matrix.columns
                .tolist()
            )

            for i in range(
                len(correlation_columns)
            ):

                for j in range(
                    i + 1,
                    len(correlation_columns)
                ):

                    column_1 = (
                        correlation_columns[i]
                    )

                    column_2 = (
                        correlation_columns[j]
                    )

                    correlation = (
                        correlation_matrix
                        .loc[
                            column_1,
                            column_2
                        ]
                    )

                    if pd.isna(
                        correlation
                    ):
                        continue

                    correlation_value = float(
                        correlation
                    )

                    if (
                        abs(
                            correlation_value
                        )
                        >= 0.90
                    ):

                        high_correlation_pairs.append({
                            "column_1": column_1,

                            "column_2": column_2,

                            "correlation": round(
                                correlation_value,
                                3
                            )
                        })

    # ==========================================================
    # 13. TARGET PROFILE
    # ==========================================================

    target_profile = {
        "provided": (
            target_column is not None
        ),

        "column": target_column
    }

    if target_column is not None:

        target_series = df[
            target_column
        ]

        target_non_missing = (
            target_series.dropna()
        )

        target_missing_count = int(
            target_series
            .isna()
            .sum()
        )

        target_missing_percentage = round(
            (
                target_missing_count
                / rows
            ) * 100,
            2
        )

        target_profile.update({
            "dtype": str(
                target_series.dtype
            ),

            "missing_count": (
                target_missing_count
            ),

            "missing_percentage": (
                target_missing_percentage
            ),

            "unique_count": int(
                target_non_missing.nunique()
            )
        })

        # ======================================================
        # CLASSIFICATION TARGET
        # ======================================================

        if (
            detected_problem_type
            == "classification"
        ):

            class_counts = (
                target_non_missing
                .value_counts()
            )

            target_count = int(
                len(target_non_missing)
            )

            class_distribution = {}

            for class_value, count in (
                class_counts.items()
            ):

                if target_count > 0:

                    percentage = round(
                        (
                            int(count)
                            / target_count
                        ) * 100,
                        2
                    )

                else:

                    percentage = 0.0

                class_distribution[
                    str(class_value)
                ] = {
                    "count": int(count),
                    "percentage": percentage
                }

            if (
                len(class_counts) > 0
                and target_count > 0
            ):

                largest_class_count = int(
                    class_counts.iloc[0]
                )

                largest_class_percentage = round(
                    (
                        largest_class_count
                        / target_count
                    ) * 100,
                    2
                )

            else:

                largest_class_percentage = 0.0

            #
            # Current DataDoctor imbalance threshold.
            #
            # Telco Churn at 73.46% will therefore
            # continue to trigger the warning.
            #

            imbalanced = (
                largest_class_percentage
                >= 70
            )

            target_profile.update({
                "class_distribution": (
                    class_distribution
                ),

                "largest_class_percentage": (
                    largest_class_percentage
                ),

                "imbalanced": (
                    imbalanced
                )
            })

        # ======================================================
        # REGRESSION TARGET
        # ======================================================

        elif (
            detected_problem_type
            == "regression"
        ):

            numeric_target = pd.to_numeric(
                target_non_missing,
                errors="coerce"
            ).dropna()

            if len(numeric_target) > 0:

                target_profile.update({
                    "mean": round(
                        float(
                            numeric_target.mean()
                        ),
                        3
                    ),

                    "median": round(
                        float(
                            numeric_target.median()
                        ),
                        3
                    ),

                    "min": float(
                        numeric_target.min()
                    ),

                    "max": float(
                        numeric_target.max()
                    )
                })

            target_profile[
                "imbalanced"
            ] = False

    else:

        target_profile.update({
            "imbalanced": False
        })

    # ==========================================================
    # 14. ML FEATURE COUNTS
    # ==========================================================

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    numeric_feature_count = sum(
        1
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    )

    categorical_feature_count = (
        len(feature_columns)
        - numeric_feature_count
    )

    ml_context = {
        "problem_type": (
            detected_problem_type
        ),

        "target_column": (
            target_column
        ),

        "feature_count": (
            len(feature_columns)
        ),

        "numeric_feature_count": (
            numeric_feature_count
        ),

        "categorical_feature_count": (
            categorical_feature_count
        )
    }

    # ==========================================================
    # 15. FINAL PROFILE
    # ==========================================================

    profile = {

        "dataset": {
            "rows": int(rows),

            "columns": int(
                column_count
            ),

            "total_cells": int(
                total_cells
            ),

            "missing_cells": int(
                missing_cells
            ),

            "overall_missing_percentage": (
                overall_missing_percentage
            ),

            "duplicate_rows": (
                duplicate_rows
            ),

            "duplicate_percentage": (
                duplicate_percentage
            )
        },

        "ml_context": (
            ml_context
        ),

        "target": (
            target_profile
        ),

        "columns": (
            columns_profile
        ),

        "possible_identifiers": list(
            dict.fromkeys(
                possible_identifiers
            )
        ),

        "constant_columns": list(
            dict.fromkeys(
                constant_columns
            )
        ),

        "near_constant_columns": list(
            dict.fromkeys(
                near_constant_columns
            )
        ),

        "possible_numeric_columns": list(
            dict.fromkeys(
                possible_numeric_columns
            )
        ),

        "possible_date_columns": list(
            dict.fromkeys(
                possible_date_columns
            )
        ),

        "high_correlation_pairs": (
            high_correlation_pairs
        )
    }

    return profile


def generate_issues(profile):
    """
    DataDoctor Severity Engine

    Converts profiler findings into prioritized,
    explainable data-quality and ML-readiness issues.

    Repeated column-level findings are grouped into
    dataset-level issues to avoid excessive issue counts.

    IMPORTANT:
    Possible numeric datatype problems remain column-level
    because the safe-fix engine may need the exact column.

    Severity:
        HIGH   -> potentially significant ML/data-quality problem
        MEDIUM -> should generally be reviewed
        LOW    -> improvement or monitoring recommendation
    """

    issues = []

    # ==========================================================
    # HELPER FUNCTIONS
    # ==========================================================

    def add_issue(
        severity,
        category,
        issue,
        evidence,
        impact,
        recommendation,
        column=None,
        affected_columns=None
    ):
        result = {
            "severity": severity,
            "category": category,
            "issue": issue,
            "column": column,
            "evidence": evidence,
            "why_it_matters": impact,
            "recommended_action": recommendation
        }

        if affected_columns is not None:
            result["affected_columns"] = affected_columns
            result["affected_column_count"] = len(affected_columns)

        issues.append(result)

    def format_columns(column_list):
        """
        Convert:
        ["A", "B", "C"]

        into:
        "'A', 'B', 'C'"
        """

        return ", ".join(
            f"'{column}'"
            for column in column_list
        )

    def format_column_percentages(items, percentage_key):
        """
        Example output:

        'Alley' (93.77%), 'PoolQC' (99.52%)
        """

        return ", ".join(
            f"'{item['column']}' "
            f"({item[percentage_key]}%)"
            for item in items
        )

    # ==========================================================
    # GET PROFILE INFORMATION
    # ==========================================================

    dataset = profile.get("dataset", {})

    ml_context = profile.get(
        "ml_context",
        {}
    )

    target = profile.get(
        "target",
        {}
    )

    columns = profile.get(
        "columns",
        {}
    )

    # ==========================================================
    # BASIC DATASET INFORMATION
    # ==========================================================

    rows = dataset.get(
        "rows",
        0
    )

    duplicate_rows = dataset.get(
        "duplicate_rows",
        0
    )

    duplicate_percentage = dataset.get(
        "duplicate_percentage",
        0
    )

    overall_missing_percentage = dataset.get(
        "overall_missing_percentage",
        0
    )

    # ==========================================================
    # ML CONTEXT
    # ==========================================================

    problem_type = ml_context.get(
        "problem_type",
        "unknown"
    )

    target_column = ml_context.get(
        "target_column"
    )

    # ==========================================================
    # 1. DATASET SIZE
    # ==========================================================

    if rows < 100:

        add_issue(
            severity="HIGH",
            category="Dataset Size",
            issue="Very small dataset",
            evidence=f"The dataset contains only {rows} rows.",
            impact=(
                "A very small dataset may lead to unstable "
                "statistical estimates and poor machine-learning "
                "generalization."
            ),
            recommendation=(
                "Collect more observations if possible before "
                "training a production ML model."
            )
        )

    elif rows < 500:

        add_issue(
            severity="MEDIUM",
            category="Dataset Size",
            issue="Relatively small dataset",
            evidence=f"The dataset contains {rows} rows.",
            impact=(
                "A relatively small sample can limit the reliability "
                "of model evaluation."
            ),
            recommendation=(
                "Use appropriate cross-validation and consider "
                "collecting additional data."
            )
        )

    # ==========================================================
    # 2. OVERALL MISSING VALUES
    # ==========================================================

    if overall_missing_percentage >= 30:

        add_issue(
            severity="HIGH",
            category="Missing Values",
            issue="High overall missingness",
            evidence=(
                f"{overall_missing_percentage}% of all dataset "
                "cells are missing."
            ),
            impact=(
                "A large amount of missing data can reduce usable "
                "information and introduce bias."
            ),
            recommendation=(
                "Investigate the missing-data mechanism and apply "
                "appropriate deletion or imputation strategies."
            )
        )

    elif overall_missing_percentage >= 10:

        add_issue(
            severity="MEDIUM",
            category="Missing Values",
            issue="Moderate overall missingness",
            evidence=(
                f"{overall_missing_percentage}% of dataset cells "
                "are missing."
            ),
            impact=(
                "Missing values may reduce model performance or "
                "cause problems in algorithms that do not support "
                "missing data."
            ),
            recommendation=(
                "Review missingness by column and apply an "
                "appropriate imputation strategy."
            )
        )

    elif overall_missing_percentage > 0:

        add_issue(
            severity="LOW",
            category="Missing Values",
            issue="Some missing values detected",
            evidence=(
                f"{overall_missing_percentage}% of dataset cells "
                "are missing."
            ),
            impact=(
                "Even limited missingness should be reviewed "
                "before model training."
            ),
            recommendation=(
                "Inspect affected columns and choose suitable "
                "imputation or removal methods."
            )
        )

    # ==========================================================
    # 3. DUPLICATE ROWS
    # ==========================================================

    if duplicate_percentage >= 10:

        add_issue(
            severity="HIGH",
            category="Duplicates",
            issue="High number of duplicate rows",
            evidence=(
                f"{duplicate_rows} duplicate rows were detected "
                f"({duplicate_percentage}% of the dataset)."
            ),
            impact=(
                "Duplicates can overweight repeated observations "
                "and may cause misleading model performance."
            ),
            recommendation=(
                "Investigate duplicate records and remove genuine "
                "duplicates before model training."
            )
        )

    elif duplicate_percentage >= 2:

        add_issue(
            severity="MEDIUM",
            category="Duplicates",
            issue="Duplicate rows detected",
            evidence=(
                f"{duplicate_rows} duplicate rows were detected "
                f"({duplicate_percentage}% of the dataset)."
            ),
            impact=(
                "Repeated observations may bias analysis and "
                "machine-learning models."
            ),
            recommendation=(
                "Verify whether duplicates represent legitimate "
                "repeated observations or accidental duplication."
            )
        )

    elif duplicate_rows > 0:

        add_issue(
            severity="LOW",
            category="Duplicates",
            issue="Small number of duplicate rows",
            evidence=(
                f"{duplicate_rows} duplicate rows were detected."
            ),
            impact=(
                "Duplicate observations can slightly influence "
                "analysis and model training."
            ),
            recommendation=(
                "Review duplicate rows and remove them if they "
                "represent accidental duplication."
            )
        )

    # ==========================================================
    # 4. POSSIBLE IDENTIFIER COLUMNS
    # ==========================================================

    identifiers = profile.get(
        "possible_identifiers",
        []
    )

    if identifiers:

        add_issue(
            severity="MEDIUM",
            category="Feature Quality",
            issue="Possible identifier column",
            evidence=(
                f"{len(identifiers)} column(s) have very high "
                f"uniqueness and appear to behave like identifiers: "
                f"{format_columns(identifiers)}."
            ),
            impact=(
                "Identifiers usually do not contain meaningful "
                "predictive information and can introduce noise "
                "or misleading patterns."
            ),
            recommendation=(
                "Review these columns and exclude them from ML "
                "features unless there is a justified modeling "
                "reason to keep them."
            ),
            column=identifiers[0] if len(identifiers) == 1 else None,
            affected_columns=identifiers
        )

    # ==========================================================
    # 5. CONSTANT COLUMNS
    # ==========================================================

    constant_columns = profile.get(
        "constant_columns",
        []
    )

    if constant_columns:

        add_issue(
            severity="HIGH",
            category="Feature Quality",
            issue="Constant columns detected",
            evidence=(
                f"{len(constant_columns)} column(s) contain only "
                f"one unique value: "
                f"{format_columns(constant_columns)}."
            ),
            impact=(
                "Constant features contain no useful variation "
                "for most machine-learning workflows."
            ),
            recommendation=(
                "Review and remove constant columns before "
                "model training where appropriate."
            ),
            column=(
                constant_columns[0]
                if len(constant_columns) == 1
                else None
            ),
            affected_columns=constant_columns
        )

    # ==========================================================
    # 6. NEAR-CONSTANT COLUMNS
    # ==========================================================

    near_constant_columns = profile.get(
        "near_constant_columns",
        []
    )

    if near_constant_columns:

        add_issue(
            severity="MEDIUM",
            category="Feature Quality",
            issue="Near-constant columns detected",
            evidence=(
                f"{len(near_constant_columns)} column(s) have one "
                f"dominant value representing at least 99% of "
                f"non-missing values: "
                f"{format_columns(near_constant_columns)}."
            ),
            impact=(
                "Features with almost no variation generally "
                "provide limited predictive information."
            ),
            recommendation=(
                "Review these columns and consider removing them "
                "if they do not provide meaningful information."
            ),
            column=(
                near_constant_columns[0]
                if len(near_constant_columns) == 1
                else None
            ),
            affected_columns=near_constant_columns
        )

    # ==========================================================
    # 7. POSSIBLE NUMERIC COLUMNS
    # ==========================================================
    #
    # KEEP THESE INDIVIDUAL.
    #
    # The safe-fix engine may need the exact column name
    # to perform pd.to_numeric().
    # ==========================================================

    possible_numeric_columns = profile.get(
        "possible_numeric_columns",
        []
    )

    for column in possible_numeric_columns:

        column_info = columns.get(
            column,
            {}
        )

        numeric_like_percentage = column_info.get(
            "numeric_like_percentage",
            0
        )

        if column == target_column:
            continue

        if numeric_like_percentage >= 95:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        add_issue(
            severity=severity,
            category="Data Type",
            issue="Possible incorrect numeric data type",
            column=column,
            evidence=(
                f"{numeric_like_percentage}% of non-missing values "
                f"in '{column}' appear to be numeric, but the "
                "column is currently stored as a non-numeric type."
            ),
            impact=(
                "A numeric feature stored as text may be excluded "
                "from numerical analysis or interpreted incorrectly "
                "by machine-learning pipelines."
            ),
            recommendation=(
                f"Validate '{column}', handle invalid values, "
                "and convert it to a numeric data type if appropriate."
            )
        )

    # ==========================================================
    # 8. POSSIBLE DATE COLUMNS
    # ==========================================================

    possible_date_columns = profile.get(
        "possible_date_columns",
        []
    )

    date_items = []

    for column in possible_date_columns:

        column_info = columns.get(
            column,
            {}
        )

        date_items.append({
            "column": column,
            "date_like_percentage": column_info.get(
                "date_like_percentage",
                0
            )
        })

    if date_items:

        evidence = ", ".join(
            f"'{item['column']}' "
            f"({item['date_like_percentage']}% date-like)"
            for item in date_items
        )

        date_columns = [
            item["column"]
            for item in date_items
        ]

        add_issue(
            severity="LOW",
            category="Data Type",
            issue="Possible date columns stored as text",
            evidence=(
                f"{len(date_items)} possible date-like column(s) "
                f"were detected: {evidence}."
            ),
            impact=(
                "Treating dates as plain text can prevent correct "
                "time-based analysis and feature engineering."
            ),
            recommendation=(
                "Validate these columns and convert appropriate "
                "date-like columns to datetime types."
            ),
            column=(
                date_columns[0]
                if len(date_columns) == 1
                else None
            ),
            affected_columns=date_columns
        )

    # ==========================================================
    # 9. COLUMN-LEVEL MISSING VALUES
    # ==========================================================

    very_high_missing = []
    high_missing = []

    for column, info in columns.items():

        missing_percentage = info.get(
            "missing_percentage",
            0
        )

        if missing_percentage >= 50:

            very_high_missing.append({
                "column": column,
                "missing_percentage": missing_percentage
            })

        elif missing_percentage >= 20:

            high_missing.append({
                "column": column,
                "missing_percentage": missing_percentage
            })

    # ----------------------------------------------------------
    # VERY HIGH MISSINGNESS
    # ----------------------------------------------------------

    if very_high_missing:

        affected = [
            item["column"]
            for item in very_high_missing
        ]

        add_issue(
            severity="HIGH",
            category="Missing Values",
            issue="Very high column-level missingness",
            evidence=(
                f"{len(very_high_missing)} column(s) have at least "
                f"50% missing values: "
                f"{format_column_percentages(
                    very_high_missing,
                    'missing_percentage'
                )}."
            ),
            impact=(
                "Features with more than half of their values "
                "missing may provide limited reliable information."
            ),
            recommendation=(
                "Investigate whether these columns should be removed "
                "or whether domain-specific imputation strategies "
                "are appropriate."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ----------------------------------------------------------
    # MODERATE/HIGH COLUMN MISSINGNESS
    # ----------------------------------------------------------

    if high_missing:

        affected = [
            item["column"]
            for item in high_missing
        ]

        add_issue(
            severity="MEDIUM",
            category="Missing Values",
            issue="High column-level missingness",
            evidence=(
                f"{len(high_missing)} column(s) have between "
                f"20% and 50% missing values: "
                f"{format_column_percentages(
                    high_missing,
                    'missing_percentage'
                )}."
            ),
            impact=(
                "Substantial missingness can reduce the reliability "
                "of affected features."
            ),
            recommendation=(
                "Investigate the reason for missing values in these "
                "columns and apply appropriate imputation or removal "
                "strategies."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ==========================================================
    # 10. NUMERIC OUTLIERS
    # ==========================================================

    high_outliers = []
    medium_outliers = []
    low_outliers = []

    for column, info in columns.items():

        outlier_percentage = info.get(
            "outlier_percentage",
            0
        )

        min_value = info.get("min")
        max_value = info.get("max")

        unique_count = info.get(
            "unique_count",
            0
        )

        is_binary = (
            min_value is not None
            and max_value is not None
            and min_value >= 0
            and max_value <= 1
            and unique_count <= 2
        )

        if is_binary:
            continue

        if outlier_percentage >= 10:

            high_outliers.append({
                "column": column,
                "outlier_percentage": outlier_percentage
            })

        elif outlier_percentage >= 5:

            medium_outliers.append({
                "column": column,
                "outlier_percentage": outlier_percentage
            })

        elif outlier_percentage > 0:

            low_outliers.append({
                "column": column,
                "outlier_percentage": outlier_percentage
            })

    # ----------------------------------------------------------
    # HIGH OUTLIERS
    # ----------------------------------------------------------

    if high_outliers:

        affected = [
            item["column"]
            for item in high_outliers
        ]

        add_issue(
            severity="HIGH",
            category="Outliers",
            issue="High proportion of potential outliers",
            evidence=(
                f"{len(high_outliers)} column(s) contain at least "
                f"10% potential outliers based on the IQR method: "
                f"{format_column_percentages(
                    high_outliers,
                    'outlier_percentage'
                )}."
            ),
            impact=(
                "Extreme observations can strongly influence some "
                "statistical analyses and machine-learning algorithms."
            ),
            recommendation=(
                "Investigate these outliers and determine whether "
                "they are valid observations or data-quality problems."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ----------------------------------------------------------
    # MEDIUM OUTLIERS
    # ----------------------------------------------------------

    if medium_outliers:

        affected = [
            item["column"]
            for item in medium_outliers
        ]

        add_issue(
            severity="MEDIUM",
            category="Outliers",
            issue="Potential outliers detected",
            evidence=(
                f"{len(medium_outliers)} column(s) contain between "
                f"5% and 10% potential outliers: "
                f"{format_column_percentages(
                    medium_outliers,
                    'outlier_percentage'
                )}."
            ),
            impact=(
                "Outliers may affect models that are sensitive "
                "to extreme values."
            ),
            recommendation=(
                "Review these distributions and consider "
                "transformation, capping, or robust modeling "
                "where appropriate."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ----------------------------------------------------------
    # LOW OUTLIERS
    # ----------------------------------------------------------

    if low_outliers:

        affected = [
            item["column"]
            for item in low_outliers
        ]

        add_issue(
            severity="LOW",
            category="Outliers",
            issue="Small number of potential outliers",
            evidence=(
                f"{len(low_outliers)} column(s) contain less than "
                f"5% potential outliers: "
                f"{format_column_percentages(
                    low_outliers,
                    'outlier_percentage'
                )}."
            ),
            impact=(
                "A small number of extreme observations may still "
                "affect certain models."
            ),
            recommendation=(
                "Review the identified observations before deciding "
                "whether treatment is necessary."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ==========================================================
    # 11. HIGH SKEWNESS
    # ==========================================================

    skewed_features = []

    for column, info in columns.items():

        skewness = info.get(
            "skewness"
        )

        highly_skewed = info.get(
            "highly_skewed",
            False
        )

        if (
            highly_skewed
            and skewness is not None
        ):

            skewed_features.append({
                "column": column,
                "skewness": skewness
            })

    if skewed_features:

        affected = [
            item["column"]
            for item in skewed_features
        ]

        skew_evidence = ", ".join(
            f"'{item['column']}' "
            f"({item['skewness']})"
            for item in skewed_features
        )

        add_issue(
            severity="MEDIUM",
            category="Distribution",
            issue="Highly skewed numeric features",
            evidence=(
                f"{len(skewed_features)} numeric feature(s) "
                f"show high skewness: {skew_evidence}."
            ),
            impact=(
                "Strongly skewed distributions can affect "
                "distance-based, linear, and statistical models."
            ),
            recommendation=(
                "Inspect these distributions and consider suitable "
                "transformations if required by the chosen algorithm."
            ),
            column=affected[0] if len(affected) == 1 else None,
            affected_columns=affected
        )

    # ==========================================================
    # 12. HIGH CORRELATION
    # ==========================================================

    high_correlation_pairs = profile.get(
        "high_correlation_pairs",
        []
    )

    if high_correlation_pairs:

        pair_evidence = []
        affected = set()

        for pair in high_correlation_pairs:

            col1 = pair.get(
                "column_1"
            )

            col2 = pair.get(
                "column_2"
            )

            correlation = pair.get(
                "correlation"
            )

            affected.add(col1)
            affected.add(col2)

            pair_evidence.append(
                f"'{col1}' and '{col2}' ({correlation})"
            )

        affected = sorted(
            column
            for column in affected
            if column is not None
        )

        add_issue(
            severity="MEDIUM",
            category="Multicollinearity",
            issue="Highly correlated features detected",
            evidence=(
                f"{len(high_correlation_pairs)} highly correlated "
                f"feature pair(s) were detected: "
                f"{', '.join(pair_evidence)}."
            ),
            impact=(
                "Highly correlated features can introduce redundant "
                "information and may cause multicollinearity "
                "in some models."
            ),
            recommendation=(
                "Review these correlated features and consider "
                "feature selection or dimensionality reduction "
                "where appropriate."
            ),
            affected_columns=affected
        )

    # ==========================================================
    # 13. TARGET MISSING VALUES
    # ==========================================================

    if target.get("provided"):

        if target_column in columns:

            target_missing_percentage = columns[
                target_column
            ].get(
                "missing_percentage",
                0
            )

            if target_missing_percentage >= 20:

                add_issue(
                    severity="HIGH",
                    category="Target Quality",
                    issue="High target missingness",
                    column=target_column,
                    evidence=(
                        f"The target column '{target_column}' "
                        f"has {target_missing_percentage}% "
                        "missing values."
                    ),
                    impact=(
                        "Rows without target values cannot normally "
                        "be used for standard supervised model training."
                    ),
                    recommendation=(
                        f"Investigate why '{target_column}' is missing "
                        "and determine whether affected rows should "
                        "be removed or the target reconstructed."
                    )
                )

            elif target_missing_percentage > 0:

                add_issue(
                    severity="MEDIUM",
                    category="Target Quality",
                    issue="Missing target values",
                    column=target_column,
                    evidence=(
                        f"The target column '{target_column}' "
                        f"has {target_missing_percentage}% "
                        "missing values."
                    ),
                    impact=(
                        "Rows with missing target values cannot "
                        "directly contribute to supervised training."
                    ),
                    recommendation=(
                        f"Investigate and appropriately handle "
                        f"missing '{target_column}' values."
                    )
                )

    # ==========================================================
    # 14. CLASS IMBALANCE
    # ==========================================================

    target_imbalanced = target.get(
        "imbalanced",
        False
    )

    if target_imbalanced:

        largest_class_percentage = target.get(
            "largest_class_percentage",
            0
        )

        add_issue(
            severity="HIGH",
            category="Target Distribution",
            issue="Potential class imbalance",
            column=target_column,
            evidence=(
                f"The largest target class represents "
                f"{largest_class_percentage}% of the target "
                "observations."
            ),
            impact=(
                "A strongly imbalanced target can cause a model "
                "to favor the majority class and produce misleading "
                "accuracy."
            ),
            recommendation=(
                "Use stratified validation and evaluate precision, "
                "recall, F1-score, ROC-AUC or PR-AUC as appropriate. "
                "Consider class weighting or resampling when justified."
            )
        )

    # ==========================================================
    # 15. UNSUPERVISED ML CHECKS
    # ==========================================================

    if (
        problem_type in [
            "clustering",
            "unsupervised",
            "unknown_unsupervised_possible"
        ]
        or target_column is None
    ):

        numeric_feature_count = ml_context.get(
            "numeric_feature_count",
            0
        )

        if numeric_feature_count == 0:

            add_issue(
                severity="HIGH",
                category="Unsupervised ML Readiness",
                issue="No numeric features detected",
                evidence=(
                    "The profiler did not identify any numeric "
                    "features available for standard numerical "
                    "clustering or distance-based analysis."
                ),
                impact=(
                    "Many common clustering algorithms require "
                    "numeric feature representations."
                ),
                recommendation=(
                    "Encode suitable categorical variables or "
                    "select an algorithm that supports categorical data."
                )
            )

        elif numeric_feature_count == 1:

            add_issue(
                severity="MEDIUM",
                category="Unsupervised ML Readiness",
                issue="Only one numeric feature detected",
                evidence=(
                    "Only one numeric feature is available."
                ),
                impact=(
                    "A single numeric feature provides limited "
                    "information for many clustering applications."
                ),
                recommendation=(
                    "Review categorical variables and determine "
                    "whether additional meaningful features can "
                    "be encoded or engineered."
                )
            )

    # ==========================================================
    # 16. UNKNOWN PROBLEM TYPE
    # ==========================================================

    if problem_type in [
        "unknown",
        "unknown_unsupervised_possible"
    ]:

        add_issue(
            severity="LOW",
            category="ML Context",
            issue="Machine-learning problem type is not confirmed",
            evidence=(
                "No confirmed target/problem type was supplied "
                "to the profiler."
            ),
            impact=(
                "Different ML tasks require different preprocessing, "
                "validation strategies and evaluation metrics."
            ),
            recommendation=(
                "Confirm whether the objective is classification, "
                "regression, clustering, anomaly detection, or another task."
            )
        )

    # ==========================================================
    # SORT BY SEVERITY
    # ==========================================================

    severity_order = {
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3
    }

    issues = sorted(
        issues,
        key=lambda x: severity_order.get(
            x["severity"],
            99
        )
    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    high_count = sum(
        1
        for issue in issues
        if issue["severity"] == "HIGH"
    )

    medium_count = sum(
        1
        for issue in issues
        if issue["severity"] == "MEDIUM"
    )

    low_count = sum(
        1
        for issue in issues
        if issue["severity"] == "LOW"
    )

    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================

    return {
        "total_issues": len(issues),

        "summary": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count
        },

        "issues": issues
    }


def calculate_quality_score(profile, issues_result):
    """
    Calculate a deterministic DataDoctor quality score.

    Score starts at 100.

    Issue penalties are severity-based and capped so that
    a dataset with many related warnings is not unfairly
    pushed to zero.

    Same dataset + same profiler rules = same score.
    """

    score = 100.0

    deductions = []

    # ==========================================================
    # GET INFORMATION
    # ==========================================================

    issues = issues_result.get(
        "issues",
        []
    )

    # ==========================================================
    # SEVERITY PENALTIES
    # ==========================================================

    severity_penalty = {
        "HIGH": 12,
        "MEDIUM": 3,
        "LOW": 1
    }

    # Maximum total deduction allowed per severity
    severity_caps = {
        "HIGH": 36,
        "MEDIUM": 24,
        "LOW": 8
    }

    severity_totals = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    # ==========================================================
    # PROCESS ISSUES
    # ==========================================================

    for issue in issues:

        severity = issue.get(
            "severity",
            "LOW"
        )

        penalty = severity_penalty.get(
            severity,
            0
        )

        cap = severity_caps.get(
            severity,
            0
        )

        if penalty <= 0:
            continue

        current_total = severity_totals.get(
            severity,
            0
        )

        remaining_cap = cap - current_total

        # ------------------------------------------------------
        # CAP ALREADY REACHED
        # ------------------------------------------------------

        if remaining_cap <= 0:

            deductions.append({
                "severity": severity,
                "category": issue.get("category"),
                "issue": issue.get("issue"),
                "column": issue.get("column"),
                "base_penalty": penalty,
                "deduction": 0,
                "capped": True,
                "reason": (
                    f"{severity} severity deduction cap "
                    "has already been reached."
                )
            })

            continue

        # ------------------------------------------------------
        # APPLY PENALTY
        # ------------------------------------------------------

        applied_penalty = min(
            penalty,
            remaining_cap
        )

        score -= applied_penalty

        severity_totals[severity] += applied_penalty

        deductions.append({
            "severity": severity,
            "category": issue.get("category"),
            "issue": issue.get("issue"),
            "column": issue.get("column"),
            "base_penalty": penalty,
            "deduction": applied_penalty,
            "capped": applied_penalty < penalty
        })

    # ==========================================================
    # KEEP SCORE BETWEEN 0 AND 100
    # ==========================================================

    score = max(
        0,
        min(
            100,
            score
        )
    )

    score = round(
        score,
        2
    )

    # ==========================================================
    # SCORE INTERPRETATION
    # ==========================================================

    if score >= 90:

        rating = "Excellent"

        description = (
            "The dataset appears highly prepared for "
            "analysis or machine-learning workflows."
        )

    elif score >= 75:

        rating = "Good"

        description = (
            "The dataset is generally usable, but some "
            "data-quality issues should be reviewed."
        )

    elif score >= 60:

        rating = "Needs Improvement"

        description = (
            "Several data-quality issues may affect "
            "analysis or machine-learning performance."
        )

    elif score >= 40:

        rating = "Poor"

        description = (
            "Significant data-quality issues should be "
            "addressed before reliable ML modeling."
        )

    else:

        rating = "Critical"

        description = (
            "The dataset has severe issues and requires "
            "substantial cleaning before ML use."
        )

    # ==========================================================
    # ISSUE COUNTS
    # ==========================================================

    high_count = sum(
        1
        for issue in issues
        if issue.get("severity") == "HIGH"
    )

    medium_count = sum(
        1
        for issue in issues
        if issue.get("severity") == "MEDIUM"
    )

    low_count = sum(
        1
        for issue in issues
        if issue.get("severity") == "LOW"
    )

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    return {
        "quality_score": score,

        "rating": rating,

        "description": description,

        "issue_counts": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "total": len(issues)
        },

        "severity_deductions": {
            "HIGH": severity_totals["HIGH"],
            "MEDIUM": severity_totals["MEDIUM"],
            "LOW": severity_totals["LOW"]
        },

        "deductions": deductions
    }


def generate_fixes(df, issues):
    """
    Generate safe, actionable data-quality fixes based on
    issues detected by DataDoctor.

    IMPORTANT:
    This function does NOT modify the original dataframe.
    It only creates a structured fix plan.
    """

    fixes = []

    # ---------------------------------------------------------
    # Helper function
    # ---------------------------------------------------------

    def add_fix(
        issue,
        column,
        action_type,
        action,
        reason,
        safety,
        auto_fix
    ):
        fixes.append({
            "issue": issue,
            "column": column,
            "action_type": action_type,
            "action": action,
            "reason": reason,
            "safety": safety,
            "auto_fix": auto_fix
        })

    # ---------------------------------------------------------
    # Check detected issues
    # ---------------------------------------------------------

    for issue in issues.get("issues", []):

        issue_text = str(issue).lower()

        column = issue.get("column", None)

        # =====================================================
        # 1. NUMERIC COLUMN STORED AS TEXT
        # =====================================================

        if (
            "incorrect numeric data type" in issue_text
            or "numeric data type" in issue_text
        ):

            if column and column in df.columns:

                add_fix(
                    issue="Possible incorrect numeric data type",
                    column=column,
                    action_type="automatic",
                    action=f"Convert '{column}' to numeric after validating non-numeric values.",
                    reason="The profiler identified a column that appears numeric but is stored as a non-numeric data type.",
                    safety="safe_after_validation",
                    auto_fix=True
                )

        # =====================================================
        # 2. DUPLICATE ROWS
        # =====================================================

        elif "duplicate" in issue_text:

            duplicate_count = int(df.duplicated().sum())

            if duplicate_count > 0:

                add_fix(
                    issue="Duplicate rows",
                    column=None,
                    action_type="automatic",
                    action="Remove exact duplicate rows.",
                    reason="Exact duplicate rows can cause repeated observations during analysis and model training.",
                    safety="generally_safe",
                    auto_fix=True
                )

        # =====================================================
        # 3. IDENTIFIER COLUMN
        # =====================================================

        elif (
            "identifier" in issue_text
            or "possible identifier" in issue_text
        ):

            if column and column in df.columns:

                add_fix(
                    issue="Possible identifier column",
                    column=column,
                    action_type="recommendation",
                    action=f"Exclude '{column}' from ML features.",
                    reason="Identifier-like columns generally represent records rather than predictive characteristics.",
                    safety="review_before_removal",
                    auto_fix=False
                )

        # =====================================================
        # 4. MISSING VALUES
        # =====================================================

        elif (
            "missing" in issue_text
            or "null" in issue_text
        ):

            if column and column in df.columns:

                missing_count = int(df[column].isna().sum())

                if missing_count > 0:

                    add_fix(
                        issue="Missing values",
                        column=column,
                        action_type="recommendation",
                        action=f"Handle {missing_count} missing values in '{column}' using an appropriate imputation or removal strategy.",
                        reason="Missing values may prevent some ML algorithms from processing the feature correctly.",
                        safety="requires_review",
                        auto_fix=False
                    )

        # =====================================================
        # 5. CLASS IMBALANCE
        # =====================================================

        elif (
            "class imbalance" in issue_text
            or "target distribution" in issue_text
        ):

            add_fix(
                issue="Potential class imbalance",
                column=column,
                action_type="recommendation",
                action="Use stratified validation and consider class weighting or resampling when appropriate.",
                reason="Class imbalance can cause models to favor the majority class.",
                safety="requires_modeling_decision",
                auto_fix=False
            )

        # =====================================================
        # 6. OUTLIERS
        # =====================================================

        elif "outlier" in issue_text:

            add_fix(
                issue="Potential outliers",
                column=column,
                action_type="recommendation",
                action=f"Investigate potential outliers in '{column}' before deciding whether to transform, cap, or remove them.",
                reason="Outliers may represent legitimate observations or data-quality problems.",
                safety="requires_review",
                auto_fix=False
            )

    # ---------------------------------------------------------
    # Return structured result
    # ---------------------------------------------------------

    automatic_fixes = [
        fix for fix in fixes
        if fix["auto_fix"] is True
    ]

    recommendations = [
        fix for fix in fixes
        if fix["auto_fix"] is False
    ]

    return {
        "summary": {
            "total_actions": len(fixes),
            "automatic_fixes": len(automatic_fixes),
            "recommendations": len(recommendations)
        },
        "automatic_fixes": automatic_fixes,
        "recommendations": recommendations
    }


def apply_safe_fixes(df, fixes):
    """
    Apply only fixes explicitly marked as safe/automatic
    by DataDoctor.

    The original dataframe is not modified.
    A cleaned copy is returned.
    """

    cleaned_df = df.copy()

    applied_fixes = []
    skipped_fixes = []

    # ---------------------------------------------------------
    # Process automatic fixes
    # ---------------------------------------------------------

    for fix in fixes.get("automatic_fixes", []):

        column = fix.get("column")
        issue = fix.get("issue", "")

        # =====================================================
        # Numeric data type conversion
        # =====================================================

        if (
            "numeric data type" in issue.lower()
            and column in cleaned_df.columns
        ):

            original_dtype = str(
                cleaned_df[column].dtype
            )

            # Convert to numeric.
            # Invalid values become NaN so they can be
            # reviewed rather than silently deleted.
            converted = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )

            invalid_count = int(
                converted.isna().sum()
                - cleaned_df[column].isna().sum()
            )

            cleaned_df[column] = converted

            applied_fixes.append({
                "issue": issue,
                "column": column,
                "action": f"Converted '{column}' to numeric.",
                "original_dtype": original_dtype,
                "new_dtype": str(
                    cleaned_df[column].dtype
                ),
                "new_invalid_values": invalid_count
            })

        # =====================================================
        # Duplicate removal
        # =====================================================

        elif "duplicate" in issue.lower():

            before = len(cleaned_df)

            cleaned_df = cleaned_df.drop_duplicates().reset_index(
                drop=True
            )

            removed = before - len(cleaned_df)

            applied_fixes.append({
                "issue": issue,
                "column": None,
                "action": "Removed exact duplicate rows.",
                "rows_removed": removed
            })

        else:

            skipped_fixes.append({
                "issue": issue,
                "column": column,
                "reason": "Automatic fix is not implemented for this issue."
            })

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------

    return {
        "dataframe": cleaned_df,

        "summary": {
            "fixes_applied": len(applied_fixes),
            "fixes_skipped": len(skipped_fixes)
        },

        "applied_fixes": applied_fixes,

        "skipped_fixes": skipped_fixes
    }


def generate_comparison(
    score_before,
    score_after,
    issues_before,
    issues_after,
    fix_result
):
    """
    Generate an intelligent Before vs After comparison.

    Identifies:
    - resolved issues
    - newly discovered issues
    - remaining issues
    - quality score improvement
    - applied fixes
    """

    # ---------------------------------------------------------
    # Scores
    # ---------------------------------------------------------

    before_score = float(
        score_before.get("quality_score", 0)
    )

    after_score = float(
        score_after.get("quality_score", 0)
    )

    score_change = round(
        after_score - before_score,
        2
    )

    # ---------------------------------------------------------
    # Issue lists
    # ---------------------------------------------------------

    before_issues = issues_before.get(
        "issues", []
    )

    after_issues = issues_after.get(
        "issues", []
    )

    # ---------------------------------------------------------
    # Create issue identifiers
    # ---------------------------------------------------------

    def issue_key(issue):

        return (
            issue.get("category"),
            issue.get("issue"),
            issue.get("column")
        )

    before_keys = {
        issue_key(issue)
        for issue in before_issues
    }

    after_keys = {
        issue_key(issue)
        for issue in after_issues
    }

    # ---------------------------------------------------------
    # Resolved issues
    # ---------------------------------------------------------

    resolved_issues = []

    for issue in before_issues:

        key = issue_key(issue)

        if key not in after_keys:

            resolved_issues.append({
                "severity": issue.get("severity"),
                "category": issue.get("category"),
                "issue": issue.get("issue"),
                "column": issue.get("column"),
                "status": "RESOLVED"
            })

    # ---------------------------------------------------------
    # Newly discovered issues
    # ---------------------------------------------------------

    new_issues = []

    for issue in after_issues:

        key = issue_key(issue)

        if key not in before_keys:

            new_issues.append({
                "severity": issue.get("severity"),
                "category": issue.get("category"),
                "issue": issue.get("issue"),
                "column": issue.get("column"),
                "evidence": issue.get("evidence"),
                "status": "NEWLY_DISCOVERED"
            })

    # ---------------------------------------------------------
    # Remaining issues
    # ---------------------------------------------------------

    remaining_issues = []

    for issue in after_issues:

        key = issue_key(issue)

        if key in before_keys:

            remaining_issues.append({
                "severity": issue.get("severity"),
                "category": issue.get("category"),
                "issue": issue.get("issue"),
                "column": issue.get("column"),
                "evidence": issue.get("evidence"),
                "recommended_action": issue.get(
                    "recommended_action"
                ),
                "status": "REMAINING"
            })

    # ---------------------------------------------------------
    # Applied fixes
    # ---------------------------------------------------------

    applied_fixes = []

    for fix in fix_result.get(
        "applied_fixes", []
    ):

        applied_fixes.append({
            "issue": fix.get("issue"),
            "column": fix.get("column"),
            "action": fix.get("action")
        })

    # ---------------------------------------------------------
    # Issue counts
    # ---------------------------------------------------------

    before_summary = issues_before.get(
        "summary", {}
    )

    after_summary = issues_after.get(
        "summary", {}
    )

    # ---------------------------------------------------------
    # Improvement status
    # ---------------------------------------------------------

    if score_change > 0:

        status = "Improved"

    elif score_change < 0:

        status = "Needs Further Cleaning"

    else:

        status = "No Score Change"

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    comparison = {

        "before": {
            "quality_score": before_score,
            "rating": score_before.get("rating"),
            "high_issues": before_summary.get("high", 0),
            "medium_issues": before_summary.get("medium", 0),
            "low_issues": before_summary.get("low", 0),
            "total_issues": issues_before.get(
                "total_issues", 0
            )
        },

        "after": {
            "quality_score": after_score,
            "rating": score_after.get("rating"),
            "high_issues": after_summary.get("high", 0),
            "medium_issues": after_summary.get("medium", 0),
            "low_issues": after_summary.get("low", 0),
            "total_issues": issues_after.get(
                "total_issues", 0
            )
        },

        "improvement": {
            "score_change": score_change,
            "status": status,
            "issues_resolved": len(
                resolved_issues
            ),
            "new_issues_discovered": len(
                new_issues
            ),
            "remaining_issues": len(
                remaining_issues
            )
        },

        "applied_fixes": applied_fixes,

        "resolved_issues": resolved_issues,

        "new_issues_discovered": new_issues,

        "remaining_issues": remaining_issues
    }

    return comparison



def create_ai_payload(
    profile_before,
    issues_before,
    quality_before,
    fix_plan,
    fix_result,
    profile_after,
    issues_after,
    quality_after,
    comparison
):
    """
    Creates a clean, structured payload
    for the DataDoctor AI analysis layer.

    Python generates all factual evidence.
    Gemini interprets the evidence and explains it.
    """

    # --------------------------------------------------------
    # BEFORE ML CONTEXT
    # --------------------------------------------------------

    ml_context_before = profile_before.get(
        "ml_context",
        {}
    )

    dataset_before = profile_before.get(
        "dataset",
        {}
    )

    # --------------------------------------------------------
    # AFTER ML CONTEXT
    # --------------------------------------------------------

    ml_context_after = profile_after.get(
        "ml_context",
        {}
    )

    dataset_after = profile_after.get(
        "dataset",
        {}
    )

    # --------------------------------------------------------
    # FINAL AI PAYLOAD
    # --------------------------------------------------------

    return {

        "system": "DataDoctor",

        "purpose": (
            "Analyze dataset quality and ML readiness using "
            "evidence generated by the Python profiler, "
            "safe-fix engine, and before-vs-after comparison."
        ),

        # ====================================================
        # DATASET SUMMARY
        # ====================================================

        "dataset_summary": {

            "rows_before": dataset_before.get(
                "rows"
            ),

            "columns_before": dataset_before.get(
                "columns"
            ),

            "rows_after": dataset_after.get(
                "rows"
            ),

            "columns_after": dataset_after.get(
                "columns"
            ),

            "duplicate_rows_before": dataset_before.get(
                "duplicate_rows"
            ),

            "missing_percentage_before": dataset_before.get(
                "overall_missing_percentage"
            ),

            "missing_percentage_after": dataset_after.get(
                "overall_missing_percentage"
            )
        },

        # ====================================================
        # ML CONTEXT
        # ====================================================

        "ml_context": {

            "problem_type": ml_context_before.get(
                "problem_type"
            ),

            "target_column": ml_context_before.get(
                "target_column"
            ),

            "feature_count": ml_context_before.get(
                "feature_count"
            ),

            "numeric_feature_count": ml_context_before.get(
                "numeric_feature_count"
            ),

            "categorical_feature_count": ml_context_before.get(
                "categorical_feature_count"
            )
        },

        # ====================================================
        # BEFORE ANALYSIS
        # ====================================================

        "before": {

            "quality_score": quality_before.get(
                "quality_score"
            ),

            "rating": quality_before.get(
                "rating"
            ),

            "issue_counts": quality_before.get(
                "issue_counts"
            ),

            "issues": issues_before.get(
                "issues",
                []
            )
        },

        # ====================================================
        # FIX PLAN
        # ====================================================

        "fix_plan": {

            "summary": fix_plan.get(
                "summary",
                {}
            ),

            "automatic_fixes": fix_plan.get(
                "automatic_fixes",
                []
            ),

            "recommendations": fix_plan.get(
                "recommendations",
                []
            )
        },

        # ====================================================
        # FIXES ACTUALLY APPLIED
        # ====================================================

        "fix_result": {

            "summary": fix_result.get(
                "summary",
                {}
            ),

            "applied_fixes": fix_result.get(
                "applied_fixes",
                []
            ),

            "skipped_fixes": fix_result.get(
                "skipped_fixes",
                []
            )
        },

        # ====================================================
        # AFTER ANALYSIS
        # ====================================================

        "after": {

            "quality_score": quality_after.get(
                "quality_score"
            ),

            "rating": quality_after.get(
                "rating"
            ),

            "issue_counts": quality_after.get(
                "issue_counts"
            ),

            "issues": issues_after.get(
                "issues",
                []
            )
        },

        # ====================================================
        # BEFORE VS AFTER
        # ====================================================

        "comparison": comparison

    }

# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def root():
    return {
        "service": "DataDoctor",
        "status": "running"
    }


# =========================
# PROFILE ENDPOINT
# =========================

@app.post("/profile")
async def profile_endpoint(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    problem_type: str | None = Form(None)
):
    
    # --------------------------------------------------------
    # 1. Validate file
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are currently supported."
        )


    # --------------------------------------------------------
    # 2. Read CSV
    # --------------------------------------------------------

    try:

        contents = await file.read()

        if len(contents) == 0:

            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {str(e)}"
        )


    # --------------------------------------------------------
    # 3. Validate dataset
    # --------------------------------------------------------

    if df.empty:

        raise HTTPException(
            status_code=400,
            detail="The uploaded dataset contains no rows."
        )


    # --------------------------------------------------------
    # 4. Clean form inputs
    # --------------------------------------------------------

    target_column = (
        target_column.strip()
        if target_column and target_column.strip()
        else None
    )

    problem_type = (
        problem_type.strip()
        if problem_type and problem_type.strip()
        else None
    )


    # --------------------------------------------------------
    # 5. Handle Swagger placeholder values
    # --------------------------------------------------------

    if target_column and target_column.lower() == "string":

        target_column = None

    if problem_type and problem_type.lower() == "string":

        problem_type = None


    # --------------------------------------------------------
    # 6. Validate target column
    # --------------------------------------------------------

    if target_column:

        if target_column not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' "
                    f"was not found in the dataset."
                )
            )


    # ========================================================
    # BEFORE ANALYSIS
    # ========================================================

    # --------------------------------------------------------
    # 7. Profile dataset
    # --------------------------------------------------------

    profile_before = profile_dataset(
        df=df,
        target_column=target_column,
        problem_type=problem_type
    )


    # --------------------------------------------------------
    # 8. Generate issues
    # --------------------------------------------------------

    issues_before = generate_issues(
        profile_before
    )


    # --------------------------------------------------------
    # 9. Calculate quality score
    # --------------------------------------------------------

    score_before = calculate_quality_score(
        profile_before,
        issues_before
    )


    # ========================================================
    # FIX PLAN
    # ========================================================

    # --------------------------------------------------------
    # 10. Generate fixes
    # --------------------------------------------------------

    fix_plan = generate_fixes(
        df,
        issues_before
    )


    # --------------------------------------------------------
    # 11. Apply safe fixes
    # --------------------------------------------------------

    fix_result = apply_safe_fixes(
        df,
        fix_plan
    )


    # --------------------------------------------------------
    # 12. Get cleaned dataframe
    # --------------------------------------------------------

    df_cleaned = fix_result["dataframe"]


    # ==========================================================
    # PREPARE CLEANED CSV FOR DOWNLOAD
    # ==========================================================
    
    cleaned_csv_text = df_cleaned.to_csv(
         index=False
    )

    cleaned_csv_base64 = base64.b64encode(
        cleaned_csv_text.encode("utf-8")
    ).decode("utf-8")


    # Create a clean download filename

    original_filename = file.filename or "dataset.csv"

    original_stem = Path(original_filename).stem

    cleaned_filename = f"{original_stem}_cleaned.csv"

    # ========================================================
    # AFTER ANALYSIS
    # ========================================================

    # --------------------------------------------------------
    # 13. Re-profile cleaned dataset
    # --------------------------------------------------------

    profile_after = profile_dataset(
        df=df_cleaned,
        target_column=target_column,
        problem_type=problem_type
    )


    # --------------------------------------------------------
    # 14. Generate issues after fixes
    # --------------------------------------------------------

    issues_after = generate_issues(
        profile_after
    )


    # --------------------------------------------------------
    # 15. Calculate quality score after fixes
    # --------------------------------------------------------

    score_after = calculate_quality_score(
        profile_after,
        issues_after
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    # --------------------------------------------------------
    # 16. Before vs After comparison
    # --------------------------------------------------------

    comparison = generate_comparison(
        score_before=score_before,
        score_after=score_after,
        issues_before=issues_before,
        issues_after=issues_after,
        fix_result=fix_result
    )


    # ========================================================
    # AI PAYLOAD
    # ========================================================

    # --------------------------------------------------------
    # 17. Create compact AI payload
    # --------------------------------------------------------

    ai_payload = create_ai_payload(
        profile_before=profile_before,
        issues_before=issues_before,
        quality_before=score_before,
        fix_plan=fix_plan,
        fix_result=fix_result,
        profile_after=profile_after,
        issues_after=issues_after,
        quality_after=score_after,
        comparison=comparison
    )

    

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    # --------------------------------------------------------
    # 18. Build response
    # --------------------------------------------------------

    response_data = {

        "status": "success",

        "file": {

            "filename": file.filename,

            "content_type": file.content_type,

            "size_bytes": len(contents),

            "rows": int(df.shape[0]),

            "columns": int(df.shape[1])
        },

        "ml_context": {

            "target_column": target_column,

            "problem_type": problem_type
        },

        "before": {

            "profile": profile_before,

            "issues": issues_before,

            "quality_score": score_before
        },

        "fix_plan": fix_plan,

        "fix_result": {

            "summary": fix_result["summary"],

            "applied_fixes": fix_result["applied_fixes"],

            "skipped_fixes": fix_result["skipped_fixes"]
        },

        "after": {

            "profile": profile_after,

            "issues": issues_after,

            "quality_score": score_after
        },

        "comparison": comparison,

        "ai_payload": ai_payload,

        "cleaned_dataset": {
            
            "filename": cleaned_filename,

            "mime_type": "text/csv",

            "encoding": "base64",

            "data": cleaned_csv_base64,

            "rows": len(df_cleaned),

            "columns": len(df_cleaned.columns)
        }
    }

    # --------------------------------------------------------
    # 19. Convert NumPy/Pandas values to JSON-safe values
    # --------------------------------------------------------

    safe_response = make_json_safe(response_data)

    return safe_response



