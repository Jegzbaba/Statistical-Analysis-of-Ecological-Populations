
import pandas as pd
import math

# --------------------------------------------------
# STEP 1: Load the initial dataset
# --------------------------------------------------

df = pd.read_csv("measurements.csv")

print("ORIGINAL DATA")
print(df.to_string(index=False))


# --------------------------------------------------
# STEP 2: Exclude rows where QC is not PASS
# --------------------------------------------------

df = df[
    df["qc"].astype(str).str.strip().str.upper() == "PASS"
].copy()

print("\nDATA AFTER QC FILTER")
print(df.to_string(index=False))


# --------------------------------------------------
# STEP 3: Convert values and uncertainties to grams
# --------------------------------------------------

unit_factors = {
    "g": 1.0,
    "mg": 0.001,
    "ug": 0.000001
}

# Normalize unit labels to handle extra spaces/case
df["unit"] = df["unit"].astype(str).str.strip().str.lower()

# Stop if any unit is not supported
unsupported = ~df["unit"].isin(unit_factors)

if unsupported.any():
    raise ValueError(
        f"Unsupported units: {df.loc[unsupported, 'unit'].unique()}"
    )

# Convert BOTH value and uncertainty to grams
df["value_g"] = (
    df["value"] * df["unit"].map(unit_factors)
)

df["uncertainty_g"] = (
    df["uncertainty"] * df["unit"].map(unit_factors)
)

# Validate numerical measurements
if not (
    df["value_g"].map(math.isfinite).all()
    and df["uncertainty_g"].map(math.isfinite).all()
):
    raise ValueError("Measurements must be finite numbers.")

if (df["uncertainty_g"] <= 0).any():
    raise ValueError("Uncertainties must be greater than zero.")

print("\nDATA AFTER UNIT CONVERSION")
print(
    df[
        ["run_id", "value_g", "uncertainty_g", "qc"]
    ].to_string(index=False)
)


# --------------------------------------------------
# STEP 4: Calculate inverse-variance weights
# w = 1 / uncertainty_g^2
# --------------------------------------------------

df["weight"] = 1 / (df["uncertainty_g"] ** 2)


# --------------------------------------------------
# STEP 5: Calculate the weighted mean
# weighted_mean = sum(w*x) / sum(w)
# --------------------------------------------------

weighted_mean = (
    (df["weight"] * df["value_g"]).sum()
    / df["weight"].sum()
)


# --------------------------------------------------
# STEP 6: Calculate weighted standard error
# SE = sqrt(1 / sum(w))
# --------------------------------------------------

weighted_se = math.sqrt(1 / df["weight"].sum())


# --------------------------------------------------
# STEP 7: Calculate normalized deviations
# abs(x - weighted_mean) / uncertainty_g
# --------------------------------------------------

df["normalized_deviation"] = (
    (df["value_g"] - weighted_mean).abs()
    / df["uncertainty_g"]
)


# --------------------------------------------------
# STEP 8: Identify the run with the largest deviation
# --------------------------------------------------

largest_index = df["normalized_deviation"].idxmax()
largest_run = df.loc[largest_index]


# --------------------------------------------------
# STEP 9: Round numeric outputs to 6 decimal places
# --------------------------------------------------

numeric_columns = [
    "value_g",
    "uncertainty_g",
    "weight",
    "normalized_deviation"
]

df[numeric_columns] = df[numeric_columns].round(6)

weighted_mean = round(weighted_mean, 6)
weighted_se = round(weighted_se, 6)

print("\nFINAL CLEANED DATA")
print(
    df[
        [
            "run_id",
            "value_g",
            "uncertainty_g",
            "weight",
            "normalized_deviation"
        ]
    ].to_string(index=False)
)

print("\nWEIGHTED STATISTICS")
print(f"Weighted mean (g): {weighted_mean:.6f}")
print(f"Weighted standard error (g): {weighted_se:.6f}")

print("\nLARGEST NORMALIZED DEVIATION")
print(f"Run ID: {largest_run['run_id']}")
print(
    "Absolute normalized deviation: "
    f"{largest_run['normalized_deviation']:.6f}"
)

# --------------------------------------------------
# STEP 10: Save the cleaned results
# --------------------------------------------------

df.to_csv("cleaned_measurements.csv", index=False)

print("\nCleaned data saved to cleaned_measurements.csv")
# STEP 11: Automated validation checks

assert len(df) == 5, "Expected exactly 5 valid runs"
assert "R04" not in df["run_id"].values, "Failed QC run was not removed"
assert (df["uncertainty_g"] > 0).all(), "Invalid uncertainty detected"
assert df["value_g"].notna().all(), "Missing converted values detected"
assert df["uncertainty_g"].notna().all(), "Missing converted uncertainties detected"
assert (df["uncertainty_g"] > 0).all(), "Invalid uncertainty detected"

assert abs(
    df["value_g"].loc[df["run_id"] == "R01"].iloc[0] - 1.0
) < 1e-9, "R01 conversion failed"

assert abs(
    df["value_g"].loc[df["run_id"] == "R03"].iloc[0] - 0.98
) < 1e-9, "R03 conversion failed"

print("\nAll automated validation checks passed.")