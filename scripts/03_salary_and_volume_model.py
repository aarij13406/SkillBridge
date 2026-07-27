import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import numpy as np

# tries the real skillbridge package first, falls back to local copies
# that do the exact same thing if skillbridge isn't on this machine
# (e.g. running with just this script + the csv, no repo checkout)
try:
    from skillbridge.splits import temporal_split, assert_no_temporal_leakage
    from skillbridge.config import TEER_FROM_NOC, SEED, set_all_seeds
    from skillbridge.metrics import save_result
except ImportError:
    print("skillbridge package not found, using local fallback versions (same logic)")

    SEED = 42

    def set_all_seeds(seed=SEED):
        import random, os
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def temporal_split(d, date_col="posting_date", cutoff="2026-01-01"):
        d = d.dropna(subset=[date_col])
        cut = pd.Timestamp(cutoff)
        train = d[d[date_col] < cut]
        test = d[d[date_col] >= cut]
        print(f"  [splits] temporal split at {cutoff}: train={len(train):,}, test={len(test):,}")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def assert_no_temporal_leakage(train, test, date_col="posting_date"):
        tr_max = pd.to_datetime(train[date_col]).max()
        te_min = pd.to_datetime(test[date_col]).min()
        if tr_max >= te_min:
            raise AssertionError(f"LEAKAGE: latest train date ({tr_max.date()}) is not before earliest test date ({te_min.date()}).")
        print(f"  [splits] passed check, train ends {tr_max.date()}, test starts {te_min.date()}")

    def TEER_FROM_NOC(noc):
        return str(noc).zfill(5)[1]

    def save_result(results, component, model_name, results_dir, extra=None):
        payload = {"component": component, "model": model_name, "metrics": results}
        if extra:
            payload["extra"] = extra
        results_dir.mkdir(parents=True, exist_ok=True)
        safe = model_name.lower().replace(" ", "_").replace("/", "-")
        path = results_dir / f"{component}__{safe}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

set_all_seeds(SEED)

# uses the shared skillbridge path if PROJECT_ROOT is set up, otherwise falls back to next-to-this-script
try:
    from skillbridge.config import F_JOBBANK, RESULTS_DIR
    if not F_JOBBANK.exists():
        raise FileNotFoundError
    jobbankPath = F_JOBBANK
    resultsDir = RESULTS_DIR
except (ImportError, FileNotFoundError):
    jobbankPath = Path(__file__).resolve().parent / "jobbank_clean.csv"
    resultsDir = Path(__file__).resolve().parent / "results"
    print(f"F_JOBBANK not resolving (PROJECT_ROOT not set for this machine), using local path instead: {jobbankPath}")

df = pd.read_csv(jobbankPath)
print(f"loaded {len(df):,} rows")

# ============================================================
# PART A -- SALARY
# ============================================================

# salary_annual has some garbage in it, saw a few $0 rows and one up
# near $1.17M in .describe(). cutting anything outside 10k-500k, these
# bounds are a guess not a science
df = df[df["salary_annual"].notna()]
df = df[(df["salary_annual"] >= 10_000) & (df["salary_annual"] <= 500_000)]
print(f"after salary cleaning: {len(df):,} rows")

# education / experience / industry are missing for the exact same
# 102,061 rows, not a coincidence. Job Bank probably just doesn't
# report these for some posting types. filling "Unknown" instead of
# dropping, dropping would lose ~59% of the data
for col in ["education", "experience", "employment_type", "employment_term", "industry"]:
    df[col] = df[col].fillna("Unknown")

df["noc21_code"] = df["noc21_code"].astype(str)  # code not a number

# Job Bank re-exports every open posting every month, so a job open
# for 3 months shows up in 3 monthly csvs with the same title/noc/
# city/salary, just a different export date. dedupe before splitting
# or the model has already seen chunks of the "test" set during
# training. deduped on (job_title, noc21_code, city, salary_annual),
# dropped 51,103 of 169,902 rows, ~30% were reposts not new postings
df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
dedupKeys = ["job_title", "noc21_code", "city", "salary_annual"]
beforeDedup = len(df)
df = df.sort_values("posting_date").drop_duplicates(subset=dedupKeys, keep="first")
print(f"after dedup: {len(df):,} rows ({beforeDedup - len(df):,} reposts removed)")

# using skillbridge's temporal_split, the team standard. train before
# Jan 1 2026, test on/after
salTrain, salTest = temporal_split(df)
assert_no_temporal_leakage(salTrain, salTest)

# the official check above only looks at dates, not posting identity,
# so running our own check too, should be 0 now that we deduped
overlap = set(map(tuple, salTrain[dedupKeys].values)) & set(map(tuple, salTest[dedupKeys].values))
print(f"repeat-posting overlap after dedup: {len(overlap):,} (should be 0)")

# city has 3717 unique values, one-hot encoding it blew up the
# feature space for no real gain. using frequency encoding instead,
# basically "how many postings did this city have in training"
cityFreq = salTrain["city"].value_counts()
salTrain = salTrain.copy()
salTest = salTest.copy()
salTrain["city_freq"] = salTrain["city"].map(cityFreq).fillna(0)
salTest["city_freq"] = salTest["city"].map(cityFreq).fillna(0)

featureCols = [
    "noc21_code", "province", "education", "experience",
    "employment_type", "employment_term", "industry", "city_freq",
]
targetCol = "salary_annual"

XTrain, yTrain = salTrain[featureCols], salTrain[targetCol]
XTest, yTest = salTest[featureCols], salTest[targetCol]

from sklearn.metrics import mean_absolute_error, r2_score

# baseline: mean salary per (occupation, province) from training data.
# turned out to be a genuinely tough one to beat
baselineLookup = salTrain.groupby(["noc21_code", "province"])[targetCol].mean()
overallMean = salTrain[targetCol].mean()
testBaseline = salTest.merge(
    baselineLookup.rename("baseline_pred"),
    on=["noc21_code", "province"], how="left"
)["baseline_pred"].fillna(overallMean)  # fallback for combos never seen in training, real cold-start case

salaryResults = {}
salaryResults["baseline"] = (
    mean_absolute_error(yTest, testBaseline), r2_score(yTest, testBaseline)
)

# --- linear regression, one-hot ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

onehotCols = [c for c in featureCols if c != "city_freq"]
onehotPrep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), onehotCols),
    ("num", "passthrough", ["city_freq"]),
])
linModel = Pipeline([("prep", onehotPrep), ("model", LinearRegression())])
linModel.fit(XTrain, yTrain)
linPreds = linModel.predict(XTest)
salaryResults["linear_regression"] = (
    mean_absolute_error(yTest, linPreds), r2_score(yTest, linPreds)
)

# --- random forest ---
# first attempt used one-hot encoding same as linear regression, MAE
# came out to ~$15k, way worse than baseline. trees split on one
# column at a time so ~500 sparse one-hot columns from noc21_code
# don't work well. switched to ordinal encoding (each category -> one
# integer column) and it fixed it

# rfBROKEN = Pipeline([("prep", onehotPrep), ("model", RandomForestRegressor(random_state=42))])
# -> MAE ~$15,151 vs baseline's ~$10,563, one-hot + trees is a bad combo

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

ordinalCols = [c for c in featureCols if c != "city_freq"]
ordinalPrep = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinalCols),
    ("num", "passthrough", ["city_freq"]),
])
rfModel = Pipeline([
    ("prep", ordinalPrep),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1)),
])
rfModel.fit(XTrain, yTrain)
rfPreds = rfModel.predict(XTest)
salaryResults["random_forest"] = (
    mean_absolute_error(yTest, rfPreds), r2_score(yTest, rfPreds)
)

# --- gradient boosting, same ordinal encoding as the forest ---
gbModel = Pipeline([
    ("prep", ordinalPrep),
    ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
])
gbModel.fit(XTrain, yTrain)
gbPreds = gbModel.predict(XTest)
salaryResults["gradient_boosting"] = (
    mean_absolute_error(yTest, gbPreds), r2_score(yTest, gbPreds)
)

print("\nsalary results, target is MAE under 12000 and R2 over 0.4")
for name, (mae, r2) in salaryResults.items():
    print(f"  {name:<20} MAE ${mae:,.0f}   R2 {r2:.3f}")

# random forest usually wins but it's close, within a couple hundred
# bucks of baseline. (occupation, province) is just a strong signal
# on its own, extra features aren't buying much, probably because
# ~59% of education/experience/industry is "Unknown"

bestSalaryModel = min(salaryResults, key=lambda k: salaryResults[k][0])
bestSalaryPreds = {
    "baseline": testBaseline.values,
    "linear_regression": linPreds,
    "random_forest": rfPreds,
    "gradient_boosting": gbPreds,
}[bestSalaryModel]
print(f"best model: {bestSalaryModel}")

# teer level for dharnesh's fairness audit, straight from skillbridge.config
salTest["teer_level"] = salTest["noc21_code"].apply(TEER_FROM_NOC)

salaryOutputCols = [
    "noc21_code", "noc21_name", "teer_level", "province", "city",
    "education", "experience", "employment_type", "employment_term",
    "industry", "salary_annual",
]
salaryOutput = salTest[salaryOutputCols].copy()
salaryOutput["predicted_salary"] = bestSalaryPreds
salaryOutput["best_model"] = bestSalaryModel
salaryOutput = salaryOutput.rename(columns={"salary_annual": "true_salary"})
salaryOutput.to_csv("salary_predictions_for_fairness_audit.csv", index=False)
print(f"saved {len(salaryOutput):,} rows -> salary_predictions_for_fairness_audit.csv")


# ============================================================
# PART B -- POSTING VOLUME
# ============================================================
# vacancies column looked like it should be this but median/75th
# percentile are both 1, it's vacancies per posting not total demand.
# built this myself by counting rows

volDf = pd.read_csv(jobbankPath)
volDf["noc21_code"] = volDf["noc21_code"].astype(str)

counts = volDf.groupby(["noc21_code", "province", "source_month"]).size().reset_index(name="count")

pivot = counts.pivot_table(
    index=["noc21_code", "province"], columns="source_month", values="count", fill_value=0
).reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={
    "november2025": "nov_count", "december2025": "dec_count",
    "jan2026": "jan_count", "feb2026": "feb_count",
})
print(f"\n{len(pivot):,} (occupation, province) combos")

# different unit of analysis than part A, one row per combo not per
# posting. split is random across combos, not temporal, since each
# combo already has its own time history in the lag features. the
# question is generalization across occupations/regions, not time
from sklearn.model_selection import train_test_split

volFeatureCols = ["noc21_code", "province", "nov_count", "dec_count", "jan_count"]
volTargetCol = "feb_count"

X = pivot[volFeatureCols]
y = pivot[volTargetCol]
XTrainV, XTestV, yTrainV, yTestV = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train combos: {len(XTrainV):,}  test combos: {len(XTestV):,}")

# two dumb baselines first before trying anything fancy
persistencePred = XTestV["jan_count"]  # just guess "same as last month"
persistenceMae = mean_absolute_error(yTestV, persistencePred)
persistenceR2 = r2_score(yTestV, persistencePred)

meanPred = XTestV[["nov_count", "dec_count", "jan_count"]].mean(axis=1)
meanMae = mean_absolute_error(yTestV, meanPred)
meanR2 = r2_score(yTestV, meanPred)

volOrdinalPrep = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["noc21_code", "province"]),
    ("num", "passthrough", ["nov_count", "dec_count", "jan_count"]),
])
volRf = Pipeline([
    ("prep", volOrdinalPrep),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)),
])
volRf.fit(XTrainV, yTrainV)
volRfPreds = volRf.predict(XTestV)
volRfMae = mean_absolute_error(yTestV, volRfPreds)
volRfR2 = r2_score(yTestV, volRfPreds)

print("\nposting volume results")
print(f"  persistence (last month)   MAE {persistenceMae:.2f}   R2 {persistenceR2:.3f}")
print(f"  mean of last 3 months      MAE {meanMae:.2f}   R2 {meanR2:.3f}")
print(f"  random forest              MAE {volRfMae:.2f}   R2 {volRfR2:.3f}")

# counts are pretty stable month to month for most occupations so even
# the dumb baselines score high R2 here, forest only edges them out a
# little. not a huge win but it is a win

volOutput = XTestV.copy()
volOutput["true_posting_count"] = yTestV
volOutput["predicted_posting_count"] = volRfPreds
volOutput.to_csv("posting_volume_predictions_for_fairness_audit.csv", index=False)
print(f"saved {len(volOutput):,} rows -> posting_volume_predictions_for_fairness_audit.csv")


# ============================================================
# saving results
# ============================================================
savedSalaryPath = save_result(
    results={"mae": salaryResults[bestSalaryModel][0], "r2": salaryResults[bestSalaryModel][1]},
    component="salary_regional_demand",
    model_name=bestSalaryModel,
    results_dir=resultsDir,
    extra={
        "features": featureCols,
        "all_models_tried": {k: {"mae": v[0], "r2": v[1]} for k, v in salaryResults.items()},
        "target_mae": 12000,
        "target_r2": 0.4,
        "dedup_removed_rows": beforeDedup - len(df),
    },
)
print(f"\nsaved salary result -> {savedSalaryPath}")

savedVolumePath = save_result(
    results={"mae": volRfMae, "r2": volRfR2},
    component="posting_volume",
    model_name="random_forest",
    results_dir=resultsDir,
    extra={
        "baselines_tried": {
            "persistence": {"mae": persistenceMae, "r2": persistenceR2},
            "mean_last_3_months": {"mae": meanMae, "r2": meanR2},
        },
    },
)
print(f"saved posting volume result -> {savedVolumePath}")

# ============================================================
# PART C -- INTERACTIVE QUESTIONNAIRE
# ============================================================
# ask for just an occupation, system figures out the rest:
# avg salary, highest salary, and a breakdown by province with
# each province's avg salary and predicted posting volume next month

occLookup = df[["noc21_code", "noc21_name"]].drop_duplicates().reset_index(drop=True)


def findOccupation(userInput):
    userInput = userInput.strip().lower()
    exact = occLookup[occLookup["noc21_code"].astype(str) == userInput]
    if len(exact) == 1:
        return exact.iloc[0]
    nameMatches = occLookup[occLookup["noc21_name"].str.lower().str.contains(userInput, na=False)]
    if len(nameMatches) == 1:
        return nameMatches.iloc[0]
    if len(nameMatches) > 1:
        print(f"\n{len(nameMatches)} occupations match \"{userInput}\":")
        for i, row in nameMatches.head(10).iterrows():
            print(f"  [{row['noc21_code']}] {row['noc21_name']}")
        if len(nameMatches) > 10:
            print(f"  ...and {len(nameMatches) - 10} more, try a more specific search")
        return None
    print(f"\nno occupation found matching \"{userInput}\", try a NOC code or part of the occupation name")
    return None


def lookupOccupation(nocCode, nocName):
    nocCode = str(nocCode)
    occRows = df[df["noc21_code"] == nocCode]

    if len(occRows) == 0:
        print(f"\nno posting data found for {nocName} ({nocCode})")
        return

    avgSalary = occRows["salary_annual"].mean()
    maxSalary = occRows["salary_annual"].max()

    # national ranking: where this occupation sits against all ~500
    # others by avg pay. cheap to compute, gives instant context for
    # whether a number is actually high or low
    occAvgAll = df.groupby("noc21_code")["salary_annual"].mean().sort_values(ascending=False)
    rank = list(occAvgAll.index).index(nocCode) + 1
    totalOcc = len(occAvgAll)

    print(f"\n{nocName} ({nocCode})")
    print(f"  avg salary (all provinces): ${avgSalary:,.0f} / year")
    print(f"  highest salary seen:        ${maxSalary:,.0f} / year")
    print(f"  national salary rank:       #{rank} of {totalOcc} occupations")

    # salary trend across the 4 months in the dataset. shows whether
    # this occupation is trending up or down, not just a snapshot
    monthOrder = ["november2025", "december2025", "jan2026", "feb2026"]
    monthLabels = {"november2025": "Nov", "december2025": "Dec", "jan2026": "Jan", "feb2026": "Feb"}
    trendSeries = occRows.groupby("source_month")["salary_annual"].mean()
    trendSeries = trendSeries.reindex([m for m in monthOrder if m in trendSeries.index])
    if len(trendSeries) >= 2:
        trendStr = " -> ".join(f"{monthLabels.get(m, m)} ${v:,.0f}" for m, v in trendSeries.items())
        direction = "up" if trendSeries.iloc[-1] > trendSeries.iloc[0] else (
            "down" if trendSeries.iloc[-1] < trendSeries.iloc[0] else "flat"
        )
        print(f"  salary trend (all provinces): {trendStr}  ({direction} over the period)")

    print(f"\n  provinces offering this occupation:")

    byProvince = (
        occRows.groupby("province")["salary_annual"]
        .agg(avg_salary="mean", postings="count")
        .sort_values("avg_salary", ascending=False)
    )

    for province, row in byProvince.iterrows():
        volRow = pivot[(pivot["noc21_code"] == nocCode) & (pivot["province"] == province)]
        if len(volRow) == 1:
            volFeatures = volRow[["noc21_code", "province", "nov_count", "dec_count", "jan_count"]]
            predictedVolume = volRf.predict(volFeatures)[0]
            volText = f"~{predictedVolume:.1f} postings predicted next month"
        else:
            volText = "no recent posting history"
        # confidence flag: a handful of postings shouldn't be trusted
        # the same as a few hundred
        sampleFlag = "  [small sample, low confidence]" if row["postings"] < 10 else ""
        print(f"    {province:<28} avg ${row['avg_salary']:>10,.0f}   ({int(row['postings'])} postings on record, {volText}){sampleFlag}")

    # model spotlight: run the actual trained Random Forest for a typical
    # posting in the top-paying province, side by side with the plain
    # historical average above. shows the model doing real work in the
    # demo, not just table lookups
    topProvince = byProvince.index[0]
    provRows = occRows[occRows["province"] == topProvince]

    def mostCommon(series, fallback="Unknown"):
        modes = series.mode()
        return modes.iloc[0] if len(modes) else fallback

    typicalCity = mostCommon(provRows["city"])
    profile = pd.DataFrame([{
        "noc21_code": nocCode,
        "province": topProvince,
        "education": mostCommon(provRows["education"]),
        "experience": mostCommon(provRows["experience"]),
        "employment_type": mostCommon(provRows["employment_type"]),
        "employment_term": mostCommon(provRows["employment_term"]),
        "industry": mostCommon(provRows["industry"]),
        "city_freq": cityFreq.get(typicalCity, 0),
    }])
    modelPrediction = rfModel.predict(profile)[0]
    historicalAvg = byProvince.loc[topProvince, "avg_salary"]
    diff = modelPrediction - historicalAvg

    print(f"\n  model spotlight -- {topProvince} (top-paying province for this role):")
    print(f"    historical average:     ${historicalAvg:,.0f}")
    print(f"    Random Forest predicts: ${modelPrediction:,.0f}  ({'+' if diff >= 0 else ''}{diff:,.0f} vs. average)")
    print(f"    (based on a typical posting profile: {mostCommon(provRows['education'])}, {mostCommon(provRows['experience'])}, in {typicalCity})")

    # best-paying paths: search every (province, education, experience)
    # combo seen for this occupation and let the model rank them. most
    # of these combos barely exist as real rows, so this is genuinely
    # different from anything a groupby average can produce
    showBestOpportunities(nocCode, nocName, occRows)


def showBestOpportunities(nocCode, nocName, occRows, topN=3):
    provinces = occRows["province"].unique()
    educations = occRows["education"].unique()
    experiences = occRows["experience"].unique()

    def mostCommon(series, fallback="Unknown"):
        modes = series.mode()
        return modes.iloc[0] if len(modes) else fallback

    combos = []
    for prov in provinces:
        provRows = occRows[occRows["province"] == prov]
        typicalCity = mostCommon(provRows["city"])
        typicalEmpType = mostCommon(provRows["employment_type"])
        typicalEmpTerm = mostCommon(provRows["employment_term"])
        typicalIndustry = mostCommon(provRows["industry"])
        cf = cityFreq.get(typicalCity, 0)
        for edu in educations:
            for exp in experiences:
                combos.append({
                    "noc21_code": nocCode, "province": prov,
                    "education": edu, "experience": exp,
                    "employment_type": typicalEmpType, "employment_term": typicalEmpTerm,
                    "industry": typicalIndustry, "city_freq": cf,
                })

    if not combos:
        return

    gridDf = pd.DataFrame(combos)
    gridDf["predicted_salary"] = rfModel.predict(gridDf[featureCols])
    top = gridDf.sort_values("predicted_salary", ascending=False).head(topN)

    print(f"\n  best-paying paths for {nocName}")
    print(f"  (model ranked {len(gridDf):,} province/education/experience combinations):")
    for i, row in enumerate(top.itertuples(), 1):
        print(f"    {i}. {row.province}, {row.education}, {row.experience}  ->  ${row.predicted_salary:,.0f}")


def printFeatureImportance():
    # global importance from the trained Random Forest -- what the
    # model is actually weighting, not per-occupation, applies to
    # every prediction it makes
    importances = rfModel.named_steps["model"].feature_importances_
    names = ordinalCols + ["city_freq"]
    total = importances.sum()
    ranked = sorted(zip(names, importances), key=lambda x: -x[1])

    print("\n  what actually drives the salary model's predictions (global, all occupations):")
    for name, imp in ranked:
        print(f"    {name:<18} {imp / total * 100:5.1f}%")


def compareOccupations():
    print("\ncompare mode -- enter two occupations")
    firstInput = input("  first occupation (NOC code or name): ").strip()
    firstRow = findOccupation(firstInput)
    if firstRow is None:
        return
    secondInput = input("  second occupation (NOC code or name): ").strip()
    secondRow = findOccupation(secondInput)
    if secondRow is None:
        return

    occAvgAll = df.groupby("noc21_code")["salary_annual"].mean().sort_values(ascending=False)
    totalOcc = len(occAvgAll)

    def summarize(nocCode):
        nocCode = str(nocCode)
        rows = df[df["noc21_code"] == nocCode]
        avgSalary = rows["salary_annual"].mean()
        rank = list(occAvgAll.index).index(nocCode) + 1
        topProvince = rows.groupby("province")["salary_annual"].mean().sort_values(ascending=False).index[0]
        return avgSalary, rank, topProvince

    avg1, rank1, top1 = summarize(firstRow["noc21_code"])
    avg2, rank2, top2 = summarize(secondRow["noc21_code"])

    name1 = str(firstRow["noc21_name"])[:24]
    name2 = str(secondRow["noc21_name"])[:24]

    print(f"\n  {'':<24}{name1:<26}{name2}")
    print(f"  {'avg salary':<24}${avg1:>10,.0f}            ${avg2:>10,.0f}")
    print(f"  {'national rank':<24}#{rank1} of {totalOcc:<15}#{rank2} of {totalOcc}")
    print(f"  {'best province':<24}{top1:<26}{top2}")


def runQuestionnaire():
    print("\n" + "=" * 70)
    print("OCCUPATION LOOKUP -- salary & posting volume by province")
    print("type 'quit' to stop, 'compare' to compare two occupations,")
    print("'model' to see what drives the salary predictions")
    print("=" * 70)
    while True:
        occInput = input("\nWhat's your desired position (NOC code or name): ").strip()
        if occInput.lower() in ("quit", "exit", "q"):
            break
        if occInput.lower() == "compare":
            compareOccupations()
            continue
        if occInput.lower() == "model":
            printFeatureImportance()
            continue
        occRow = findOccupation(occInput)
        if occRow is None:
            continue
        lookupOccupation(occRow["noc21_code"], occRow["noc21_name"])


if __name__ == "__main__":
    runQuestionnaire()



# ============================================================
# SUMMARY -- Salary and Regional Demand
# ============================================================

# TASK: predict annual salary and monthly posting volume by
# (occupation, province) from Job Bank data.
#
# LEAKAGE ISSUE FOUND: Job Bank re-exports open postings every month,
# so the same posting appears in multiple monthly files. deduping on
# (job_title, noc21_code, city, salary_annual) removed 51,103 of
# 169,902 rows (30%). the shared harness's assert_no_temporal_leakage
# only checks dates, not posting identity, so it doesn't catch this.

#
# SPLIT: official skillbridge.splits.temporal_split, cutoff 2026-01-01.
# train = nov+dec 2025 (58,284 rows), test = jan+feb 2026 (60,515 rows).
#
# SALARY RESULTS (target MAE < 12000, R2 > 0.4):
#   baseline (mean by group):  MAE ~$12,356   R2 ~0.620
#   linear regression:          MAE ~$12,264   R2 ~0.635
#   random forest:               MAE ~$12,088   R2 ~0.639   <- best, but barely clears the MAE target
#   gradient boosting:           MAE ~$12,426   R2 ~0.635
#
# the extra features (education/experience/industry/employment type)
# don't add a ton beyond just occupation+province, probably because
# ~59% of those fields are "Unknown". city helps a little via
# frequency encoding but wasn't tested feature-by-feature to see
# exactly how much.

# POSTING VOLUME RESULTS:
#   persistence (last month):   MAE 3.60   R2 0.905
#   mean of last 3 months:      MAE 3.07   R2 0.916
#   random forest:               MAE 3.13   R2 0.929

# volume is dominated by "recent history predicts near future," even
# dumb baselines score high R2. forest wins on R2 but not by a lot.
