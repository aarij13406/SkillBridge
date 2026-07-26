import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # TO KNOW WHERE THE ACTUAL FILES ARE 

import pandas as pd
from skillbridge.config import CLEAN_DIR

cops = pd.read_csv(CLEAN_DIR / "cops_projections_clean.csv") # taking only dataset which I need to work 
#lets clean the data first , wrote till line 56 just to find NaN -_- 
cops = cops[cops["noc_code"].str.len() == 5]
#print(cops.head()) # cheaking the labels to see which feature i have to work with 

# the above lines didn't work as its not showing all the feature names hmm...

# lets try to print the names of the features
#print(cops.columns.tolist()) 

# got it so now there are 13 features and what am seeking here is future condition and am able to see it. .... nice 

#time to see whats inside the future_condition 
futureCondition = cops["future_conditions"].value_counts()

#print(futureCondition)

# all I got is pain :-(   ......  only 17 surplus thats gonna be a problem as the model will tend to bias on balance ... hmmmm ... ahhhhh

# first there are 5 state here Balance , Moderate risk of Shortage , Strong risk of Shortage  ,Moderate risk of Surplus and Strong risk of Surplus 
# but as per plan I am gonna put them in three class Shortage  , Surplus and balance

changedName = {
    "Strong risk of Shortage": "Shortage",
    "Moderate risk of Shortage": "Shortage",
    "Balance": "Balance",
    "Moderate risk of Surplus": "Surplus",
    "Strong risk of Surplus": "Surplus",
}

cops["new_label"] = cops["future_conditions"].map(changedName)
#print(cops["new_label"].value_counts())
#time to check with actuall occupations , iam trying with jobbank_clean.csv as it has 174K rows

jobs = pd.read_csv(CLEAN_DIR / "jobbank_clean.csv")
jobsheadings = jobs.columns.tolist()
#print (jobsheadings)
#there is noc code so i can group by that 
postingCounts = jobs.groupby("noc21_code").size()
#print(postingCounts) 
#nice 507 different values across 174k rows 

#since this dont have same number of digits lets add zeroes in front 

postingCounts.index = postingCounts.index.astype(str).str.zfill(5) # this fills with zeros to make all 5 digits

#lets make a new class mapped with this 
#print (postingCounts.head(10))
cops["posting"] = cops["noc_code"].map(postingCounts) # just mapping the noc_code with postingcounts
#print(cops[["noc_code", "new_label", "posting"]].head(10))

#even after removing NaN still its coming becuase its real not available of postings lets remove them too

cops["posting"] = cops["posting"].fillna(0)

#print(len(cops))

# now its 516, hmmm lets drops these rows for label gap for new_label

cops = cops.dropna(subset=["new_label"])

#print(len(cops))
# bruh already less data now only 485 aahh

#print(cops[["noc_code", "new_label", "posting"]].head(10))
#print(len(cops))

#now the data is clean time to train some model but before that iam gonna take a floor value 

mojorityGuess = cops["new_label"].mode()[0] 
#print(mojorityGuess)
accuracy = (cops["new_label"] == mojorityGuess).mean()

#print(accuracy)

# as expected balanced is the mode and the mean is 0.752 ... hmmmm I have a bad feeling on this just a waste guesser got 75 percentage so out model should be much better that this 


# time to train models even though we know its gonna perform barely high that this guesser ... PAINNNN 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # first let me try the LogisticRegression alone
from sklearn.metrics import accuracy_score, f1_score


X = cops[["posting"]]
y = cops["new_label"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # common seed 42 and test_size 0.2 because we have less DATA TO TRAIN WITH...... AHHH


#model = LogisticRegression()
#model.fit(Xtrain, ytrain)
#predicts = model.predict(Xtest)


#print("accuracy:", accuracy_score(ytest, predicts))
#print("macro f1:", f1_score(ytest, predicts, average="macro"))
# as expected accuracy is same as the waste lazy guesser also the macro f1 is 0.28 sheeessssh
#this shows the model is bad , because its leanred to predict the balance

#lets do the same but with class weight for balance 

# model = LogisticRegression(class_weight="balanced")
# model.fit(Xtrain, ytrain)
# predicts = model.predict(Xtest)


# print("accuracy:", accuracy_score(ytest, predicts))
# print("macro f1:", f1_score(ytest, predicts, average="macro"))

#completely waste of time but the lesson imbalance need a real feature with value


#lets get a good feature from the oasis same things again 

oasis = pd.read_csv(CLEAN_DIR / "oasis_descriptors_long.csv", dtype={"noc_code": str})

avgRating = oasis.groupby("noc_code")["rating"].mean()

cops["avg_rating"] = cops["noc_code"].map(avgRating)

#print(cops[["noc_code", "new_label", "posting", "avg_rating"]].head(10))

#now we have a rating feature so again lets try logestic regression 

# X = cops[["posting", "avg_rating"]] #added one extra feature , no other changes
# y = cops["new_label"]

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# model = LogisticRegression()
# model.fit(Xtrain, ytrain)
# predicts = model.predict(Xtest)

# print("accuracy:", accuracy_score(ytest, predicts))
# print("macro f1:", f1_score(ytest, predicts, average="macro"))

#PAIN is permanent .. again waste of time hmmm i dont wanna repeat this lets try to find a new way from internet


#this time first lets scale it 


# X = cops[["posting", "avg_rating"]] #added one extra feature , no other changes
# y = cops["new_label"]

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler # nice way to scale the data, still needed below for the CV loops

# scaler = StandardScaler()
# Xtrain = scaler.fit_transform(Xtrain)
# Xtest = scaler.transform(Xtest)


# model = LogisticRegression()
# model.fit(Xtrain, ytrain)
# predicts = model.predict(Xtest)

#print("accuracy:", accuracy_score(ytest, predicts))
#print("macro f1:", f1_score(ytest, predicts, average="macro"))
#old single-split result, replaced by the cross-validated version further down -- keeping commented as history

#now we are talking atlest now its better that a random guesser 76 percent accuracy '-' ... lets try to get a good feature


# from sklearn.feature_selection import mutual_info_classif
# print(mutual_info_classif(cops[["posting", "avg_rating"]], cops["new_label"]))

# hmm yet not good as both the features we took scores low 

#this time we only take core rated 4 or more 
# coreFraction = oasis.groupby("noc_code")["rating"].apply(lambda r: (r >= 4).mean())
# cops["core_fraction"] = cops["noc_code"].map(coreFraction)

# from sklearn.feature_selection import mutual_info_classif
# print(mutual_info_classif(cops[["posting", "avg_rating", "core_fraction"]], cops["new_label"]))


# 4th failure [0.03250943 0.03824089 0.0230531 ] the core is lowest hmmm , done for today aahh



# NEW SECTION -- picking back up: employment_growth + cross-validation


# feature: employment_growth (already in cops, no merge needed) 
# print(cops["employment_growth"].isna().sum())   # check for gaps before using it
# cops["employment_growth"] = cops["employment_growth"].fillna(0)


coreFraction = oasis.groupby("noc_code")["rating"].apply(lambda r: (r >= 4).mean())
cops["core_fraction"] = cops["noc_code"].map(coreFraction)

#  cross-validated evaluation using the team's shared stratified_folds() 
from skillbridge.splits import stratified_folds
import numpy as np

featureCols = ["posting", "avg_rating", "core_fraction", "employment_growth"]
X = cops[featureCols]
y = cops["new_label"].values

foldScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    scalerCV = StandardScaler()
    XtrainCV = scalerCV.fit_transform(XtrainCV)
    XtestCV = scalerCV.transform(XtestCV)

    modelCV = LogisticRegression()
    modelCV.fit(XtrainCV, ytrainCV)
    predsCV = modelCV.predict(XtestCV)

    foldScores.append(f1_score(ytestCV, predsCV, average="macro"))

print("logistic regression cv macro f1 mean:", np.mean(foldScores))


#hmmm i think the classifer which i choose for initial might have this limit , lets change to random forest
#  trying Random Forest, same folds, same features -- no scaler needed for trees 
from sklearn.ensemble import RandomForestClassifier

rfScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    modelRF = RandomForestClassifier(random_state=42)
    modelRF.fit(XtrainCV, ytrainCV)
    predsRF = modelRF.predict(XtestCV)

    rfScores.append(f1_score(ytestCV, predsRF, average="macro"))

print("random forest cv macro f1 mean:", np.mean(rfScores))

# trying Gradient Boosting, same folds, same features -- also no scaler needed for trees
from sklearn.ensemble import GradientBoostingClassifier

gbScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    modelGB = GradientBoostingClassifier(random_state=42)
    modelGB.fit(XtrainCV, ytrainCV)
    predsGB = modelGB.predict(XtestCV)

    gbScores.append(f1_score(ytestCV, predsGB, average="macro"))

print("gradient boosting cv macro f1 per fold:", gbScores)
print("gradient boosting cv macro f1 mean:", np.mean(gbScores))

# trying a small MLP (neural net), same folds, same features -- needs scaling like logistic regression did 
from sklearn.neural_network import MLPClassifier

mlpScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    scalerCV = StandardScaler()
    XtrainCV = scalerCV.fit_transform(XtrainCV)
    XtestCV = scalerCV.transform(XtestCV)

    modelMLP = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
    modelMLP.fit(XtrainCV, ytrainCV)
    predsMLP = modelMLP.predict(XtestCV)

    mlpScores.append(f1_score(ytestCV, predsMLP, average="macro"))

print("mlp cv macro f1 per fold:", mlpScores)
print("mlp cv macro f1 mean:", np.mean(mlpScores))
### niceeee we got atlest close to 0.50 , now the MLP gives 0.402  so i think this is our winner
### now the imbalance study on top of it , first SMOTE

from imblearn.over_sampling import SMOTE

smoteScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    smote = SMOTE(random_state=42, k_neighbors=3) #surplus has very few rows so keeping k small
    XtrainCV, ytrainCV = smote.fit_resample(XtrainCV, ytrainCV) #only train gets the fake rows, test stays real

    scalerCV = StandardScaler()
    XtrainCV = scalerCV.fit_transform(XtrainCV)
    XtestCV = scalerCV.transform(XtestCV)

    modelMLP = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
    modelMLP.fit(XtrainCV, ytrainCV)
    predsMLP = modelMLP.predict(XtestCV)

    smoteScores.append(f1_score(ytestCV, predsMLP, average="macro"))

print("mlp + smote cv macro f1 per fold:", smoteScores)
print("mlp + smote cv macro f1 mean:", np.mean(smoteScores))

### second imbalance technique: class weighting
### MLP has no class_weight option at all, so this one has to run on random forest instead

fromRfScores = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    modelRFCW = RandomForestClassifier(random_state=42, class_weight="balanced")
    modelRFCW.fit(XtrainCV, ytrainCV)
    predsRFCW = modelRFCW.predict(XtestCV)

    fromRfScores.append(f1_score(ytestCV, predsRFCW, average="macro"))

print("random forest + class_weight cv macro f1 per fold:", fromRfScores)
print("random forest + class_weight cv macro f1 mean:", np.mean(fromRfScores))

### this is our winner (0.412), now lets do the error analysis on it
### collecting predictions across ALL folds so we get one confusion matrix for the whole dataset

from skillbridge.metrics import classification_report_full

allTrue = []
allPred = []
for trainIdx, testIdx in stratified_folds(X, y, n_folds=5, seed=42):
    XtrainCV, XtestCV = X.iloc[trainIdx], X.iloc[testIdx]
    ytrainCV, ytestCV = y[trainIdx], y[testIdx]

    modelFinal = RandomForestClassifier(random_state=42, class_weight="balanced")
    modelFinal.fit(XtrainCV, ytrainCV)
    predsFinal = modelFinal.predict(XtestCV)

    allTrue.extend(ytestCV)
    allPred.extend(predsFinal)

report = classification_report_full(np.array(allTrue), np.array(allPred), labels=["Shortage", "Balance", "Surplus"])

print("per class breakdown:", report["per_class"])
print("confusion matrix (rows=true, cols=pred, order Shortage/Balance/Surplus):")
print(report["confusion_matrix"])

### saving this the same way everyone else's results are saved, so it lands in results/ next to theirs

from skillbridge.config import RESULTS_DIR
from skillbridge.metrics import save_result

savedPath = save_result(
    results=report,
    component="shortage_classifier",
    model_name="random_forest_class_weight",
    results_dir=RESULTS_DIR,
    extra={
        "features": featureCols,
        "cv_macro_f1_per_fold": fromRfScores,
        "cv_macro_f1_mean": float(np.mean(fromRfScores)),
        "baselines_tried": {
            "logistic_regression": float(np.mean(foldScores)),
            "random_forest": float(np.mean(rfScores)),
            "gradient_boosting": float(np.mean(gbScores)),
            "mlp": float(np.mean(mlpScores)),
            "mlp_smote": float(np.mean(smoteScores)),
        },
        "majority_baseline_accuracy": float(accuracy),
    },
)

print("saved to:", savedPath)


# (summary moved to the very end of this file, after round 2)


# ============================================================
# ROUND 2 -- trying to push past 0.412
# same lesson as before still applies: better features > swapping algorithms.
# so this time: richer features first, then tune the model on top of that.
# ============================================================

from sklearn.feature_selection import mutual_info_classif

# --- new feature 1: salary stats per occupation ---
salaryStats = jobs.groupby("noc21_code")["salary_annual"].agg(["mean", "std"])
salaryStats.index = salaryStats.index.astype(str).str.zfill(5)
salaryStats.columns = ["salary_mean", "salary_std"]

cops["salary_mean"] = cops["noc_code"].map(salaryStats["salary_mean"])
cops["salary_std"] = cops["noc_code"].map(salaryStats["salary_std"])
cops["salary_mean"] = cops["salary_mean"].fillna(cops["salary_mean"].median())
cops["salary_std"] = cops["salary_std"].fillna(0)  # occupations with 1 posting have no spread

# --- new feature 2: OaSIS broken out by category instead of one blended average ---
# averaging Abilities+Knowledge+WorkActivities+Skills+PersonalAttributes together
# was erasing signal -- an occupation low on Knowledge but high on Abilities just
# showed up as "medium" in the blended number, hiding both facts.
categoryAvg = oasis.groupby(["noc_code", "category"])["rating"].mean().unstack("category")
categoryAvg.columns = ["oa_" + c.replace(" ", "_").lower() for c in categoryAvg.columns]
cops = cops.join(categoryAvg, on="noc_code")
categoryCols = list(categoryAvg.columns)
cops[categoryCols] = cops[categoryCols].fillna(cops[categoryCols].median())

# --- new feature 3: log-transform posting count, its raw scale is very skewed ---
cops["posting_log"] = np.log1p(cops["posting"])

# --- check the new candidates before committing to them ---
newFeatureCols = ["posting", "posting_log", "avg_rating", "core_fraction",
                   "employment_growth", "salary_mean", "salary_std"] + categoryCols
print("mutual info on the expanded feature set:")
miScores = mutual_info_classif(cops[newFeatureCols], cops["new_label"], random_state=42)  # pinned, was wobbling run-to-run before
for name, score in zip(newFeatureCols, miScores):
    print(f"  {name}: {score:.4f}")

Xv2 = cops[newFeatureCols].astype("float64")  # force plain numpy floats, arrow-backed dtype otherwise breaks sklearn indexing
yv2 = cops["new_label"].astype(str).to_numpy()  # same arrow-backed issue, on the label column

# --- hyperparameter search on random forest, same folds, same metric ---
from sklearn.model_selection import GridSearchCV, StratifiedKFold

paramGrid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 3, 5],
}

skfV2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    paramGrid,
    scoring="f1_macro",
    cv=skfV2,
)
search.fit(Xv2, yv2)

print("best params:", search.best_params_)
print("best cv macro f1:", search.best_score_)

# --- confusion matrix for this tuned model, out-of-fold across all 5 folds ---
allTrueV2 = []
allPredV2 = []
for trainIdx, testIdx in skfV2.split(Xv2, yv2):
    XtrainV2, XtestV2 = Xv2.iloc[trainIdx], Xv2.iloc[testIdx]
    ytrainV2, ytestV2 = yv2[trainIdx], yv2[testIdx]

    modelV2 = RandomForestClassifier(random_state=42, class_weight="balanced", **search.best_params_)
    modelV2.fit(XtrainV2, ytrainV2)
    predsV2 = modelV2.predict(XtestV2)

    allTrueV2.extend(ytestV2)
    allPredV2.extend(predsV2)

reportV2 = classification_report_full(np.array(allTrueV2), np.array(allPredV2),
                                       labels=["Shortage", "Balance", "Surplus"])
print("round 2 per class breakdown:", reportV2["per_class"])
print("round 2 confusion matrix (rows=true, cols=pred, order Shortage/Balance/Surplus):")
print(reportV2["confusion_matrix"])

savedPathV2 = save_result(
    results=reportV2,
    component="shortage_classifier",
    model_name="random_forest_tuned_v2",
    results_dir=RESULTS_DIR,
    extra={
        "features": newFeatureCols,
        "best_params": search.best_params_,
        "cv_macro_f1_mean": float(search.best_score_),
        "round1_macro_f1_mean": float(np.mean(fromRfScores)),
    },
)
print("saved to:", savedPathV2)

# --- third imbalance technique: cost-sensitive learning, a manually-designed cost matrix ---
# class_weight="balanced" (already tried above) IS a form of cost-sensitive learning --
# it just picks weights automatically, purely from how rare each class is. true
# cost-sensitive learning means WE decide the relative cost of each mistake, based on
# reasoning, not just frequency. here: getting Surplus wrong is worse than its rarity
# alone would suggest -- telling someone a shrinking field is "fine" can genuinely
# mislead a real career decision. so Surplus gets extra weight beyond pure inverse-frequency.
customCostWeights = {"Shortage": 2, "Balance": 1, "Surplus": 6}

costScores = []
allTrueCost = []
allPredCost = []
for trainIdx, testIdx in stratified_folds(Xv2, yv2, n_folds=5, seed=42):
    XtrainCost, XtestCost = Xv2.iloc[trainIdx], Xv2.iloc[testIdx]
    ytrainCost, ytestCost = yv2[trainIdx], yv2[testIdx]

    modelCost = RandomForestClassifier(random_state=42, class_weight=customCostWeights, **search.best_params_)
    modelCost.fit(XtrainCost, ytrainCost)
    predsCost = modelCost.predict(XtestCost)

    costScores.append(f1_score(ytestCost, predsCost, average="macro"))
    allTrueCost.extend(ytestCost)
    allPredCost.extend(predsCost)

print("cost-sensitive (custom weights) cv macro f1 per fold:", costScores)
print("cost-sensitive (custom weights) cv macro f1 mean:", np.mean(costScores))

reportCost = classification_report_full(np.array(allTrueCost), np.array(allPredCost),
                                         labels=["Shortage", "Balance", "Surplus"])
print("cost-sensitive per class breakdown:", reportCost["per_class"])
print("cost-sensitive confusion matrix (rows=true, cols=pred, order Shortage/Balance/Surplus):")
print(reportCost["confusion_matrix"])

savedPathCost = save_result(
    results=reportCost,
    component="shortage_classifier",
    model_name="random_forest_cost_sensitive",
    results_dir=RESULTS_DIR,
    extra={
        "features": newFeatureCols,
        "cost_weights": customCostWeights,
        "cv_macro_f1_mean": float(np.mean(costScores)),
        "round2_balanced_macro_f1_mean": float(search.best_score_),
    },
)
print("saved to:", savedPathCost)


# ============================================================
# SUMMARY -- Labour Shortage Classifier (Irai)
# for the final report, whenever we write it. paste/rewrite from here.
# ============================================================
#
# TASK: predict COPS's 10-year outlook per occupation -- Shortage, Balance,
# or Surplus -- from independent signals only. deliberately excluded COPS's
# own gap/gap_pct columns: that's the arithmetic COPS itself uses to assign
# the label, so using them would be target leakage (the model would just be
# re-deriving the answer instead of learning anything).
#
# DATA: 526 rows in COPS -> 485 usable, after dropping 41 unlabeled rows and
# rows that were category aggregates (e.g. "NOC1_0"), not real occupations.
# label split: Balance 365 (75%), Shortage 103 (21%), Surplus 17 (3.5%).
#
# BASELINE: always guessing "Balance" gets 75.2% accuracy but is useless --
# it never once identifies a Shortage or Surplus occupation. that's exactly
# why macro-F1 (which scores all 3 classes equally, regardless of how common
# each one is) is the metric that matters here, not accuracy.
#
# EVALUATION: 5-fold stratified CV via the team's shared stratified_folds(),
# not a single train/test split -- with only 17 Surplus examples nationwide,
# one split would put ~3 in the test set and the score would swing on luck.
# all figures below are 5-fold means. confusion matrices are pooled
# out-of-fold, so every one of the 485 occupations is predicted exactly once
# by a model that never saw it in training.
#
# ---------- ROUND 1: 4 features ----------
# features: posting, avg_rating, core_fraction, employment_growth
# all individually weak on mutual information (0.02-0.04).
#
# MODEL COMPARISON (identical features, identical folds):
#   logistic regression:  0.332
#   random forest:        0.391
#   gradient boosting:    0.376
#   mlp (neural net):     0.402   <- best of the 4 before any imbalance fix
#
# IMBALANCE STUDY, part 1:
#   MLP + SMOTE:                  0.375  (SMOTE made MLP WORSE)
#   random forest + class_weight: 0.412  (helped, best of round 1)
#   note: MLPClassifier has no class_weight parameter at all, so the
#   class-weighting arm had to run on random forest instead -- a real
#   methodological constraint, not a shortcut.
#   finding: the "right" imbalance fix is not universal. it depends on
#   which model it's paired with.
#
# ---------- ROUND 2: 12 features + tuning ----------
# what changed, and why it mattered:
#   1. split the blended avg_rating back into its 5 OaSIS categories.
#      this was the single biggest win. oa_abilities alone scores ~0.08 on
#      mutual info -- roughly double the blended avg_rating (~0.036).
#      averaging the categories together had been erasing real signal:
#      an occupation low on Knowledge but high on Abilities just showed up
#      as "medium" and both facts were lost.
#   2. added salary stats (mean, spread) from Job Bank -- a source never
#      used in round 1.
#   3. log-transformed the heavily-skewed posting count.
#   4. GridSearchCV over n_estimators / max_depth / min_samples_leaf.
#      chose max_depth=6, n_estimators=400, min_samples_leaf=1.
#      shallower trees than the default won, which makes sense at 485 rows:
#      unrestricted depth was overfitting noise.
#
# IMBALANCE STUDY, part 2 -- cost-sensitive learning:
#   round 2 (class_weight="balanced"):     0.497   <- FINAL, best overall
#   round 2 (custom cost weights 2/1/6):   0.473
#   finding worth reporting: the hand-designed cost matrix LOST to the
#   automatic one. reason: class_weight="balanced" computes weights as
#   n_total / (n_classes * n_class), which for this data is roughly
#   1 : 3.5 : 21.5 (Balance : Shortage : Surplus). the "aggressive" custom
#   weights of 1 : 2 : 6 were therefore far LESS aggressive toward Surplus
#   than the automatic scheme they were meant to beat. lesson: domain
#   intuition about misclassification cost has to be checked against what
#   the automatic baseline is already doing, not chosen in a vacuum.
#
# FEATURES THAT FAILED (tested and rejected, reported rather than deleted):
#   core_fraction         weakest of round 1's four (~0.02)
#   oa_work_activities    scored exactly 0.0000 -- no information at all
#   employment_growth     surprisingly weak (~0.01-0.02) despite being a
#                         demand-side signal, which was the whole reason
#                         for adding it
#
# ---------- FINAL MODEL ----------
# random forest, class_weight="balanced", max_depth=6, n_estimators=400,
# min_samples_leaf=1, on the 12-feature set.
#   cv macro-F1 mean: 0.497   (proposal target was 0.50 -- essentially met)
#   per-class:  Shortage F1 0.614  (precision 0.589, recall 0.641)
#               Balance  F1 0.831  (precision 0.862, recall 0.803)
#               Surplus  F1 0.040  (precision 0.030, recall 0.059)
#
# CONFUSION MATRIX (rows=true, cols=pred, order Shortage/Balance/Surplus):
#   [[ 66,  32,   5],
#    [ 45, 293,  27],
#    [  1,  15,   1]]
#
# THE SURPLUS CEILING: round 1 got Surplus F1 exactly 0.000 -- it never
# correctly identified a single one. round 2 cracked it slightly (1 of 17),
# but this is still effectively a failure and should be reported as one.
# the cause is data, not modelling: 17 examples nationwide is ~13 per
# training fold, too few for any model to learn a reliable pattern from.
# this exact failure mode was predicted in the team's own formal proposal
# before any code was written ("the shortage classifier may struggle most
# on the Surplus class, which is the rarest and noisiest projection").
#
# REPRODUCIBILITY: verified. ran the full script 4 times, including once on
# data regenerated from scratch from the raw government CSVs via
# scripts/01_enrich_oasis.py rather than the shipped release zip.
# every figure identical to the last decimal place every time.
# mutual_info_classif is explicitly seeded (random_state=42) -- without it,
# only the printed feature scores wobble slightly between runs; model
# results were never affected.
#
# NOTE FOR D. SOMASUNDARAM (trust layer): the final model is now the ROUND 2
# config above, NOT the round-1 RandomForestClassifier(random_state=SEED,
# class_weight="balanced") with 4 features. the audit needs the tuned params
# and the 12-feature set. the exact feature list ships in
# results/shortage_classifier__random_forest_tuned_v2.json under
# extra.features, and the params under extra.best_params.
#
# HONEST TAKEAWAY: macro-F1 0.497 against a 0.50 target. Shortage prediction
# is genuinely useful (catches ~2 in 3 real cases). Balance is strong.
# Surplus is not solved and is unlikely to be solvable with this data.
# a tool that shapes real career decisions should say where it is confident
# and where it isn't, rather than implying uniform certainty.