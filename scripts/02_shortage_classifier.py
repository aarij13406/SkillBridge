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


# ============================================================
# SUMMARY -- Labour Shortage Classifier (Irai)
# for the final report, whenever we write it. paste/rewrite from here.
# ============================================================
#
# TASK: predict COPS's 3-year outlook per occupation -- Shortage, Balance,
# or Surplus -- from independent signals only (no gap/gap_pct, that's the
# arithmetic COPS itself uses to assign the label, using it would be leakage).
#
# DATA: 526 NOC codes in COPS -> 485 after dropping 41 unlabeled rows and
# rows that were category aggregates (e.g. "NOC1_0"), not real occupations.
# label split: Balance 365 (75%), Shortage 103 (21%), Surplus 17 (3.5%).
# heavily imbalanced, Surplus especially so.
#
# BASELINE: always guessing "Balance" gets 75.2% accuracy but is useless --
# it never once identifies a Shortage or Surplus occupation. this is why
# macro-F1 (which scores all 3 classes equally, ignoring how common each is)
# is the metric that matters here, not accuracy.
#
# FEATURES TRIED (mutual_info_classif scores, all individually weak):
#   posting          - job bank posting count per occupation      (0.033-0.036)
#   avg_rating       - mean OaSIS competency rating                (0.034-0.041)
#   core_fraction    - fraction of skills rated "core" (>=4)        (0.023, weakest)
#   employment_growth- COPS's own growth projection, no NaNs
# lesson: OaSIS describes what a job's skills LOOK like (static). Shortage/
# Surplus is about market DYNAMICS (growing or shrinking demand). that
# mismatch is probably why every OaSIS-derived feature scored weak.
#
# MODEL COMPARISON (5-fold stratified CV, same features, same folds):
#   logistic regression:  0.332
#   random forest:        0.391
#   gradient boosting:    0.376
#   mlp (neural net):     0.402   <- best of the 4, before imbalance fixes
#
# IMBALANCE STUDY (the lead deliverable):
#   MLP + SMOTE:                  0.375  (SMOTE made MLP WORSE)
#   random forest + class_weight: 0.412  (helped, became new best overall)
#   note: MLPClassifier has no class_weight option, so the class-weighting
#   arm had to run on random forest instead of MLP -- a real, reportable
#   methodological constraint, not a shortcut.
#   finding: the "right" imbalance fix isn't universal -- it depends on
#   which model it's paired with.
#
# FINAL MODEL: random forest + class_weight="balanced", 4 features.
#   cv macro-F1 mean: 0.412 (target from proposal was 0.50, not reached)
#   per-class:  Shortage F1 0.454 (recall 0.50 -- real signal)
#               Balance  F1 0.785 (strong, expected -- majority class)
#               Surplus  F1 0.000 (never once correctly identified)
#
# CONFUSION MATRIX (rows=true, cols=pred, order Shortage/Balance/Surplus):
#   [[52, 51,  0],
#    [71,278, 16],
#    [ 3, 14,  0]]
#   Surplus is a hard ceiling, not a bug: only 17 real examples total,
#   ~3 per test fold. no amount of class weighting can manufacture a
#   learnable pattern out of that few samples. this exact failure mode was
#   predicted in the team's own formal proposal before any code was written
#   ("the shortage classifier may struggle most on the Surplus class, the
#   rarest and noisiest projection") -- this run confirms it empirically.
#
# HONEST TAKEAWAY: macro-F1 0.412, short of the 0.50 target, for a specific
# and defensible reason (Surplus data scarcity), not a modeling mistake.
# Shortage prediction is genuinely useful (catches half of real cases);
# Surplus prediction is not currently possible with this data.