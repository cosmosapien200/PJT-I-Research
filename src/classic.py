"""Classic quality models: 5-fold CV on PhysioNet (patient-grouped folds), then a final model on all data.

  B1  Springer-style SQIs -> logistic regression (balanced) and SVM (RBF, balanced, Platt probabilities)
  B2  Das et al. features -> random forest (120 trees, max_features='sqrt', balanced)
Per-fold metrics: logs/train/<model>_cv.csv; out-of-fold predictions: data/processed/val_preds_<model>.csv
Final models: data/models/<model>.joblib
"""
import os
import numpy as np, pandas as pd, joblib
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from common import PROC, LOGT, MODELS, SEED

SVM_MAX = 12000   # RBF-SVM training subsample (stratified, seeded) to keep Platt scaling tractable


def make(name):
    if name == 'b1_lr':
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000))
    if name == 'b1_svm':
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED))
    if name == 'b2_rf':
        return RandomForestClassifier(n_estimators=120, max_features='sqrt', class_weight='balanced',
                                      random_state=SEED, n_jobs=4)


def feats(name, z):
    return z['b1'] if name.startswith('b1') else z['b2']


def subsample(idx, y, n, seed=SEED):
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        ic = idx[y[idx] == c]
        out.append(rng.choice(ic, int(round(n * len(ic) / len(idx))), replace=False))
    return np.sort(np.concatenate(out))


def fit_fold(args):
    name, k, X, y, fold = args
    tr = np.flatnonzero(fold != k) if k >= 0 else np.arange(len(y))
    if name == 'b1_svm':
        tr = subsample(tr, y, SVM_MAX)
    m = make(name).fit(X[tr], y[tr])
    if k < 0:
        return k, m, None
    te = np.flatnonzero(fold == k)
    return k, None, (te, m.predict_proba(X[te])[:, 1])


def main():
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    z = np.load(os.path.join(PROC, 'pn_feats.npz'))
    y, fold = meta.unusable.values, meta.fold.values
    for name in ('b1_lr', 'b1_svm', 'b2_rf'):
        X = feats(name, z)
        with ProcessPoolExecutor(3 if name == 'b2_rf' else 6) as ex:
            res = list(ex.map(fit_fold, [(name, k, X, y, fold) for k in list(range(5)) + [-1]]))
        oof = np.zeros(len(y)); rows = []
        for k, m, r in res:
            if k < 0:
                joblib.dump(m, os.path.join(MODELS, f'{name}.joblib')); continue
            te, p = r; oof[te] = p
            pr = p >= 0.5
            rows.append(dict(fold=k, auroc=roc_auc_score(y[te], p), balanced_acc=balanced_accuracy_score(y[te], pr),
                             sensitivity=recall_score(y[te], pr), specificity=recall_score(1 - y[te], 1 - pr)))
        cv = pd.DataFrame(rows).sort_values('fold')
        cv.to_csv(os.path.join(LOGT, f'{name}_cv.csv'), index=False)
        pd.DataFrame(dict(window_id=meta.window_id, fold=fold, y=y, score=oof)).to_csv(
            os.path.join(PROC, f'val_preds_{name}.csv'), index=False)
        print(name, cv.mean().round(4).to_dict())


if __name__ == '__main__':
    main()
