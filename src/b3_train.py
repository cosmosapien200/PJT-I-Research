"""B3 CardioPHON quality model on the PhysioNet common corpus, following quality_assesment.py of the CardioPHON repo:
standardise -> mutual-information ranking, keep top 20 % of features -> soft-voting ensemble of
RBF-SVM (grid C {1,10,100} x gamma {1e-4, scale, auto}, Platt probabilities), random forest (200 trees) and
gradient boosting (200 trees); no class balancing (published configuration balance='n').
5-fold CV (patient-grouped folds) + final model on all windows.

  python b3_train.py [seed]   seed != 42 (post hoc seed robustness): tag b3_cardiophon_s<seed>; the random_state of the RF,
                              GB and SVM (Platt CV) changes; MI selection and the SVM subsample keep seed 42 (same data/features)
"""
import os, math
import numpy as np, pandas as pd, joblib
from concurrent.futures import ProcessPoolExecutor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from common import PROC, LOGT, MODELS, SEED, RES
from classic import subsample

SVM_MAX = 8000   # SVM grid search + fit on a stratified subsample (tractability); RF and GB use all windows


class B3Model:
    def __init__(self, seed=SEED):
        self.seed = seed

    def fit(self, X, y):
        self.imp = SimpleImputer(strategy='median').fit(X)
        Xs = self.imp.transform(X)
        self.sc = StandardScaler().fit(Xs)
        Xs = np.clip(self.sc.transform(Xs), -20, 20)
        mi = mutual_info_classif(Xs, y, random_state=SEED)
        self.sel = np.argsort(mi)[-math.floor(len(mi) * 0.2):]
        self.mi = mi
        Xs = Xs[:, self.sel]
        idx = subsample(np.arange(len(y)), y, SVM_MAX)
        self.svm = GridSearchCV(SVC(kernel='rbf', cache_size=1000, probability=True, random_state=self.seed),
                                [{'C': [1, 10, 100], 'gamma': [0.0001, 'scale', 'auto']}], refit=True, n_jobs=1).fit(Xs[idx], y[idx])
        self.rf = RandomForestClassifier(n_estimators=200, bootstrap=True, random_state=self.seed, n_jobs=1).fit(Xs, y)
        self.gb = GradientBoostingClassifier(n_estimators=200, random_state=self.seed).fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = np.clip(self.sc.transform(self.imp.transform(X)), -20, 20)[:, self.sel]
        p = (self.svm.predict_proba(Xs)[:, 1] + self.rf.predict_proba(Xs)[:, 1] + self.gb.predict_proba(Xs)[:, 1]) / 3
        return np.column_stack([1 - p, p])


def fit_fold(args):
    k, X, y, fold, seed = args
    tr = np.flatnonzero(fold != k) if k >= 0 else np.arange(len(y))
    import b3_train
    m = b3_train.B3Model(seed).fit(X[tr], y[tr])
    if k < 0:
        return k, m, None
    te = np.flatnonzero(fold == k)
    return k, None, (te, m.predict_proba(X[te])[:, 1])


def main(seed=SEED):
    import b3_train   # pickle the model class under its module name, not __main__
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    z = np.load(os.path.join(PROC, 'pn_b3feats.npz'))
    X, names = z['X'].astype(np.float64), z['names']
    X[~np.isfinite(X)] = np.nan
    y, fold = meta.unusable.values, meta.fold.values
    tag = 'b3_cardiophon' if seed == SEED else f'b3_cardiophon_s{seed}'
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(fit_fold, [(k, X, y, fold, seed) for k in list(range(5)) + [-1]]))
    oof = np.zeros(len(y)); rows = []
    for k, m, r in res:
        if k < 0:
            joblib.dump(m, os.path.join(MODELS, f'{tag}.joblib'))
            if seed != SEED:
                continue
            pd.DataFrame(dict(feature=names, mutual_information=m.mi, selected=np.isin(np.arange(len(names)), m.sel))) \
                .sort_values('mutual_information', ascending=False).to_csv(os.path.join(RES, 'b3_feature_mutual_information.csv'), index=False)
            continue
        te, p = r; oof[te] = p
        pr = p >= 0.5
        rows.append(dict(fold=k, auroc=roc_auc_score(y[te], p), balanced_acc=balanced_accuracy_score(y[te], pr),
                         sensitivity=recall_score(y[te], pr), specificity=recall_score(1 - y[te], 1 - pr)))
    cv = pd.DataFrame(rows).sort_values('fold')
    cv.to_csv(os.path.join(LOGT, f'{tag}_cv.csv'), index=False)
    pd.DataFrame(dict(window_id=meta.window_id, fold=fold, y=y, score=oof)).to_csv(
        os.path.join(PROC, f'val_preds_{tag}.csv'), index=False)
    print(tag, cv.mean().round(4).to_dict())


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else SEED)
