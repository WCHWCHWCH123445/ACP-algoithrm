# 4_sklearn_models.py
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# ---------- 模型3：GMM ----------
def train_gmm(normal_mfccs, n_components=2, reg_covar=1e-4):
    """
    训练 GMM，增加 reg_covar 防止奇异协方差
    n_components: 建议设为 2 或 1（样本少时用1）
    reg_covar: 对角线正则化项
    """
    X_train = np.array([np.mean(mfcc, axis=0) for mfcc in normal_mfccs])
    # 如果样本数小于 n_components，强制 n_components = 1
    if X_train.shape[0] < n_components:
        n_components = 1
        print(f"⚠️ 样本数较少，GMM 组件数自动降为 1")
    model = GaussianMixture(n_components=n_components, 
                            random_state=42, 
                            reg_covar=reg_covar)  # 增加正则化
    model.fit(X_train)
    return model, X_train

def predict_gmm(model, train_features, test_mfccs):
    test_features = np.array([np.mean(mfcc, axis=0) for mfcc in test_mfccs])
    train_scores = -model.score_samples(train_features)
    test_scores = -model.score_samples(test_features)
    max_train = train_scores.max() if train_scores.max() > 0 else 1e-6
    scores = np.clip(test_scores / max_train, 0, 1)
    return scores

# ---------- 模型4：One-Class SVM ----------
def train_ocsvm(normal_mfccs, nu=0.1):
    X_train = np.array([np.mean(mfcc, axis=0) for mfcc in normal_mfccs])
    model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    model.fit(X_train)
    return model, X_train

def predict_ocsvm(model, train_features, test_mfccs):
    test_features = np.array([np.mean(mfcc, axis=0) for mfcc in test_mfccs])
    train_scores = -model.decision_function(train_features)
    test_scores = -model.decision_function(test_features)
    max_train = train_scores.max() if train_scores.max() > 0 else 1e-6
    scores = np.clip(test_scores / max_train, 0, 1)
    return scores

# ---------- 模型5：Isolation Forest ----------
def train_iforest(normal_mfccs, contamination=0.1):
    X_train = np.array([np.mean(mfcc, axis=0) for mfcc in normal_mfccs])
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model, X_train

def predict_iforest(model, train_features, test_mfccs):
    test_features = np.array([np.mean(mfcc, axis=0) for mfcc in test_mfccs])
    train_scores = -model.decision_function(train_features)
    test_scores = -model.decision_function(test_features)
    max_train = train_scores.max() if train_scores.max() > 0 else 1e-6
    scores = np.clip(test_scores / max_train, 0, 1)
    return scores