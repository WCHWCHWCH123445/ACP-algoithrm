# 7_run_acp.py
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score
import os
import sys


# ---------- 定义评估函数 ----------
def evaluate_weights(weights, scores, y_true):
    """
    计算给定权重下的调和平均得分 (OS)
    weights: 一维数组，长度=模型数量
    scores: (样本数, 模型数)
    y_true: (样本数,)
    返回 OS 分数
    """
    fused = np.average(scores, axis=1, weights=weights)
    auc = roc_auc_score(y_true, fused)
    pauc = roc_auc_score(y_true, fused, max_fpr=0.1)
    os_score = hmean([auc, pauc])
    return os_score


def heuristic_search_acp(scores, y_true, initial_combination=None, initial_weights=None,
                         max_iter=20, perturb_scale=0.1, decay_rate=0.95,
                         search_weights=True, search_combinations=True):
    """
    仅针对给定的 scores 和 y_true 进行 ACP 搜索
    """
    num_models = scores.shape[1]
    if initial_combination is None:
        initial_combination = list(range(num_models))
    if initial_weights is None:
        initial_weights = np.ones(len(initial_combination)) / len(initial_combination)
    else:
        initial_weights = np.array(initial_weights)
        # 确保长度匹配
        if len(initial_weights) != len(initial_combination):
            raise ValueError("初始权重数量与组合模型数不一致")

    current_combination = initial_combination.copy()
    current_weights = initial_weights.copy()

    best_score = -np.inf
    best_combination = current_combination.copy()
    best_weights = current_weights.copy()
    history = []

    def eval_combo_weights(combo, weights):
        # 从完整 scores 中选取 combo 对应的列
        selected_scores = scores[:, combo]
        weights = np.array(weights) / np.sum(weights)  # 归一化
        return evaluate_weights(weights, selected_scores, y_true)

    initial_perturb_scale = perturb_scale

    for it in range(max_iter):
        improved = False
        perturb_scale = initial_perturb_scale * (decay_rate ** it)

        if search_combinations:
            candidate_combos = []
            if len(current_combination) > 1:
                for rm_idx in current_combination:
                    new_combo = [i for i in current_combination if i != rm_idx]
                    candidate_combos.append(new_combo)
            not_selected = [i for i in range(num_models) if i not in current_combination]
            for add_idx in not_selected:
                new_combo = current_combination + [add_idx]
                candidate_combos.append(new_combo)

            for combo in candidate_combos:
                weights = np.ones(len(combo)) / len(combo)
                score = eval_combo_weights(combo, weights)
                if score > best_score:
                    best_score = score
                    best_combination = combo
                    best_weights = weights
                    improved = True

            if improved:
                current_combination = best_combination.copy()
                current_weights = best_weights.copy()

        if search_weights:
            for _ in range(5):
                noise = np.random.uniform(-perturb_scale, perturb_scale, size=len(current_weights))
                candidate_weights = current_weights + noise
                candidate_weights = np.clip(candidate_weights, 0.01, None)
                candidate_weights /= candidate_weights.sum()
                score = eval_combo_weights(current_combination, candidate_weights)
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights
                    improved = True

            if improved:
                current_weights = best_weights.copy()

        history.append((best_score, best_combination.copy(), best_weights.copy()))
        print(f"Iter {it+1}: Best Score={best_score:.4f}, Combination={best_combination}, Weights={best_weights}")

        if not improved:
            print("No improvement this iteration, stopping early.")
            # 可以 break，也可以继续，这里选择继续直到 max_iter

    return best_combination, best_weights, best_score, history


# ---------- 主程序 ----------
if __name__ == "__main__":
    # 加载得分矩阵和标签
    scores = np.load('model_scores.npy')   # shape: (22, 5)
    y_true = np.load('true_labels.npy')    # shape: (22,)

    
    
    scores = scores[:, [0, 1, 2, 4]]       
    print(f"加载数据：样本数 {scores.shape[0]}，模型数 {scores.shape[1]}（已剔除模型3）")

    # 运行 ACP 搜索（启发式搜索）
    print("\n开始 ACP 搜索...")
    best_combo, best_weights, best_score, history = heuristic_search_acp(
        scores,          # 使用剔除后的 scores
        y_true,
        initial_combination=None,
        initial_weights=None,
        max_iter=30,
        perturb_scale=0.2,
        decay_rate=0.95,
        search_weights=True,
        search_combinations=True
    )

    # 输出最终结果
    print("\n" + "="*50)
    print("ACP 搜索完成！")
   
    original_indices = [0, 1, 2, 4]
    best_original = [original_indices[i] for i in best_combo]
    print(f"最优模型组合 (原始索引): {best_original}")
    print(f"最优权重: {np.array(best_weights).round(4)}")
    print(f"最高 OS 得分: {best_score:.4f}")
    print("="*50)

    # 查看融合后的AUC和pAUC
    selected_scores = scores[:, best_combo]
    fused = np.average(selected_scores, axis=1, weights=best_weights)
    auc = roc_auc_score(y_true, fused)
    pauc = roc_auc_score(y_true, fused, max_fpr=0.1)
    print(f"对应 AUC: {auc:.4f}, pAUC: {pauc:.4f}")

    # 保存最优配置（保存的是原始索引，方便后续加载）
    np.save('best_combo.npy', np.array(best_original))
    np.save('best_weights.npy', best_weights)
    print("\n最优组合（原始索引）和权重已保存至 best_combo.npy 和 best_weights.npy")
