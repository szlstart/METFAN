"""
评估模块，包含评估多标签分类性能的度量指标
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, matthews_corrcoef, auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def find_best_thresholds(y_true, y_pred):
    """
    为每个类别找到基于F1分数的最佳阈值
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        
    Returns:
        最佳阈值列表
    """
    num_classes = y_true.shape[1]
    best_thresholds = []
    
    for i in range(num_classes):
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0, 1, 0.01):
            binary_pred = (y_pred[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true[:, i], binary_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        best_thresholds.append(best_threshold)
    
    return best_thresholds


def Accuracy(y_hat, y):
    """
    准确率：正确预测的标签与总标签（包括正确和错误预测的标签以及实际标签但在预测中遗漏的标签）的平均比率
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        准确率值
    """
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if union == 0:  # 避免除以零
            continue
        sorce_k += intersection / union
    return sorce_k / n if n > 0 else 0


def F1_label(y_hat, y):
    """
    计算所有标签的宏平均F1分数
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        F1分数值
    """
    n, m = y_hat.shape
    f1_sum = 0.0
    for j in range(m):
        y_pred = y_hat[:, j]
        y_true = y[:, j]
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        if TP == 0 and FP == 0 and FN == 0:
            f1_j = 1.0
        else:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            if precision + recall == 0:
                f1_j = 0.0
            else:
                f1_j = 2 * precision * recall / (precision + recall)
        f1_sum += f1_j
    return f1_sum / m if m > 0 else 0


def Aiming(y_hat, y):
    """
    Aiming率（也称为"精确率"）：反映正确预测的标签与预测标签的平均比率；衡量预测标签命中实际标签目标的百分比。
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        Aiming值
    """
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        if np.sum(y_hat[v]) == 0:  # 避免除以零
            continue
        # 使用元素级乘法而不是按位与
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        sorce_k += intersection / np.sum(y_hat[v])
    return sorce_k / n if n > 0 else 0


def Coverage(y_hat, y):
    """
    Coverage率（也称为"召回率"）：反映正确预测的标签与实际标签的平均比率；衡量预测覆盖实际标签的百分比。
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        Coverage值
    """
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        if np.sum(y[v]) == 0:  # 避免除以零
            continue
        # 使用元素级乘法而不是按位与
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        sorce_k += intersection / np.sum(y[v])
    return sorce_k / n if n > 0 else 0


def AbsoluteTrue(y_hat, y):
    """
    计算所有预测完全匹配的样本比例
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        AbsoluteTrue值
    """
    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if np.array_equal(y_hat[v], y[v]):
            score_k += 1
    return score_k / n if n > 0 else 0


def AbsoluteFalse(y_hat, y):
    """
    计算汉明损失（不正确标签的比例）
    
    Args:
        y_hat: 预测标签
        y: 真实标签
    
    Returns:
        AbsoluteFalse值
    """
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        sorce_k += sum(y_hat[v] != y[v]) / m
    return sorce_k / n if n > 0 else 0


def evaluate(y_hat, y):
    """
    使用各种指标评估模型预测
    
    Args:
        y_hat: 模型预测（原始分数）
        y: 真实标签
        
    Returns:
        包含评估指标的字典，准确率和宏平均F1作为前两个
    """
    # 为每个类别找到最佳阈值
    best_thresholds = find_best_thresholds(y, y_hat)

    # 应用阈值获得二值化预测
    y_pred = np.zeros_like(y_hat)
    for i, threshold in enumerate(best_thresholds):
        y_pred[:, i] = (y_hat[:, i] >= threshold).astype(int)

    # 计算所有指标
    accuracy = Accuracy(y_pred, y)
    macro_f1 = F1_label(y_pred, y)
    aiming = Aiming(y_pred, y)
    coverage = Coverage(y_pred, y)
    absolute_true = AbsoluteTrue(y_pred, y)
    absolute_false = AbsoluteFalse(y_pred, y)
    
    # 返回作为字典的指标，准确率和宏平均F1排在首位
    return dict(
        accuracy=accuracy,
        macro_f1=macro_f1,
        aiming=aiming, 
        coverage=coverage, 
        absolute_true=absolute_true,
        absolute_false=absolute_false
    )


def visualize_predictions(y_true, y_pred, y_pred_proba, output_dir):
    """
    可视化预测结果
    
    Args:
        y_true: 真实标签
        y_pred: 二值化预测
        y_pred_proba: 预测概率
        output_dir: 输出目录
    """
    num_classes = y_true.shape[1]
    
    # 为每个类别创建混淆矩阵
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i in range(min(num_classes, 15)):  # 最多绘制15个类别
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Class {i+1}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # 绘制每个类别的性能
    class_metrics = []
    for i in range(num_classes):
        precision = precision_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        class_metrics.append([precision, recall, f1])
    
    class_metrics = np.array(class_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)
    width = 0.25
    
    ax.bar(x - width, class_metrics[:, 0], width, label='Precision')
    ax.bar(x, class_metrics[:, 1], width, label='Recall')
    ax.bar(x + width, class_metrics[:, 2], width, label='F1-score')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(num_classes)])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_class.png'), dpi=300)
    plt.close()
    
    # 计算每个样本的正确预测标签数量
    correct_labels_per_sample = np.sum(y_pred == y_true, axis=1)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(correct_labels_per_sample, bins=num_classes+1, alpha=0.7)
    plt.xlabel('Number of Correctly Predicted Labels per Sample')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Correctly Predicted Labels per Sample')
    plt.savefig(os.path.join(output_dir, 'correct_labels_distribution.png'), dpi=300)
    plt.close()