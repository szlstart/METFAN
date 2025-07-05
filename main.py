# """
# 独立测试集多类别评估脚本

# 用途: 加载训练好的模型，在独立测试集上评估每个类别的AUC、MCC、和F1值
# 同时保存每个样本的预测结果
# """
# import os
# import torch
# import numpy as np
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

# # 导入项目模块
# from config import set_seed, DEVICE
# from data_utils import getSequenceData, PadEncode
# from feature_loader import FeatureLoader
# from models import EnhancedETFC
# from evaluation import find_best_thresholds


# def load_model(model_path):
#     """
#     从检查点加载训练好的模型
    
#     Args:
#         model_path: 模型检查点路径
        
#     Returns:
#         加载的模型和模型配置
#     """
#     # 加载检查点
#     checkpoint = torch.load(model_path, map_location=DEVICE)
#     state_dict = checkpoint['model_state_dict']
    
#     # 从检查点中提取配置
#     config = checkpoint.get('args', {})
#     feature_types = checkpoint.get('feature_types', ['textcnn'])
#     optimize_features = checkpoint.get('optimize_features', False)
#     feature_hidden_factor_dict = checkpoint.get('feature_hidden_factor_dict', {})
    
#     # 打印模型配置
#     print(f"模型配置:")
#     print(f"  特征类型: {feature_types}")
#     print(f"  特征优化: {optimize_features}")
    
#     # 从状态字典中提取维度信息
#     fan_dim = state_dict['fan.addNorm.ln.weight'].shape[1]
#     print(f"  FAN层维度: {fan_dim}")
    
#     # 提取特征维度
#     feature_dims = {}
#     for feature_name in feature_types:
#         if feature_name != 'textcnn' and optimize_features:
#             key = f"feature_transformers.{feature_name}.transform.0.weight"
#             if key in state_dict:
#                 feature_dims[feature_name] = state_dict[key].shape[1]
#                 print(f"  特征 {feature_name} 维度: {feature_dims[feature_name]}")
    
#     # 计算CNN形状
#     if 'textcnn' in feature_types:
#         cnn_shape = fan_dim - sum(feature_dims.values())
#         print(f"  计算得到的CNN形状: {cnn_shape}")
        
#         # 推断max_pool
#         max_pool = 3  # 默认值
#         if abs(cnn_shape - 6016) < 100:
#             max_pool = 2
#         elif abs(cnn_shape - 3968) < 100:
#             max_pool = 3
#         elif abs(cnn_shape - 2944) < 100:
#             max_pool = 4
#         elif abs(cnn_shape - 2304) < 100:
#             max_pool = 5
#         print(f"  推断的max_pool: {max_pool}")
#     else:
#         max_pool = 3
    
#     # 创建模型
#     model = EnhancedETFC(
#         vocab_size=50,
#         embedding_size=config.get('embedding_size', 128),
#         output_size=config.get('num_classes', 15),
#         dropout=config.get('dropout', 0.5),
#         fan_epoch=config.get('fan_epochs', 1),
#         num_heads=config.get('num_heads', 8),
#         feature_types=feature_types,
#         feature_dims=feature_dims,
#         max_pool=max_pool,
#         optimize_features=optimize_features,
#         feature_hidden_factor_dict=feature_hidden_factor_dict
#     )
    
#     # 在创建模型后，但在加载状态字典之前，手动设置cnn_shape
#     if 'textcnn' in feature_types:
#         model.cnn_shape = cnn_shape
    
#     # 加载状态字典
#     model.load_state_dict(state_dict)
#     model.to(DEVICE)
#     model.eval()
    
#     return model, feature_types, optimize_features


# def load_test_data(data_dir, max_length):
#     """
#     加载和预处理测试数据
    
#     Args:
#         data_dir: 包含test.txt文件的目录
#         max_length: 最大序列长度
        
#     Returns:
#         预处理后的测试数据和标签
#     """
#     test_dir = os.path.join(data_dir, 'test')
    
#     # 获取序列和标签
#     test_sequence_data, test_sequence_label = getSequenceData(test_dir, 'test')
    
#     # 转换为numpy数组
#     y_test = np.array(test_sequence_label)
    
#     # 编码和填充序列
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
    
#     # 转换为torch张量
#     x_test_tensor = torch.LongTensor(x_test)
#     test_length_tensor = torch.LongTensor(test_length)
#     y_test_tensor = torch.Tensor(y_test)
    
#     print(f"测试集: {x_test.shape[0]} 个样本")
    
#     return x_test_tensor, y_test_tensor, test_length_tensor, test_sequence_data


# def load_features(feature_types, feature_loader, split='test'):
#     """
#     加载指定特征类型的预提取特征
    
#     Args:
#         feature_types: 要加载的特征类型列表
#         feature_loader: FeatureLoader实例
#         split: 要加载的数据分割 ('train', 'val', 或 'test')
        
#     Returns:
#         特征字典和特征维度字典
#     """
#     features = {}
#     feature_dims = {}
    
#     # 跳过textcnn，因为它在模型内部处理
#     preextracted_features = [f for f in feature_types if f != 'textcnn']
    
#     for feature_name in preextracted_features:
#         try:
#             feature_tensor = feature_loader.load_feature(feature_name, split)
#             features[feature_name] = feature_tensor
#             feature_dims[feature_name] = feature_tensor.shape[1]
#             print(f"加载 {feature_name} 特征，维度为 {feature_dims[feature_name]}")
#         except Exception as e:
#             print(f"加载 {feature_name} 特征时出错: {e}")
    
#     return features, feature_dims


# def save_predictions(y_true, y_pred_proba, y_pred_binary, test_sequences, result_dir, best_thresholds):
#     """
#     保存每个样本的预测结果到txt文件
    
#     Args:
#         y_true: 真实标签
#         y_pred_proba: 预测概率
#         y_pred_binary: 二值化预测结果
#         test_sequences: 测试序列
#         result_dir: 结果保存目录
#         best_thresholds: 每个类别的最佳阈值
#     """
#     num_samples, num_classes = y_true.shape
    
#     # 保存详细预测结果
#     predictions_path = os.path.join(result_dir, "detailed_predictions.txt")
    
#     with open(predictions_path, 'w', encoding='utf-8') as f:
#         # 写入文件头
#         f.write("="*100 + "\n")
#         f.write("详细预测结果\n")
#         f.write("="*100 + "\n")
#         f.write(f"样本总数: {num_samples}\n")
#         f.write(f"类别总数: {num_classes}\n")
#         f.write("-"*100 + "\n")
        
#         # 写入阈值信息
#         f.write("每个类别的最佳阈值:\n")
#         for i, threshold in enumerate(best_thresholds):
#             f.write(f"  类别 {i+1}: {threshold:.4f}\n")
#         f.write("-"*100 + "\n\n")
        
#         # 为每个样本写入预测结果
#         for sample_idx in range(num_samples):
#             f.write(f"样本 {sample_idx + 1}:\n")
            
#             # 如果有序列信息，写入序列
#             if sample_idx < len(test_sequences):
#                 sequence = test_sequences[sample_idx]
#                 f.write(f"  序列: {sequence}\n")
            
#             # 写入真实标签
#             true_labels = y_true[sample_idx]
#             true_classes = [i+1 for i, label in enumerate(true_labels) if label == 1]
#             f.write(f"  真实类别: {true_classes}\n")
            
#             # 写入预测概率
#             pred_probs = y_pred_proba[sample_idx]
#             f.write("  预测概率: [")
#             f.write(", ".join([f"{prob:.4f}" for prob in pred_probs]))
#             f.write("]\n")
            
#             # 写入二值化预测
#             pred_binary = y_pred_binary[sample_idx]
#             pred_classes = [i+1 for i, label in enumerate(pred_binary) if label == 1]
#             f.write(f"  预测类别: {pred_classes}\n")
            
#             # 计算匹配情况
#             correct_predictions = np.sum(true_labels == pred_binary)
#             total_labels = len(true_labels)
#             accuracy = correct_predictions / total_labels
#             f.write(f"  匹配准确率: {accuracy:.4f} ({correct_predictions}/{total_labels})\n")
            
#             f.write("-"*50 + "\n")
    
#     print(f"详细预测结果已保存到: {predictions_path}")
    
#     # 保存简化的CSV格式结果
#     csv_data = []
#     for sample_idx in range(num_samples):
#         row_data = {
#             'sample_id': sample_idx + 1,
#             'sequence': test_sequences[sample_idx] if sample_idx < len(test_sequences) else '',
#         }
        
#         # 添加真实标签
#         true_labels = y_true[sample_idx]
#         true_classes = [i+1 for i, label in enumerate(true_labels) if label == 1]
#         row_data['true_classes'] = ','.join(map(str, true_classes))
        
#         # 添加预测概率
#         pred_probs = y_pred_proba[sample_idx]
#         for i in range(num_classes):
#             row_data[f'prob_class_{i+1}'] = f"{pred_probs[i]:.4f}"
        
#         # 添加二值化预测
#         pred_binary = y_pred_binary[sample_idx]
#         for i in range(num_classes):
#             row_data[f'pred_class_{i+1}'] = int(pred_binary[i])
        
#         # 添加预测类别
#         pred_classes = [i+1 for i, label in enumerate(pred_binary) if label == 1]
#         row_data['predicted_classes'] = ','.join(map(str, pred_classes))
        
#         # 添加匹配信息
#         correct_predictions = np.sum(true_labels == pred_binary)
#         row_data['correct_predictions'] = correct_predictions
#         row_data['total_labels'] = len(true_labels)
#         row_data['accuracy'] = f"{correct_predictions / len(true_labels):.4f}"
        
#         csv_data.append(row_data)
    
#     # 保存为CSV
#     predictions_df = pd.DataFrame(csv_data)
#     csv_path = os.path.join(result_dir, "predictions.csv")
#     predictions_df.to_csv(csv_path, index=False)
#     print(f"预测结果CSV已保存到: {csv_path}")
    
#     # 保存概率矩阵
#     prob_matrix_path = os.path.join(result_dir, "probability_matrix.txt")
#     np.savetxt(prob_matrix_path, y_pred_proba, fmt='%.6f', 
#                header=f"Probability matrix ({num_samples} samples x {num_classes} classes)")
#     print(f"概率矩阵已保存到: {prob_matrix_path}")
    
#     # 保存二值化预测矩阵
#     binary_matrix_path = os.path.join(result_dir, "binary_predictions.txt")
#     np.savetxt(binary_matrix_path, y_pred_binary, fmt='%d',
#                header=f"Binary predictions matrix ({num_samples} samples x {num_classes} classes)")
#     print(f"二值化预测矩阵已保存到: {binary_matrix_path}")


# def evaluate_per_class(y_true, y_pred_proba):
#     """
#     计算每个类别的指标
    
#     Args:
#         y_true: 真实标签
#         y_pred_proba: 预测概率
        
#     Returns:
#         包含每个类别指标的DataFrame
#     """
#     # 查找最佳阈值
#     best_thresholds = find_best_thresholds(y_true, y_pred_proba)
    
#     # 应用阈值
#     y_pred = np.zeros_like(y_pred_proba)
#     for i, threshold in enumerate(best_thresholds):
#         y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
    
#     num_classes = y_true.shape[1]
#     results = []
    
#     for i in range(num_classes):
#         true_class = y_true[:, i]
#         pred_proba_class = y_pred_proba[:, i]
#         pred_class = y_pred[:, i]
        
#         # 跳过测试集中只有一种类别的类
#         if len(np.unique(true_class)) < 2:
#             print(f"警告: 类别 {i+1} 在测试集中只有一种标签类型。跳过AUC计算。")
#             auc_score = float('nan')
#         else:
#             try:
#                 auc_score = roc_auc_score(true_class, pred_proba_class)
#             except Exception as e:
#                 print(f"计算类别 {i+1} 的AUC时出错: {e}")
#                 auc_score = float('nan')
        
#         try:
#             mcc = matthews_corrcoef(true_class, pred_class)
#         except:
#             mcc = float('nan')
            
#         try:
#             f1 = f1_score(true_class, pred_class)
#         except:
#             f1 = float('nan')
        
#         # 计算类别统计信息
#         class_positive = np.sum(true_class == 1)
#         class_negative = np.sum(true_class == 0)
#         class_prevalence = class_positive / len(true_class)
        
#         results.append({
#             '类别': i+1,
#             'AUC': auc_score,
#             'MCC': mcc,
#             'F1': f1,
#             '阈值': best_thresholds[i],
#             '正例数': class_positive,
#             '负例数': class_negative,
#             '正例比例': class_prevalence
#         })
    
#     # 创建DataFrame
#     results_df = pd.DataFrame(results)
    
#     # 计算总体指标
#     overall_f1 = results_df['F1'].mean()
    
#     print(f"\n总体宏平均F1分数: {overall_f1:.4f}")
    
#     return results_df, best_thresholds


# def plot_metrics(results_df, result_dir):
#     """
#     可视化指标 - 修复版
    
#     Args:
#         results_df: 包含指标的DataFrame
#         result_dir: 保存图表的目录
#     """
#     # 按类别编号排序
#     results_df = results_df.sort_values('类别')
    
#     # 为每个类别绘制AUC、MCC和F1
#     plt.figure(figsize=(15, 5))
    
#     # 将AUC、MCC、F1绘制在一张图上
#     plt.subplot(1, 2, 1)
#     bar_width = 0.25
#     classes = results_df['类别'].values  # Convert to numpy array
#     x = np.arange(len(classes))
    
#     plt.bar(x - bar_width, results_df['AUC'].values, bar_width, label='AUC', alpha=0.7)
#     plt.bar(x, results_df['MCC'].values, bar_width, label='MCC', alpha=0.7)
#     plt.bar(x + bar_width, results_df['F1'].values, bar_width, label='F1', alpha=0.7)
    
#     plt.xlabel('类别')
#     plt.ylabel('分数')
#     plt.title('各类别性能指标')
#     plt.xticks(x, classes)
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # 绘制类别正例比例
#     plt.subplot(1, 2, 2)
#     plt.bar(classes, results_df['正例比例'].values, alpha=0.7, color='green')
#     plt.xlabel('类别')
#     plt.ylabel('正例比例')
#     plt.title('测试集中各类别的正例比例')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, 'class_metrics.png'), dpi=300)
#     plt.close()
    
#     # 绘制指标与正例比例的关系
#     plt.figure(figsize=(15, 5))
    
#     metrics = ['AUC', 'MCC', 'F1']
#     colors = ['blue', 'orange', 'green']
    
#     for i, metric in enumerate(metrics):
#         plt.subplot(1, 3, i+1)
        
#         # Convert pandas Series to numpy arrays
#         x_values = results_df['正例比例'].values
#         y_values = results_df[metric].values
        
#         plt.scatter(x_values, y_values, alpha=0.7, color=colors[i])
        
#         # 添加趋势线
#         z = np.polyfit(x_values, y_values, 1)
#         p = np.poly1d(z)
#         x_range = np.linspace(min(x_values), max(x_values), 100)  # Create smooth x range for trend line
#         plt.plot(x_range, p(x_range), "r--", alpha=0.7)
        
#         plt.xlabel('类别正例比例')
#         plt.ylabel(metric)
#         plt.title(f'{metric} vs 类别正例比例')
#         plt.grid(linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, 'metrics_vs_prevalence.png'), dpi=300)
#     plt.close()


# def main(args):
#     # 设置随机种子
#     set_seed(args.seed)
    
#     # 创建结果目录
#     os.makedirs(args.result_dir, exist_ok=True)
    
#     # 加载模型
#     print("\n" + "="*60)
#     print("加载模型")
#     print("="*60)
#     model, feature_types, optimize_features = load_model(args.model_path)
    
#     # 加载测试数据
#     print("\n" + "="*60)
#     print("加载测试数据")
#     print("="*60)
#     x_test, y_test, test_length, test_sequences = load_test_data(args.data_dir, args.max_length)
    
#     # 设置特征加载器
#     feature_loader = FeatureLoader(features_dir=args.features_dir)
    
#     # 加载特征
#     print("\n" + "="*60)
#     print("加载特征")
#     print("="*60)
#     features, feature_dims = load_features(feature_types, feature_loader)
    
#     # 更新模型的特征维度
#     model.feature_dims = feature_dims
    
#     # 获取模型预测
#     print("\n" + "="*60)
#     print("生成预测")
#     print("="*60)
    
#     with torch.no_grad():
#         batch_size = 32
#         all_preds = []
        
#         for i in range(0, len(x_test), batch_size):
#             batch_x = x_test[i:i+batch_size].to(DEVICE)
#             batch_len = test_length[i:i+batch_size].to(DEVICE)
            
#             # 准备特征张量
#             batch_features = {}
#             for feature_name, feature_tensor in features.items():
#                 batch_features[feature_name] = feature_tensor[i:i+batch_size].to(DEVICE)
            
#             # 前向传播
#             try:
#                 outputs = model(batch_x, batch_len, batch_features)
#                 outputs = torch.sigmoid(outputs)
#                 all_preds.extend(outputs.cpu().numpy())
#             except Exception as e:
#                 print(f"生成批次 {i//batch_size} 的预测时出错: {e}")
#                 print(f"批次大小: {batch_x.shape}")
#                 print(f"特征: {[k for k in batch_features.keys()]}")
#                 raise
    
#     # 转换预测为numpy数组
#     y_pred = np.array(all_preds)
#     y_true = y_test.numpy()
    
#     # 评估每个类别
#     print("\n" + "="*60)
#     print("评估各类别指标")
#     print("="*60)
#     results_df, best_thresholds = evaluate_per_class(y_true, y_pred)
    
#     # 计算二值化预测
#     y_pred_binary = np.zeros_like(y_pred)
#     for i, threshold in enumerate(best_thresholds):
#         y_pred_binary[:, i] = (y_pred[:, i] >= threshold).astype(int)
    
#     # 保存预测结果
#     print("\n" + "="*60)
#     print("保存预测结果")
#     print("="*60)
#     save_predictions(y_true, y_pred, y_pred_binary, test_sequences, args.result_dir, best_thresholds)
    
#     # 打印结果
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.precision', 4)
#     print("\n各类别指标:")
#     print(results_df)
    
#     # 保存结果到CSV
#     results_path = os.path.join(args.result_dir, "per_class_metrics.csv")
#     results_df.to_csv(results_path, index=False)
#     print(f"\n结果已保存到 {results_path}")
    
#     # 绘制指标
#     plot_metrics(results_df, args.result_dir)
    
#     return results_df


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="在测试集上评估训练好的模型的每个类别性能")
    
#     # 路径
#     parser.add_argument('--data_dir', type=str, default='dataset/pre',
#                         help='数据集目录 (default: dataset/pre)')
#     parser.add_argument('--result_dir', type=str, default='result/per_class_eval',
#                         help='结果保存目录 (default: result/per_class_eval)')
#     parser.add_argument('--features_dir', type=str, default='features',
#                         help='特征目录 (default: features)')
#     parser.add_argument('--model_path', type=str, required=True,
#                         help='训练好的模型检查点路径')
    
#     # 参数
#     parser.add_argument('--max_length', type=int, default=50,
#                         help='最大序列长度 (default: 50)')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='随机种子 (default: 42)')
    
#     args = parser.parse_args()
    
#     # 如果未指定模型路径，使用默认路径
#     if args.model_path is None:
#         print("错误: 必须指定模型路径 (--model_path)")
#         parser.print_help()
#         exit(1)
    
#     main(args)
    
"""
肽分类项目主脚本，用于运行训练和评估
"""
import os
import torch
import numpy as np
import json
import datetime
from tqdm import tqdm

from config import get_args, set_seed, DEVICE
from data_utils import load_dataset, create_dataloaders
from feature_loader import FeatureLoader
from models import EnhancedETFC
from train import train_model, evaluate_model
import warnings

# 屏蔽 Python 警告
warnings.filterwarnings("ignore")

# 屏蔽 PyTorch/TorchVision 警告
os.environ["PYTORCH_DISABLE_LEGACY_WARNINGS"] = "1"

# 屏蔽 Transformers 警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 显式屏蔽 TorchVision Beta 警告
try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except:
    pass

def main(args):
    """主函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建带有序号的结果目录
    result_dir = os.path.join(args.result_dir, str(args.num))
    os.makedirs(result_dir, exist_ok=True)
    
    # 记录所使用的特征
    feature_str = '_'.join(args.features)
    feature_str += "_optimized" if args.optimize_features else ""
    print(f"\n使用特征: {feature_str}")
    
    # 创建特征特定子目录
    result_dir = os.path.join(result_dir, feature_str)
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载数据集
    print("\n" + "="*60)
    print("加载数据集")
    print("="*60)
    
    (train_sequence_data, val_sequence_data, test_sequence_data,
     x_train_tensor, y_train_tensor, train_length_tensor,
     x_val_tensor, y_val_tensor, val_length_tensor,
     x_test_tensor, y_test_tensor, test_length_tensor) = load_dataset(args)
    
    # 初始化特征加载器
    feature_loader = FeatureLoader(features_dir=args.features_dir)
    
    # 验证特征并获取有效特征
    valid_features, preextracted_features = feature_loader.validate_features(args.features)
    
    if not valid_features:
        print("错误: 没有有效的特征可用")
        return
    
    print(f"有效特征: {', '.join(valid_features)}")
    print(f"将加载的预提取特征: {', '.join(preextracted_features)}")
    
    # 显示特征优化状态
    if args.optimize_features:
        print(f"特征优化已启用，将对预提取特征进行动态学习和优化")
        
        # 显示特征特定的因子
        if args.feature_hidden_factor_dict:
            print("特征特定的隐藏层因子:")
            for feature_name, factor in args.feature_hidden_factor_dict.items():
                if feature_name in valid_features:
                    print(f"  - {feature_name}: {factor}")
        else:
            print("未指定特征特定的隐藏层因子，将使用默认值")
    else:
        print(f"特征优化未启用，预提取特征将保持静态")
    
    # 加载预提取特征
    all_features = feature_loader.load_all_features(
        preextracted_features, 
        splits=('train', 'val', 'test')
    )
    
    # 获取特征维度
    feature_dims = feature_loader.get_features_dims(valid_features)
    print("特征维度:", feature_dims)

    # 打印指定的特征顺序
    print(f"模型将按以下顺序处理特征: {valid_features}")

    # 准备预提取特征的顺序列表（除去textcnn）
    preextracted_feature_order = [f for f in valid_features if f != 'textcnn']
    print(f"预提取特征处理顺序: {preextracted_feature_order}")

    # 创建DataLoaders - 注意这里传递了feature_order参数
    train_loader, val_loader, test_loader = create_dataloaders(
        x_train_tensor, y_train_tensor, train_length_tensor,
        x_val_tensor, y_val_tensor, val_length_tensor,
        x_test_tensor, y_test_tensor, test_length_tensor,
        all_features=all_features,
        batch_size=args.batch_size,
        feature_order=preextracted_feature_order  # 传递除了textcnn之外的特征顺序
    )
    
    # 创建模型
    print("\n" + "="*60)
    print("创建模型")
    print("="*60)
    
    model = EnhancedETFC(
        vocab_size=50,  # 氨基酸字母表大小+PAD
        embedding_size=args.embedding_size,
        output_size=args.num_classes,
        dropout=args.dropout,
        fan_epoch=args.fan_epochs,
        num_heads=args.num_heads,
        feature_types=valid_features,
        feature_dims=feature_dims,
        max_pool=args.max_pool,
        optimize_features=args.optimize_features,
        feature_hidden_factor_dict=args.feature_hidden_factor_dict
    ).to(DEVICE)
    
    # 计算并打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 如果启用特征优化，计算特征变换器参数
    if args.optimize_features:
        feature_transformer_params = sum(p.numel() for name, p in model.named_parameters() 
                                       if 'feature_transformers' in name or 'feature_adapters' in name)
        print(f"特征优化参数数量: {feature_transformer_params:,} ({feature_transformer_params/trainable_params*100:.2f}%)")
    
    # 模式选择
    if args.mode == 'train':
        # 训练模型
        print("\n" + "="*60)
        print("训练模型")
        print("="*60)
        
        model, history = train_model(model, train_loader, val_loader, args, DEVICE, result_dir)
        
        # 评估模型
        print("\n" + "="*60)
        print("评估模型")
        print("="*60)
        
        test_metrics, best_thresholds = evaluate_model(model, test_loader, DEVICE, result_dir)
        
        # 保存结果
        results = {
            "args": vars(args),
            "features_used": valid_features,
            "optimize_features": args.optimize_features,
            "feature_hidden_factors": args.feature_hidden_factor_dict,
            "test_metrics": test_metrics,
            "best_thresholds": best_thresholds.tolist(),
            "model_params": {
                "total": total_params,
                "trainable": trainable_params,
            }
        }
        
        # 添加特征优化相关信息
        if args.optimize_features:
            results["model_params"]["feature_optimization"] = feature_transformer_params
        
        with open(os.path.join(result_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # 打印测试结果
        print("\n测试结果:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    elif args.mode == 'evaluate':
        # 加载预训练模型
        if not args.model_path:
            raise ValueError("评估模式需要指定预训练模型路径 (--model_path)")
        
        print(f"加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 从检查点获取特征类型（如果有）
        if 'feature_types' in checkpoint:
            print(f"检查点中使用的特征: {checkpoint['feature_types']}")
            # 检查是否与当前选择的特征一致
            if set(checkpoint['feature_types']) != set(valid_features):
                print("警告: 当前选择的特征与模型训练时使用的特征不同!")
        
        # 检查特征优化设置
        if 'optimize_features' in checkpoint:
            loaded_optimize = checkpoint['optimize_features']
            if loaded_optimize != args.optimize_features:
                print(f"警告: 模型训练时特征优化设置为 {loaded_optimize}，当前设置为 {args.optimize_features}")
        
        # 检查特征特定的隐藏层因子
        if 'feature_hidden_factor_dict' in checkpoint:
            loaded_factors = checkpoint['feature_hidden_factor_dict']
            if loaded_factors:
                print("模型训练时使用的特征特定隐藏层因子:")
                for feature, factor in loaded_factors.items():
                    print(f"  - {feature}: {factor}")
                
                # 检查是否与当前设置一致
                if args.feature_hidden_factor_dict and args.feature_hidden_factor_dict != loaded_factors:
                    print("警告: 当前特征特定隐藏层因子与训练时使用的不同!")
                    print("训练时的因子:", loaded_factors)
                    print("当前指定的因子:", args.feature_hidden_factor_dict)
        
        # 评估模型
        print("\n" + "="*60)
        print("评估模型")
        print("="*60)
        
        test_metrics, best_thresholds = evaluate_model(model, test_loader, DEVICE, result_dir)
        
        # 保存结果
        results = {
            "args": vars(args),
            "features_used": valid_features,
            "optimize_features": args.optimize_features,
            "feature_hidden_factors": args.feature_hidden_factor_dict,
            "test_metrics": test_metrics,
            "best_thresholds": best_thresholds.tolist()
        }
        
        with open(os.path.join(result_dir, "eval_results.json"), 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # 打印测试结果
        print("\n测试结果:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")


def run_ablation_study(args):
    """运行消融实验，测试不同特征组合和优化设置"""
    # 初始化特征加载器
    feature_loader = FeatureLoader(features_dir=args.features_dir)
    
    # 获取可用的预提取特征
    available_features = feature_loader.available_features
    print(f"发现可用特征: {', '.join(available_features)}")
    
    # 定义要测试的特征组合
    feature_combinations = [
        # 基础单一特征
        ['textcnn'],  # 仅TextCNN
    ]
    
    # 为每个预提取特征添加单独测试
    for feature in available_features:
        feature_combinations.append([feature])
    
    # TextCNN与单个特征组合
    for feature in available_features:
        feature_combinations.append(['textcnn', feature])
    
    # 两两特征组合（不包括TextCNN）
    for i, feature1 in enumerate(available_features):
        for feature2 in available_features[i+1:]:
            feature_combinations.append([feature1, feature2])
    
    # 三特征组合 (TextCNN + 两个预提取特征)
    for i, feature1 in enumerate(available_features):
        for feature2 in available_features[i+1:]:
            feature_combinations.append(['textcnn', feature1, feature2])
    
    # 添加三特征组合（不包括TextCNN）
    for i, feature1 in enumerate(available_features):
        for j, feature2 in enumerate(available_features[i+1:], i+1):
            for feature3 in available_features[j+1:]:
                feature_combinations.append([feature1, feature2, feature3])
    
    # 添加四特征组合
    if len(available_features) >= 3:
        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features[i+1:], i+1):
                for k, feature3 in enumerate(available_features[j+1:], j+1):
                    feature_combinations.append(['textcnn', feature1, feature2, feature3])
    
    # 添加所有特征的组合
    all_features = ['textcnn'] + available_features
    feature_combinations.append(all_features)
    
    # 测试优化与非优化对比
    optimization_settings = [False, True]
    
    # 记录实验结果
    results = {}
    
    # 遍历所有实验配置
    for features in feature_combinations:
        for optimize in optimization_settings:
            # 如果只有textcnn，不需要测试优化版本
            if features == ['textcnn'] and optimize:
                continue
                
            # 如果没有预提取特征，也不需要测试优化版本
            if not any(f != 'textcnn' for f in features) and optimize:
                continue
                
            print("\n" + "="*80)
            exp_name = ' + '.join(features)
            if optimize:
                exp_name += " (优化)"
            print(f"实验配置: {exp_name}")
            print("="*80)
            
            # 更新参数
            args.features = features
            args.optimize_features = optimize
            
            # 如果是优化版本，使用特征特定的最佳hidden_factor
            if optimize:
                # 清空现有的特征特定因子
                args.feature_hidden_factor_dict = {}
                
                # 为每个预提取特征设置其最佳的隐藏层因子
                # 这些值应该来自于您之前的实验结果
                best_factors = {
                    'prot_t5_mean': 0.4,
                    'esm1v_max': 0.6,
                    'esm2': 0.8,
                    'prott5': 0.5,
                    'esm1b': 0.7,
                    # 可以根据您的实验结果添加更多特征的最佳因子
                }
                
                # 应用最佳因子（只为存在于当前特征集中的特征设置）
                for feature in features:
                    if feature in best_factors:
                        args.feature_hidden_factor_dict[feature] = best_factors[feature]
                        print(f"为特征 {feature} 设置最佳隐藏层因子: {best_factors[feature]}")
                    elif feature != 'textcnn':
                        # 对于没有预定义最佳因子的非textcnn特征，使用默认值
                        print(f"特征 {feature} 没有预定义的最佳因子，将使用模型中的默认值")
            
            # 运行主函数
            try:
                main(args)
                
                # 记录该配置的结果路径
                feature_str = '_'.join(features)
                if optimize:
                    feature_str += "_optimized"
                results[feature_str] = os.path.join(args.result_dir, str(args.num), feature_str, "results.json")
            except Exception as e:
                print(f"特征组合 {exp_name} 失败: {e}")
    
    # 汇总所有实验结果
    summary = {}
    for feature_str, result_path in results.items():
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
                summary[feature_str] = {
                    "metrics": result_data['test_metrics'],
                    "optimize_features": result_data.get('optimize_features', False),
                    "feature_hidden_factors": result_data.get('feature_hidden_factors', {})
                }
    
    # 保存汇总结果
    summary_path = os.path.join(args.result_dir, str(args.num), "ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("消融实验结果汇总")
    print("="*100)
    print(f"{'特征组合':<50} {'特征优化':<10} {'准确率':<10} {'宏平均F1':<10} {'绝对真率':<10}")
    print("-"*100)
    
    for feature_str, data in summary.items():
        metrics = data["metrics"]
        optimize = "是" if data["optimize_features"] else "否"
        print(f"{feature_str:<50} {optimize:<10} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} {metrics['absolute_true']:<10.4f}")


if __name__ == "__main__":
    

    # # 检查是否运行所有实验
    # if args.num < 0:
    #     # 运行消融实验
    #     run_ablation_study(args)
    # else:
        # 常规运行单个配置
    for i in range(1,6):
        args = get_args()
        args.num = i
        main(args)