"""
独立测试集多类别评估脚本

用途: 加载训练好的模型，在独立测试集上评估每个类别的AUC、MCC、F1、SE、SP和ACC值
"""
import os
import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix

# 导入项目模块
from config import set_seed, DEVICE
from data_utils import getSequenceData, PadEncode
from feature_loader import FeatureLoader
from models import EnhancedETFC
from evaluation import find_best_thresholds, Accuracy, F1_label


def load_model(model_path):
    """
    从检查点加载训练好的模型
    
    Args:
        model_path: 模型检查点路径
        
    Returns:
        加载的模型和模型配置
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']
    
    config = checkpoint.get('args', {})
    feature_types = checkpoint.get('feature_types', ['textcnn'])
    optimize_features = checkpoint.get('optimize_features', False)
    feature_hidden_factor_dict = checkpoint.get('feature_hidden_factor_dict', {})
    
    print("模型配置:")
    print(f"  特征类型: {feature_types}")
    print(f"  特征优化: {optimize_features}")
    
    fan_dim = state_dict['fan.addNorm.ln.weight'].shape[1]
    print(f"  FAN层维度: {fan_dim}")
    
    feature_dims = {}
    
    # 对于优化模型，从特征变换器权重推断维度
    if optimize_features:
        for feature_name in feature_types:
            if feature_name != 'textcnn':
                key = f"feature_transformers.{feature_name}.transform.0.weight"
                if key in state_dict:
                    feature_dims[feature_name] = state_dict[key].shape[1]
                    print(f"  特征 {feature_name} 维度: {feature_dims[feature_name]}")
    else:
        # 对于非优化模型，需要从特征加载器获取维度信息
        # 创建临时的特征加载器来获取特征维度
        temp_feature_loader = FeatureLoader(features_dir='features')
        
        for feature_name in feature_types:
            if feature_name != 'textcnn':
                try:
                    # 尝试加载一个特征文件来获取维度
                    temp_feature = temp_feature_loader.load_feature(feature_name, 'train')
                    feature_dims[feature_name] = temp_feature.shape[1]
                    print(f"  特征 {feature_name} 维度: {feature_dims[feature_name]}")
                except Exception as e:
                    print(f"  警告：无法加载特征 {feature_name} 来获取维度: {e}")
                    # 如果无法加载特征，尝试从FAN层维度推断
                    continue
        
        # 如果仍然无法获取所有特征维度，尝试从FAN层维度推断
        if not feature_dims and feature_types != ['textcnn']:
            print("  尝试从FAN层维度推断特征维度...")
            
            # 计算textcnn的维度（如果使用）
            textcnn_dim = 0
            if 'textcnn' in feature_types:
                # 从状态字典推断max_pool
                conv_keys = [k for k in state_dict.keys() if k.startswith('conv') and 'weight' in k]
                if conv_keys:
                    max_pool = 3  # 默认值
                    if abs(fan_dim - 6016) < 100:
                        max_pool = 2
                        textcnn_dim = 6016
                    elif abs(fan_dim - 3968) < 100:
                        max_pool = 3
                        textcnn_dim = 3968
                    elif abs(fan_dim - 2944) < 100:
                        max_pool = 4
                        textcnn_dim = 2944
                    elif abs(fan_dim - 2304) < 100:
                        max_pool = 5
                        textcnn_dim = 2304
                    else:
                        textcnn_dim = 2304
            
            # 如果只有一个非textcnn特征，可以直接计算其维度
            non_textcnn_features = [f for f in feature_types if f != 'textcnn']
            if len(non_textcnn_features) == 1:
                feature_name = non_textcnn_features[0]
                feature_dims[feature_name] = fan_dim - textcnn_dim
                print(f"  推断特征 {feature_name} 维度: {feature_dims[feature_name]}")
    
    # 确定max_pool参数
    if 'textcnn' in feature_types:
        cnn_shape = fan_dim - sum(feature_dims.values())
        print(f"  计算得到的CNN形状: {cnn_shape}")
        
        max_pool = 3
        if abs(cnn_shape - 6016) < 100:
            max_pool = 2
        elif abs(cnn_shape - 3968) < 100:
            max_pool = 3
        elif abs(cnn_shape - 2944) < 100:
            max_pool = 4
        elif abs(cnn_shape - 2304) < 100:
            max_pool = 5
        print(f"  推断的max_pool: {max_pool}")
    else:
        max_pool = 3
    
    model = EnhancedETFC(
        vocab_size=50,
        embedding_size=config.get('embedding_size', 128),
        output_size=config.get('num_classes', 15),
        dropout=config.get('dropout', 0.5),
        fan_epoch=config.get('fan_epochs', 1),
        num_heads=config.get('num_heads', 8),
        feature_types=feature_types,
        feature_dims=feature_dims,
        max_pool=max_pool,
        optimize_features=optimize_features,
        feature_hidden_factor_dict=feature_hidden_factor_dict
    )
    
    if 'textcnn' in feature_types:
        model.cnn_shape = cnn_shape
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, feature_types, optimize_features


def load_test_data(data_dir, max_length):
    """
    加载和预处理测试数据
    """
    test_dir = os.path.join(data_dir, 'test')
    test_sequence_data, test_sequence_label = getSequenceData(test_dir, 'test')
    y_test = np.array(test_sequence_label)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
    x_test_tensor = torch.LongTensor(x_test)
    test_length_tensor = torch.LongTensor(test_length)
    y_test_tensor = torch.Tensor(y_test)
    print(f"测试集: {x_test.shape[0]} 个样本")
    return x_test_tensor, y_test_tensor, test_length_tensor, test_sequence_data, test_sequence_label


def load_features(feature_types, feature_loader, split='test'):
    """
    加载指定特征类型的预提取特征
    """
    features = {}
    feature_dims = {}
    preextracted = [f for f in feature_types if f != 'textcnn']
    for name in preextracted:
        try:
            tensor = feature_loader.load_feature(name, split)
            features[name] = tensor
            feature_dims[name] = tensor.shape[1]
            print(f"加载 {name} 特征，维度为 {feature_dims[name]}")
        except Exception as e:
            print(f"加载 {name} 特征时出错: {e}")
    return features, feature_dims


def evaluate_model(y_true, y_pred_proba):
    """
    计算模型评估指标：AUC、MCC、F1、SE、SP、ACC
    """
    best_thresholds = find_best_thresholds(y_true, y_pred_proba)
    y_pred = np.zeros_like(y_pred_proba)
    for i, thr in enumerate(best_thresholds):
        y_pred[:, i] = (y_pred_proba[:, i] >= thr).astype(int)
    
    overall_acc = Accuracy(y_pred, y_true)
    overall_f1  = F1_label(y_pred, y_true)
    print(f"整体样本级别ACC: {overall_acc:.4f}")
    print(f"整体标签级别F1: {overall_f1:.4f}")
    
    num_classes = y_true.shape[1]
    class_results = []
    for i in range(num_classes):
        tc = y_true[:, i]
        pp = y_pred_proba[:, i]
        pc = y_pred[:, i]
        
        # AUC
        if len(np.unique(tc)) < 2:
            auc_score = 0.0
        else:
            try:
                auc_score = roc_auc_score(tc, pp)
            except:
                auc_score = 0.0
        
        # MCC
        try:
            mcc = matthews_corrcoef(tc, pc)
            if np.isnan(mcc): mcc = 0.0
        except:
            mcc = 0.0
        
        # F1
        try:
            f1 = f1_score(tc, pc)
            if np.isnan(f1): f1 = 0.0
        except:
            f1 = 0.0
        
        # 混淆矩阵算 SE, SP, ACC
        tn, fp, fn, tp = confusion_matrix(tc, pc).ravel()
        se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        class_results.append({
            'AUC': round(auc_score, 3),
            'MCC': round(mcc, 3),
            'F1' : round(f1, 3),
            'SE' : round(se,  3),
            'SP' : round(sp,  3),
            'ACC': round(acc, 3)
        })
    
    return overall_acc, overall_f1, class_results, best_thresholds


def save_results(overall_acc, overall_f1, class_results, y_true, y_pred_binary, 
                 test_sequences, test_labels, result_dir):
    """
    保存评估结果到CSV，并包含 SE、SP、ACC 指标
    """
    import os
    import pandas as pd

    # 确保目录存在
    os.makedirs(result_dir, exist_ok=True)

    peptide_names = [
        'AMP','TXP','ABP','AIP','AVP','ACP','AFP','DDV',
        'CPP','CCC','APP','AAP','AHTP','PBP','QSP'
    ]
    
    # 绝对路径
    metrics_path = os.path.abspath(os.path.join(result_dir, "per_class_metrics.csv"))
    seq_path     = os.path.abspath(os.path.join(result_dir, "per_seq_predict.csv"))
    
    # 打印写入位置
    print(f"正在写入每类指标 CSV：{metrics_path}")
    print(f"正在写入每序列预测 CSV：{seq_path}")

    # 写 per_class_metrics.csv —— 注意 header 中多了 SE,SP,ACC
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("sample_ACC,macro_F1\n")
        f.write(f"{round(overall_acc,3)},{round(overall_f1,3)}\n\n")
        f.write("type,AUC,MCC,F1,SE,SP,ACC\n")  # <- 新增 SE,SP,ACC
        for name, res in zip(peptide_names, class_results):
            f.write(
                f"{name},{res['AUC']},{res['MCC']},{res['F1']},"
                f"{res['SE']},{res['SP']},{res['ACC']}\n"
            )
    
    # 写 per_seq_predict.csv
    seq_data = []
    for seq, tl, pl in zip(test_sequences, test_labels, y_pred_binary):
        true_str = ''.join(map(str, tl.astype(int)))
        pred_str = ''.join(map(str, pl.astype(int)))
        seq_data.append({
            '序列': seq,
            '真实标签': true_str,
            '预测标签': pred_str
        })
    seq_df = pd.DataFrame(seq_data)
    seq_df.to_csv(seq_path, index=False, encoding='utf-8')

    print("保存完成！")



def main(args):
    set_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    model, feature_types, optimize_features = load_model(args.model_path)
    
    print("\n" + "="*60)
    print("加载测试数据")
    print("="*60)
    x_test, y_test, test_length, test_sequences, test_labels = load_test_data(args.data_dir, args.max_length)
    
    feature_loader = FeatureLoader(features_dir=args.features_dir)
    print("\n" + "="*60)
    print("加载特征")
    print("="*60)
    features, feature_dims = load_features(feature_types, feature_loader)
    model.feature_dims = feature_dims
    
    print("\n" + "="*60)
    print("生成预测")
    print("="*60)
    with torch.no_grad():
        all_preds = []
        batch_size = 32
        for i in range(0, len(x_test), batch_size):
            bx = x_test[i:i+batch_size].to(DEVICE)
            bl = test_length[i:i+batch_size].to(DEVICE)
            bf = {name: feat[i:i+batch_size].to(DEVICE) for name, feat in features.items()}
            outputs = model(bx, bl, bf)
            outputs = torch.sigmoid(outputs)
            all_preds.extend(outputs.cpu().numpy())
    y_pred = np.array(all_preds)
    y_true = y_test.numpy()
    
    print("\n" + "="*60)
    print("评估模型性能")
    print("="*60)
    overall_acc, overall_f1, class_results, best_thresholds = evaluate_model(y_true, y_pred)
    
    y_pred_binary = np.zeros_like(y_pred)
    for i, thr in enumerate(best_thresholds):
        y_pred_binary[:, i] = (y_pred[:, i] >= thr).astype(int)
    
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    save_results(overall_acc, overall_f1, class_results, y_true, y_pred_binary,
                 test_sequences, test_labels, args.result_dir)
    
    print("\n评估完成！")
    return overall_acc, overall_f1, class_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在测试集上评估训练好的模型的每个类别性能")
    parser.add_argument('--data_dir', type=str, default='dataset/pre', help='数据集目录')
    parser.add_argument('--result_dir', type=str, default='result/per_class_eval', help='结果保存目录')
    parser.add_argument('--features_dir', type=str, default='features', help='特征目录')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型检查点路径')
    parser.add_argument('--max_length', type=int, default=50, help='最大序列长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    if args.model_path is None:
        print("错误: 必须指定模型路径 (--model_path)")
        parser.print_help()
        exit(1)
    
    main(args)
