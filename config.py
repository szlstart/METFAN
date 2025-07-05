"""
配置文件，包含模型参数和训练设置
"""
import argparse
import os
import torch
import json

def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='多功能肽分类项目')
    
    # 路径设置
    parser.add_argument('--data_dir', type=str, default='dataset/pre',
                        help='数据集目录 (default: dataset/pre)')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='结果保存目录 (default: result)')
    parser.add_argument('--features_dir', type=str, default='features',
                        help='特征目录 (default: features)')
    parser.add_argument('--model_path', type=str, default='',
                        help='预训练模型路径，为空则从头训练')

    # 通用参数
    parser.add_argument('--num_classes', type=int, default=15,
                        help='输出类别数量 (default: 15)')
    parser.add_argument('--max_length', type=int, default=50,
                        help='最大序列长度 (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')

    # 特征选择参数 - 支持多个特征
    parser.add_argument('--features', type=str, nargs='+', 
                    default=['textcnn'],
                    help='要使用的特征类型，可以多选 (default: textcnn)')
                    
    # 特征优化参数
    parser.add_argument('--optimize_features', action='store_true',
                    help='是否对预提取特征进行优化 (default: False)')
    parser.add_argument('--feature_hidden_factors', type=str, nargs='+', default=[],
                    help='特征特定的隐藏层大小因子，格式为：feature_name=factor，例如：prot_t5_mean=0.4 esm1v_max=0.6')
                        
    # 特征提取器选项
    parser.add_argument('--num', type=int, default=0,
                    help='实验编号 (default: 0)')
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='运行模式: train=训练模型, evaluate=评估模型')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    
    # 解析特征特定的隐藏层因子
    args.feature_hidden_factor_dict = {}
    for factor_str in args.feature_hidden_factors:
        try:
            feature_name, factor_value = factor_str.split('=')
            args.feature_hidden_factor_dict[feature_name] = float(factor_value)
        except ValueError:
            print(f"警告: 无效的特征隐藏层因子格式: {factor_str}. 应该为 'feature_name=factor_value'")
    
    # 添加预定义的特征隐藏层因子默认值（如果未通过命令行指定）
    default_feature_factors = {
        'prot_t5_mean': 0.4,
        'esm1v_max': 0.6,
        # 可以添加更多特征的默认因子
    }
    
    # 添加最佳参数
    best_params = {
        "num_heads": 8,
        "clip_pos": 0.5,
        "clip_neg": 0.3,
        "pos_weight": 0.9,
        "embedding_size": 128,
        "embedding_size_factor": 16,
        "fan_epochs": 1,
        "max_pool": 3,
        "batch_size": 256,
        "learning_rate": 0.0005033107271891663,
        "dropout": 0.7641795882302234,
        "weight_decay": 4.1993525936447685e-06,
        "warmup_steps": 400,
        "scheduler_type": "constant",
        "optimizer_type": "adam"
    }
    
    # 将最佳参数添加到args
    for key, value in best_params.items():
        setattr(args, key, value)
    
    return args

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保可重现性
def set_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)