"""
训练模块，包含模型训练和评估的核心功能
"""
import torch
import torch.nn as nn
import numpy as np
import os
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from evaluation import evaluate, visualize_predictions, find_best_thresholds


class FocalDiceLoss(nn.Module):
    """多标签焦点Dice损失"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        """
        初始化FocalDiceLoss
        
        Args:
            p_pos: 正例的幂次
            p_neg: 负例的幂次
            clip_pos: 正例的裁剪值
            clip_neg: 负例的裁剪值
            pos_weight: 正例的权重
            reduction: 减少方式('mean', 'sum', 'none')
        """
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        """
        计算损失
        
        Args:
            input: 模型输出 
            target: 目标标签
            
        Returns:
            计算的损失值
        """
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # 处理正例
        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        # 处理负例
        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        # 计算最终损失
        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        
        # 根据reduction方式返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class CosineScheduler:
    """余弦退火学习率调度器，带预热"""
    
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        """
        初始化调度器
        
        Args:
            max_update: 最大更新次数
            base_lr: 基础学习率
            final_lr: 最终学习率
            warmup_steps: 预热步数
            warmup_begin_lr: 预热初始学习率
        """
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        """获取预热阶段的学习率"""
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        """获取当前轮的学习率"""
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def create_scheduler(scheduler_type, learning_rate, warmup_steps, max_steps):
    """
    创建学习率调度器
    
    Args:
        scheduler_type: 调度器类型 ('constant', 'cosine', 'linear')
        learning_rate: 基础学习率
        warmup_steps: 预热步数
        max_steps: 最大步数
        
    Returns:
        学习率调度函数
    """
    if scheduler_type == 'cosine':
        scheduler = CosineScheduler(
            max_update=max_steps,
            base_lr=learning_rate,
            final_lr=learning_rate * 0.1,
            warmup_steps=warmup_steps,
            warmup_begin_lr=learning_rate * 0.1
        )
        return scheduler
    elif scheduler_type == 'linear':
        def linear_scheduler(step):
            if step < warmup_steps:
                return learning_rate * 0.1 + (learning_rate - learning_rate * 0.1) * (step / warmup_steps)
            else:
                ratio = (step - warmup_steps) / (max_steps - warmup_steps)
                return learning_rate * (1 - 0.9 * ratio)
        return linear_scheduler
    elif scheduler_type == 'constant':
        def constant_scheduler(step):
            if step < warmup_steps:
                return learning_rate * 0.1 + (learning_rate - learning_rate * 0.1) * (step / warmup_steps)
            else:
                return learning_rate
        return constant_scheduler
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def train_model(model, train_loader, val_loader, args, device, result_dir):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        args: 配置参数
        device: 计算设备
        result_dir: 结果保存目录
        
    Returns:
        训练好的模型和训练历史
    """
    # 创建优化器
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # 创建损失函数
    criterion = FocalDiceLoss(
        clip_pos=args.clip_pos,
        clip_neg=args.clip_neg,
        pos_weight=args.pos_weight
    )
    
    # 估计最大步数
    max_steps = len(train_loader) * 1000  # 假设最多1000轮
    
    # 创建学习率调度器
    scheduler = create_scheduler(
        args.scheduler_type, 
        args.learning_rate, 
        args.warmup_steps, 
        max_steps
    )
    
    # 训练变量
    best_val_score = 0
    best_epoch = 0
    patience = 50
    counter = 0
    steps = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_metrics': [],
        'feature_adaptation': []  # 记录特征适应层的权重变化（如果启用特征优化）
    }
    
    # 训练循环
    print("开始训练模型...")
    for epoch in range(1, 1000):  # 最多1000轮
        # 训练
        model.train()
        train_loss = 0
        batch_count = 0
        
        train_progress = tqdm(train_loader, desc=f"轮次 {epoch}")
        for batch in train_progress:
            # 基本输入数据
            x = batch[0].to(device).long()
            y = batch[1].to(device).float()
            z = batch[2].to(device)
            
            # 准备特征参数
            features = {}
            
            # 从批次中提取特征
            feature_idx = 3
            for feature_name in model.feature_types:
                if feature_name != 'textcnn' and feature_idx < len(batch):
                    features[feature_name] = batch[feature_idx].to(device).float()
                    feature_idx += 1
            
            # 前向传播
            try:
                y_hat = model(x, z, features)
                loss = criterion(y_hat, y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（避免梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 更新学习率
                if hasattr(scheduler, 'step'):
                    scheduler.step()
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = scheduler(steps + 1)
                
                # 显示当前损失
                train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
                
                train_loss += loss.item()
                batch_count += 1
                steps += 1
            except Exception as e:
                print(f"训练时出错: {e}")
                print(f"批次大小: {x.shape}")
                print(f"特征: {[k for k in features.keys()]}")
                raise
        
        # 计算平均训练损失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        history['train_loss'].append(avg_train_loss)
        
        # 记录特征适应层权重（如果启用特征优化）
        if hasattr(model, 'optimize_features') and model.optimize_features:
            feature_weights = {}
            for feature_name, adapter in model.feature_adapters.items():
                # 使用第一个批次数据计算平均特征权重
                if feature_name in features:
                    with torch.no_grad():
                        # 获取特征
                        feature = features[feature_name]
                        # 应用特征适应层
                        weights = adapter.attention(feature)
                        # 计算平均权重
                        avg_weights = weights.mean(0).cpu().numpy().tolist()
                        feature_weights[feature_name] = avg_weights
            
            history['feature_adaptation'].append(feature_weights)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_labels = []
            
            print("验证: ", end="")
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证")):
                x = batch[0].to(device).long()
                y = batch[1].to(device).float()
                z = batch[2].to(device)
                
                # 准备特征参数
                features = {}
                
                # 从批次中提取特征
                feature_idx = 3
                for feature_name in model.feature_types:
                    if feature_name != 'textcnn' and feature_idx < len(batch):
                        features[feature_name] = batch[feature_idx].to(device).float()
                        feature_idx += 1
                
                try:
                    outputs = torch.sigmoid(model(x, z, features))
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
                except Exception as e:
                    print(f"验证时出错 (批次 {batch_idx}): {e}")
                    print(f"批次大小: {x.shape}")
                    print(f"特征: {[k for k in features.keys()]}")
                    raise
            
            # 评估验证集性能
            val_metrics = evaluate(np.array(val_preds), np.array(val_labels))
            history['val_metrics'].append(val_metrics)
            
            # 打印当前轮次的性能
            print(f"轮次 {epoch}: 训练损失 = {avg_train_loss:.4f}, 验证F1 = {val_metrics['macro_f1']:.4f}")
            
            # 检查是否有提升
            current_val_score = val_metrics['macro_f1']
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                best_epoch = epoch
                counter = 0
                
                # 保存最佳模型
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_score': best_val_score,
                    'args': vars(args),
                    'feature_types': model.feature_types,
                    'optimize_features': getattr(model, 'optimize_features', False),
                    'feature_hidden_factor_dict': getattr(model, 'feature_hidden_factor_dict', {})
                }
                best_model_path = os.path.join(result_dir, "best_model.pth")
                torch.save(checkpoint, best_model_path)
                print(f"保存最佳模型到: {best_model_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"触发早停，最佳轮次: {best_epoch}, 最佳验证F1: {best_val_score:.4f}")
                    break
    
    # 绘制训练曲线
    plot_training_curves(history, result_dir)
    
    # 如果启用特征优化，绘制特征权重变化
    if hasattr(model, 'optimize_features') and model.optimize_features and history['feature_adaptation']:
        plot_feature_adaptation(history['feature_adaptation'], result_dir)
    
    # 加载最佳模型
    best_model_path = os.path.join(result_dir, "best_model.pth")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存训练历史
    history_path = os.path.join(result_dir, "training_history.json")
    
    # 转换numpy数组为列表（以便JSON序列化）
    serializable_history = {
        'train_loss': history['train_loss'],
        'val_metrics': history['val_metrics'],
        'feature_adaptation': history.get('feature_adaptation', [])
    }
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    return model, history

def evaluate_model(model, test_loader, device, result_dir):
    """
    评估模型

    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        result_dir: 结果保存目录
        
    Returns:
        评估指标
    """
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_labels = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试评估")):
            x = batch[0].to(device).long()
            y = batch[1].to(device).float()
            z = batch[2].to(device)
            
            # 准备特征参数
            features = {}
            
            # 从批次中提取特征
            feature_idx = 3
            for feature_name in model.feature_types:
                if feature_name != 'textcnn' and feature_idx < len(batch):
                    features[feature_name] = batch[feature_idx].to(device).float()
                    feature_idx += 1

            try:
                outputs = torch.sigmoid(model(x, z, features))
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(y.cpu().numpy())
            except Exception as e:
                print(f"评估时出错 (批次 {batch_idx}): {e}")
                print(f"批次大小: {x.shape}")
                print(f"特征: {[k for k in features.keys()]}")
                raise
        
        # 转换为numpy数组
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        
        # 计算最佳阈值
        best_thresholds = find_best_thresholds(test_labels, test_preds)
        
        # 应用阈值获得二值化预测
        test_binary_preds = np.zeros_like(test_preds)
        for i, threshold in enumerate(best_thresholds):
            test_binary_preds[:, i] = (test_preds[:, i] >= threshold).astype(int)
        
        # 评估测试集性能
        test_metrics = evaluate(test_preds, test_labels)
        
        # 可视化预测结果
        visualize_predictions(test_labels, test_binary_preds, test_preds, result_dir)
        
        # 如果模型启用了特征优化，可视化特征变换前后的差异
        if hasattr(model, 'optimize_features') and model.optimize_features:
            visualize_feature_transformations(model, test_loader, device, result_dir)
    
    return test_metrics, np.array(best_thresholds)

def plot_training_curves(history, result_dir):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史
        result_dir: 结果保存目录
    """
    # 提取指标
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    val_f1 = [metrics['macro_f1'] for metrics in history['val_metrics']]
    
    # 绘制损失和F1曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # F1曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1)
    plt.title('Validation Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # 绘制更多指标
    metrics_to_plot = ['accuracy', 'aiming', 'coverage', 'absolute_true']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics[metric] for metrics in history['val_metrics']]
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, values)
        plt.title(f'Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'validation_metrics.png'), dpi=300)
    plt.close()

def plot_feature_adaptation(feature_adaptation, result_dir):
    """
    绘制特征适应层权重变化
    
    Args:
        feature_adaptation: 特征适应层权重历史
        result_dir: 结果保存目录
    """
    # 检查是否有足够的数据
    if not feature_adaptation or len(feature_adaptation) < 2:
        return
    
    # 获取所有特征名称
    feature_names = set()
    for epoch_weights in feature_adaptation:
        feature_names.update(epoch_weights.keys())
    
    # 为每个特征创建一个子图
    for feature_name in feature_names:
        # 提取该特征的权重历史
        weights_history = []
        for epoch_weights in feature_adaptation:
            if feature_name in epoch_weights:
                weights_history.append(epoch_weights[feature_name])
        
        # 检查是否有足够的数据
        if not weights_history:
            continue
        
        # 转换为numpy数组以便计算
        weights_history = np.array(weights_history)
        feature_dim = weights_history.shape[1]
        
        # 计算每个维度的权重平均值
        avg_weights = weights_history.mean(axis=1)
        
        # 绘制平均权重变化
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(avg_weights) + 1), avg_weights)
        plt.title(f'Feature Adaptation Weights - {feature_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Average Weight')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'feature_adaptation_{feature_name}.png'), dpi=300)
        plt.close()
        
        # 如果特征维度较小，绘制每个维度的权重变化
        if feature_dim <= 20:
            plt.figure(figsize=(12, 8))
            for i in range(feature_dim):
                plt.plot(range(1, len(weights_history) + 1), weights_history[:, i], label=f'Dim {i+1}')
            plt.title(f'Feature Adaptation Weights by Dimension - {feature_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Weight')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'feature_adaptation_dims_{feature_name}.png'), dpi=300)
            plt.close()

def visualize_feature_transformations(model, test_loader, device, result_dir):
    """
    可视化特征变换前后的差异
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        result_dir: 结果保存目录
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 从加载器中获取单个批次
    batch = next(iter(test_loader))
    
    with torch.no_grad():
        # 准备特征
        features = {}
        feature_idx = 3
        for feature_name in model.feature_types:
            if feature_name != 'textcnn' and feature_idx < len(batch):
                features[feature_name] = batch[feature_idx].to(device).float()
                feature_idx += 1
        
        # 对每个特征进行可视化
        for feature_name, feature in features.items():
            if feature_name in model.feature_transformers:
                # 获取变换前的特征
                original_feature = feature
                
                # 获取变换后的特征
                transformed_feature = model.feature_transformers[feature_name](original_feature)
                
                # 转移到CPU并转换为Numpy数组
                original_np = original_feature.cpu().numpy()
                transformed_np = transformed_feature.cpu().numpy()
                
                # 计算变换前后的差异
                diff_np = transformed_np - original_np
                
                # 计算统计信息
                n_samples = min(original_np.shape[0], 10)  # 仅使用前10个样本以避免图表拥挤
                feature_dim = original_np.shape[1]
                
                # 获取该特征使用的隐藏层因子
                factor = model.get_feature_factor(feature_name)
                factor_info = f" (因子: {factor})"
                
                # 绘制散点图比较变换前后的特征分布
                plt.figure(figsize=(15, 10))
                
                # 维度很多时选择一部分代表性维度
                if feature_dim > 20:
                    # 选择固定间隔的维度
                    selected_dims = np.linspace(0, feature_dim-1, 20, dtype=int)
                else:
                    selected_dims = np.arange(feature_dim)
                
                # 绘制特征分布比较
                for i, dim in enumerate(selected_dims):
                    if i >= 20:  # 最多绘制20个维度
                        break
                    row, col = i // 5, i % 5
                    plt.subplot(4, 5, i+1)
                    plt.scatter(range(n_samples), original_np[:n_samples, dim], c='blue', label='Original')
                    plt.scatter(range(n_samples), transformed_np[:n_samples, dim], c='red', label='Transformed')
                    plt.title(f'Dim {dim}')
                    plt.tight_layout()
                    if i == 0:
                        plt.legend()
                
                plt.suptitle(f'Feature Transformation Comparison - {feature_name}{factor_info}')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(result_dir, f'feature_transform_{feature_name}.png'), dpi=300)
                plt.close()
                
                # 绘制变化幅度热图
                plt.figure(figsize=(15, 8))
                
                # 计算变化幅度
                change_magnitude = np.abs(diff_np[:n_samples, :])
                
                # 绘制热图
                if feature_dim <= 50:
                    plt.imshow(change_magnitude, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Absolute Change')
                    plt.xlabel('Feature Dimension')
                    plt.ylabel('Sample Index')
                    plt.title(f'Feature Transformation Magnitude - {feature_name}{factor_info}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, f'feature_transform_heatmap_{feature_name}.png'), dpi=300)
                    plt.close()
                
                # 绘制平均变化幅度
                plt.figure(figsize=(10, 6))
                mean_change = np.mean(change_magnitude, axis=0)
                plt.bar(range(feature_dim), mean_change)
                plt.xlabel('Feature Dimension')
                plt.ylabel('Mean Absolute Change')
                plt.title(f'Mean Feature Transformation Magnitude - {feature_name}{factor_info}')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, f'feature_transform_mean_{feature_name}.png'), dpi=300)
                plt.close()