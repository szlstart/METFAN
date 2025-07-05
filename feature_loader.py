"""
特征加载模块，负责加载预先提取的特征
"""
import os
import torch
import numpy as np

class FeatureLoader:
    """特征加载器，从features目录加载预提取的特征"""
    
    def __init__(self, features_dir="features"):
        """
        初始化特征加载器
        
        Args:
            features_dir: 特征目录的路径
        """
        self.features_dir = features_dir
        self.feature_dims = {}
        
        # 创建特征目录（如果不存在）
        os.makedirs(features_dir, exist_ok=True)
        
        # 获取所有可用的特征
        self.available_features = self._get_available_features()
        print(f"可用的预提取特征: {', '.join(self.available_features)}")
    
    def _get_available_features(self):
        """
        获取features目录中所有可用的特征
        
        Returns:
            可用特征列表
        """
        if not os.path.exists(self.features_dir):
            return []
            
        available_features = []
        for item in os.listdir(self.features_dir):
            if os.path.isdir(os.path.join(self.features_dir, item)):
                available_features.append(item)
        return available_features
    
    def validate_features(self, requested_features):
        """
        验证请求的特征是否可用
        
        Args:
            requested_features: 请求的特征列表
            
        Returns:
            有效的特征列表和有效的预提取特征列表（不包括textcnn）
        """
        valid_features = []
        valid_preextracted_features = []
        
        for feature in requested_features:
            # TextCNN是在模型内部实现的，不需要预提取
            if feature == 'textcnn':
                valid_features.append(feature)
                continue
                
            # 检查其他特征
            if feature in self.available_features:
                valid_features.append(feature)
                valid_preextracted_features.append(feature)
            else:
                print(f"警告: 特征 '{feature}' 在 {self.features_dir} 目录中不可用，将被忽略")
        
        if not valid_features:
            raise ValueError(f"没有有效的特征可用。请确保至少指定一个有效特征，或者使用'textcnn'")
            
        return valid_features, valid_preextracted_features
    
    def load_feature(self, feature_name, split):
        """
        加载指定特征的指定分割数据
        
        Args:
            feature_name: 特征名称
            split: 数据集分割名称 ('train', 'val', 'test')
            
        Returns:
            加载的特征张量
        """
        feature_path = os.path.join(self.features_dir, feature_name, f"{split}_{feature_name}.pt")
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
            
        try:
            print(f"加载特征: {feature_path}")
            features = torch.load(feature_path)
            
            # 如果加载的是字典，尝试获取pooled_features
            if isinstance(features, dict) and 'pooled_features' in features:
                features = features['pooled_features']
                
            if not isinstance(features, torch.Tensor):
                raise TypeError(f"加载的特征不是张量: {feature_path}")
                
            # 缓存特征维度
            if feature_name not in self.feature_dims:
                self.feature_dims[feature_name] = features.shape[1]
                print(f"特征 {feature_name} 维度: {self.feature_dims[feature_name]}")
                
            return features
        except Exception as e:
            print(f"加载特征 {feature_path} 时出错: {e}")
            raise
    
    def load_all_features(self, feature_names, splits=('train', 'val', 'test')):
        """
        加载多个特征的多个分割数据
        
        Args:
            feature_names: 特征名称列表
            splits: 需要加载的分割列表
            
        Returns:
            包含所有特征的字典 {feature_name: {split: tensor}}
        """
        all_features = {}
        
        for feature_name in feature_names:
            # 对于textcnn，跳过加载（在模型内部处理）
            if feature_name == 'textcnn':
                continue
                
            feature_dict = {}
            for split in splits:
                try:
                    feature_dict[split] = self.load_feature(feature_name, split)
                except Exception as e:
                    print(f"加载特征 {feature_name} 的 {split} 分割时出错: {e}")
                    raise
                    
            all_features[feature_name] = feature_dict
        
        return all_features
    
    def get_feature_dim(self, feature_name):
        """
        获取指定特征的维度
        
        Args:
            feature_name: 特征名称
            
        Returns:
            特征维度
        """
        if feature_name == 'textcnn':
            # TextCNN在模型内部处理，这里返回0
            return 0
            
        if feature_name not in self.feature_dims:
            # 尝试加载训练集特征以获取维度
            try:
                self.load_feature(feature_name, 'train')
            except:
                return 0
                
        return self.feature_dims.get(feature_name, 0)
    
    def get_features_dims(self, feature_names):
        """
        获取多个特征的维度字典
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征维度字典 {feature_name: dimension}
        """
        dims = {}
        for feature_name in feature_names:
            dims[feature_name] = self.get_feature_dim(feature_name)
        return dims