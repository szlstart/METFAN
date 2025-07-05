"""
模型定义，包括注意力机制、位置编码和特征优化器
"""
import math
import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """残差连接后进行层归一化"""

    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, X, y):
        return self.ln(self.dropout(y) + X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class AttentionEncode(nn.Module):
    """注意力编码"""

    def __init__(self, dropout, embedding_size, num_heads):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.at1 = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                         num_heads=num_heads,
                                         dropout=0.6
                                         )

        self.addNorm1 = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

        self.FFN = PositionWiseFFN(ffn_num_input=64, ffn_num_hiddens=192, ffn_num_outputs=64)

    def forward(self, x, y=None):
        Multi, _ = self.at1(x, x, x)
        Multi_encode = self.addNorm1(x, Multi)
        return Multi_encode


class FAN_encode(nn.Module):
    """FAN编码"""

    def __init__(self, dropout, shape):
        super(FAN_encode, self).__init__()
        self.dropout = dropout
        self.addNorm = AddNorm(normalized=[1, shape], dropout=self.dropout)
        self.FFN = PositionWiseFFN(ffn_num_input=shape, ffn_num_hiddens=(2*shape), ffn_num_outputs=shape)
    def forward(self, x):
        encode_output = self.addNorm(x, self.FFN(x))
        return encode_output


class FeatureTransformer(nn.Module):
    """
    静态特征变换器，将预提取的静态特征转换为可学习的动态特征
    使用多层感知机和残差连接进行特征变换
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.5, use_residual=True):
        super(FeatureTransformer, self).__init__()
        # 如果未指定，保持输入维度不变
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        # 使用两层MLP进行特征变换
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 是否使用残差连接
        self.use_residual = use_residual and (input_dim == output_dim)
        
    def forward(self, x):
        transformed = self.transform(x)
        if self.use_residual:
            return transformed + x  # 残差连接
        return transformed


class FeatureAdapter(nn.Module):
    """
    特征自适应层，学习特征权重和重要性
    使用通道注意力机制对特征进行加权
    """
    def __init__(self, feature_dim, reduction=4):
        super(FeatureAdapter, self).__init__()
        # 确保reduction不会导致维度过小
        reduction = min(reduction, max(2, feature_dim // 4))
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.LayerNorm(feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算特征通道注意力权重
        weights = self.attention(x)
        return x * weights  # 应用权重


def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    valid_len = valid_len.float()
    MaxLen = X.size(1)
    mask = torch.arange(MaxLen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None].to(X.device)
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)  # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class EnhancedETFC(nn.Module):
    """
    增强版ETFC模型，支持不同特征组合及特征优化
    """
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, 
                 feature_types=['textcnn'], feature_dims=None, max_pool=3, 
                 optimize_features=False, feature_hidden_factor_dict=None):
        """
        初始化增强型ETFC模型
        
        Args:
            vocab_size: 氨基酸词汇表大小
            embedding_size: 嵌入层维度
            output_size: 输出类别数量
            dropout: Dropout概率
            fan_epoch: FAN编码轮数
            num_heads: 注意力头数量
            feature_types: 要使用的特征类型列表
            feature_dims: 各特征维度的字典 {feature_name: dim}
            max_pool: 最大池化大小
            optimize_features: 是否对预提取特征进行优化
            feature_hidden_factor_dict: 特征特定的隐藏层大小因子字典 {feature_name: factor}
        """
        super(EnhancedETFC, self).__init__()
        # 检查embedding_size是否能被num_heads整除
        assert embedding_size % num_heads == 0, \
            f"embedding_size ({embedding_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout
        self.fan_epoch = fan_epoch
        self.num_heads = num_heads
        self.max_pool = max_pool
        self.feature_types = feature_types
        self.feature_dims = feature_dims or {}
        self.optimize_features = optimize_features
        self.feature_hidden_factor_dict = feature_hidden_factor_dict or {}

        # 预定义的特征隐藏层因子默认值（如果未通过特征特定的因子字典指定）
        self.default_feature_factors = {
            'prot_t5_mean': 0.4,
            'esm1v_max': 0.6,
            # 可以添加更多特征的默认因子
        }

        # 检测是否使用TextCNN
        self.use_textcnn = 'textcnn' in feature_types
        
        # 根据max_pool计算CNN输出形状（仅在使用TextCNN时）
        if self.use_textcnn:
            if max_pool == 2:
                self.cnn_shape = 6016
            elif max_pool == 3:
                self.cnn_shape = 3968
            elif max_pool == 4:
                self.cnn_shape = 2944
            elif max_pool == 5:
                self.cnn_shape = 2304
            else:
                self.cnn_shape = 2304  # 默认值
        else:
            self.cnn_shape = 0
        
        # TextCNN组件（仅在使用TextCNN时初始化）
        if self.use_textcnn:
            self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
            self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
            
            # CNN层
            self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                        out_channels=64,
                                        kernel_size=2,
                                        stride=1)
            self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1)
            self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                        out_channels=64,
                                        kernel_size=4,
                                        stride=1)
            self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                        out_channels=64,
                                        kernel_size=5,
                                        stride=1)

            self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
            self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
        
        # 创建特征变换和适应层（用于预提取特征的优化）
        if self.optimize_features:
            self.feature_transformers = nn.ModuleDict()
            self.feature_adapters = nn.ModuleDict()
            
            for feature_name in self.feature_types:
                if feature_name != 'textcnn' and feature_name in self.feature_dims:
                    input_dim = self.feature_dims[feature_name]
                    
                    # 使用特征特定的隐藏层因子
                    factor = self.get_feature_factor(feature_name)
                    hidden_dim = int(input_dim * factor)
                    
                    print(f"为特征 {feature_name} 使用隐藏层因子: {factor} (隐藏层大小: {hidden_dim})")
                    
                    # 添加特征变换器
                    self.feature_transformers[feature_name] = FeatureTransformer(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=input_dim,
                        dropout=dropout
                    )
                    
                    # 添加特征适应器
                    self.feature_adapters[feature_name] = FeatureAdapter(
                        feature_dim=input_dim
                    )
        
        # 计算融合后的特征维度
        fused_dim = self.cnn_shape  # 如果使用TextCNN则加上CNN形状，否则为0
        
        # 添加其他特征的维度
        for feature_name in self.feature_types:
            if feature_name != 'textcnn' and feature_name in self.feature_dims:
                fused_dim += self.feature_dims[feature_name]
        
        # 确保至少有一种特征被使用
        assert fused_dim > 0, "至少需要启用一种特征类型"
        
        # FAN编码和全连接层
        self.fan = FAN_encode(self.dropout, fused_dim)
            
        # 全连接层
        fc_input_dim = fused_dim
        self.full3 = nn.Linear(fc_input_dim, 1000)
        self.full4 = nn.Linear(1000, 500)
        self.full5 = nn.Linear(500, 256)
        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output_size)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def get_feature_factor(self, feature_name):
        """
        获取特征的隐藏层因子
        
        Args:
            feature_name: 特征名称
            
        Returns:
            隐藏层因子值
        """
        # 首先从用户提供的特征因子字典中查找
        if feature_name in self.feature_hidden_factor_dict:
            return self.feature_hidden_factor_dict[feature_name]
        
        # 然后从预定义的默认因子中查找
        if feature_name in self.default_feature_factors:
            return self.default_feature_factors[feature_name]
        
        # 最后使用一个通用的默认值
        return 0.8  # 硬编码的默认值

    def TextCNN(self, x):
        """应用TextCNN提取输入特征"""
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        x3 = self.MaxPool1d(x3)

        x4 = self.conv4(x)
        x4 = torch.nn.ReLU()(x4)
        x4 = self.MaxPool1d(x4)
        
        y = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.dropout_layer(y)
        x = x.view(x.size(0), -1)
        return x
        
    def optimize_feature(self, feature_name, feature_tensor):
        """
        对预提取特征进行变换和优化
        
        Args:
            feature_name: 特征名称
            feature_tensor: 特征张量
            
        Returns:
            优化后的特征张量
        """
        if not self.optimize_features:
            return feature_tensor
            
        # 应用特征变换
        if feature_name in self.feature_transformers:
            feature_tensor = self.feature_transformers[feature_name](feature_tensor)
            
        # 应用特征适应
        if feature_name in self.feature_adapters:
            feature_tensor = self.feature_adapters[feature_name](feature_tensor)
            
        return feature_tensor

    def fuse_features(self, feature_list):
        """
        融合不同来源的特征
        
        Args:
            feature_list: 包含各种特征的列表（忽略None值）
            
        Returns:
            融合后的特征
        """
        # 过滤掉None值，将剩余特征拼接
        valid_features = [f for f in feature_list if f is not None]
        
        if len(valid_features) == 0:
            raise ValueError("没有有效的特征用于融合！请确保至少有一种特征被正确加载。")
        elif len(valid_features) == 1:
            return valid_features[0]
        else:
            return torch.cat(valid_features, dim=-1)

    def forward(self, train_data, valid_lens, features=None, in_feat=False):
        """
        模型前向传播
        
        Args:
            train_data: 输入序列数据
            valid_lens: 序列有效长度
            features: 特征字典 {feature_name: tensor}
            in_feat: 是否返回中间特征（如果in_feat=True）
            
        Returns:
            模型输出或中间特征
        """
        cnn_output = None
        features = features or {}
        
        # TextCNN特征提取
        if self.use_textcnn:
            # 获取嵌入
            embed_output = self.embed(train_data)
            
            # 位置编码
            pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
            
            # 注意力编码
            attention_output = self.attention_encode(pos_output)
            
            # 特征融合
            vectors = embed_output + attention_output
            
            # TextCNN特征提取
            cnn_input = vectors.permute(0, 2, 1)
            cnn_output = self.TextCNN(cnn_input)
        
        # 构建要融合的特征列表
        feature_list = [cnn_output] if self.use_textcnn else []
        
        # 添加其他特征（可能经过优化）
        for feature_name in self.feature_types:
            if feature_name != 'textcnn' and feature_name in features:
                # 获取预提取特征
                feature = features[feature_name]
                
                # 对特征进行优化（如果启用）
                if self.optimize_features:
                    try:
                        feature = self.optimize_feature(feature_name, feature)
                    except Exception as e:
                        # 添加详细的错误信息用于调试
                        print(f"\n特征优化错误 ({feature_name}): {e}")
                        print(f"特征形状: {feature.shape}")
                        print(f"特征变换器是否存在: {feature_name in self.feature_transformers}")
                        
                        if feature_name in self.feature_transformers:
                            # 打印特征变换器的详细信息
                            transformer = self.feature_transformers[feature_name]
                            first_linear = transformer.transform[0]
                            print(f"变换器输入维度: {first_linear.in_features}")
                            print(f"变换器输出维度: {first_linear.out_features}")
                            print(f"特征维度字典: {self.feature_dims}")
                        
                        # 重新抛出异常
                        raise
                
                feature_list.append(feature)
        
        # 融合特征
        fused_features = self.fuse_features(feature_list)
        
        # FAN编码
        fan_encode = fused_features.unsqueeze(0).permute(1, 0, 2)
        for i in range(self.fan_epoch):
            fan_encode = self.fan(fan_encode)
        
        # 展平特征
        out = fan_encode.squeeze()
        
        # 全连接层
        label = self.full3(out)
        label = torch.nn.ReLU()(label)
        label1 = self.full4(label)
        label = torch.nn.ReLU()(label1)
        label2 = self.full5(label)
        label = torch.nn.ReLU()(label2)
        label3 = self.Flatten(label)
        label = torch.nn.ReLU()(label3)
        out_label = self.out(label)
        
        if in_feat:
            return label1, label2, label3, out_label
        else:
            return out_label