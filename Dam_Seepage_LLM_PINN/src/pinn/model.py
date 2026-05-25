import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            # 求解拉普拉斯方程，Tanh 的二阶连续性是关键
            self.net.add_module(f"activation_{i}", nn.Tanh())
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))
        
        # 权重初始化会自动继承当前的 default_dtype (Float64)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class SeepagePINN(nn.Module):
    def __init__(self, scale_x, scale_y, scale_h, k=None):
        super(SeepagePINN, self).__init__()
        self.scale_global = max(float(scale_x), float(scale_y))
        self.scale_h = float(scale_h)
        self.k = float(k) if k is not None else 1.0
        self.net = MLP(layers=[2, 64, 64, 64, 64, 1])

    def forward_head(self, xy_phys):
        # 自动适应 model 的 dtype (可能是 float64)
        target_dtype = next(self.net.parameters()).dtype
        if xy_phys.dtype != target_dtype:
            xy_phys = xy_phys.to(target_dtype)
        xy_norm = xy_phys / self.scale_global
        return self.net(xy_norm) * self.scale_h