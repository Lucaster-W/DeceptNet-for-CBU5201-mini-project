import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

class AudioCNN(nn.Module):
    def __init__(self, input_size):
        super(AudioCNN, self).__init__()
        
        # 调整输入层，将特征重塑为 [batch_size, 1, feature_size]
        self.input_size = input_size
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 计算展平后的特征维度
        self.flat_size = self._get_flat_size()
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
    def _get_flat_size(self):
        # 计算卷积层输出的特征维度
        x = torch.randn(1, 1, self.input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # 确保输入形状正确 [batch_size, 1, feature_size]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 