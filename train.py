import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 用于记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # 计算训练集准确率
        train_acc = accuracy_score(train_labels, train_preds)
        
        # 验证阶段
        model.eval()
        val_acc, _, _, val_loss = evaluate_model(model, val_loader, device, criterion)
        
        # 记录历史
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config.MODELS_DIR}/best_model.pth")
            
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Training Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # 绘制训练历史
    plot_training_history(history)
    
    return history

def evaluate_model(model, data_loader, device, criterion=None):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss/len(data_loader) if criterion is not None else 0
    
    return acc, all_preds, all_labels, avg_loss

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', color='green')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # 设置y轴范围（可选）
    plt.subplot(1, 2, 2).set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['True Story', 'Deceptive Story'], 
                         normalize=False, title='Confusion Matrix'):
    """绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称
        normalize: 是否归一化
        title: 图表标题
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 添加样本总数
    total_samples = len(y_true)
    plt.text(1.5, -0.1, f'Total samples: {total_samples}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show() 