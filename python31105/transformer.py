import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 IMDB 数据集
dataset = load_dataset("imdb")

# 使用 Hugging Face 的分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 对文本数据进行编码
def encode_dataset(d):
    return tokenizer(d['text'], padding='max_length', truncation=True, max_length=128)

# 处理数据集
train_data = dataset['train'].map(encode_dataset, batched=True)
test_data = dataset['test'].map(encode_dataset, batched=True)

# 转换为 PyTorch 数据加载器
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# 加载预训练模型，并设置模型为分类任务
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练模型
num_epochs = 3
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
        labels = batch['label'].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 测试模型并可视化结果
def evaluate_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
            labels = batch['label'].numpy()

            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=1)

            predictions.extend(predicted_labels)
            true_labels.extend(labels)

    return true_labels, predictions

# 评估模型
true_labels, predictions = evaluate_model(model, test_loader)

# 打印分类报告
print(classification_report(true_labels, predictions))

# 可视化训练损失
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.grid()
plt.show()