import pandas as pd
import numpy as up
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import joblib
from sklearn.model_selection import train_test_split

# 标签映射字典
label_map = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4,
            '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9,
            '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

# 读取数据 - 确保正确解析表头
try:
    train_df = pd.read_csv('data/train_set.csv', sep='\t')
    # 检查列名是否正确
    if 'label' not in train_df.columns or 'text' not in train_df.columns:
        train_df = pd.read_csv('data/train_set.csv', sep='\t', header=None, names=['label', 'text'])
except:
    train_df = pd.read_csv('data/train_set.csv', sep='\t', header=None, names=['label', 'text'])

try:
    test_a = pd.read_csv('data/test_a.csv', sep='\t')
    if 'text' not in test_a.columns:
        test_a = pd.read_csv('data/test_a.csv', sep='\t', header=None, names=['text'])
except:
    test_a = pd.read_csv('data/test_a.csv', sep='\t', header=None, names=['text'])

# 数据预处理
train_df['text'] = train_df['text'].astype(str)
test_a['text'] = test_a['text'].astype(str)

# 检查标签分布
label_counts = train_df['label'].value_counts()
print("标签分布统计:")
print(label_counts)

# 处理样本量不足的类别 - 确保每个类别至少有2个样本
min_samples = 2
for label, count in label_counts.items():
    if count < min_samples:
        print(f"类别 {label} 只有 {count} 个样本，将进行过采样...")
        # 复制现有样本直到达到最小样本数
        additional_samples = train_df[train_df['label'] == label].sample(
            n=min_samples - count, replace=True, random_state=42)
        train_df = pd.concat([train_df, additional_samples])

# 创建新的标签分布统计
new_label_counts = train_df['label'].value_counts()
print("\n调整后的标签分布:")
print(new_label_counts)

# 分割训练验证集 - 使用分层抽样
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'], train_df['label'],
    test_size=0.2, random_state=42, stratify=train_df['label']
)

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=30000,  # 减少特征数量以加快处理速度
    token_pattern=r'\b\d+\b'
)

# 创建SVM分类器
svm = LinearSVC(
    C=0.8,
    class_weight='balanced',
    dual=False,
    max_iter=1000,
    random_state=42
)

# 创建处理管道
pipeline = make_pipeline(vectorizer, svm)

# 训练模型
print("\n开始训练模型...")
pipeline.fit(X_train, y_train)

# 验证集评估
val_preds = pipeline.predict(X_val)
f1 = f1_score(y_val, val_preds, average='macro')
print(f"\n验证集 Macro F1 分数: {f1:.4f}")

# 预测测试集
print("\n预测测试集...")
test_preds = pipeline.predict(test_a['text'])

# 生成提交文件
submit = pd.DataFrame({
    'id': range(len(test_preds)),
    'label': test_preds
})
submit.to_csv('submit.csv', index=False)
print("\n提交文件已保存为 submit.csv")

# 保存模型 (可选)
joblib.dump(pipeline, 'tfidf_svm_model.pkl')
print("模型已保存为 tfidf_svm_model.pkl")