# 微博评论情感分析系统

基于朴素贝叶斯 + TF-IDF 的中文微博评论情感分析系统，支持通过 Gradio Web 界面交互。

## 功能

- 输入中文评论文本，自动判断情感倾向（正面 / 负面）
- 输出情感标签与置信度
- Gradio Web 可视化交互界面

## 技术栈

| 组件 | 技术 |
|------|------|
| 中文分词 | jieba |
| 特征提取 | TF-IDF (TfidfVectorizer, max_features=5000) |
| 分类模型 | 多项式朴素贝叶斯 (MultinomialNB) |
| Web 界面 | Gradio |
| 模型持久化 | joblib |

## 数据集

- 数据量：119,988 条带标注微博评论
- 标签：0 = 负面，1 = 正面
- 训练集：8,000 条 | 测试集：2,000 条
- 准确率：**78.95%**

## 快速开始

### 安装依赖

```bash
pip install pandas jieba scikit-learn joblib gradio
```

### 启动 Web 服务

```bash
python run_sentiment_app.py
```

浏览器打开 **http://127.0.0.1:9000** 即可使用。

### 重新训练模型（可选）

在 Jupyter Notebook 中依次运行 `Untitled.ipynb` 的代码单元即可完成数据加载 → 分词 → 特征提取 → 模型训练 → 模型保存。

## 推理流程

```
输入文本 → jieba 分词 → TF-IDF 向量化 → MultinomialNB 预测 → 输出标签 + 置信度
```

## 模型评估

| 指标 | 负面 (0) | 正面 (1) |
|------|----------|----------|
| Precision | 0.83 | 0.76 |
| Recall | 0.73 | 0.85 |
| F1-score | 0.78 | 0.80 |

## 项目结构

```
├── Untitled.ipynb          # 数据处理 + 模型训练 Notebook
├── weibo_senti_100k.csv    # 原始数据集
├── sentiment_model.pkl     # 训练好的模型
├── tfidf_vectorizer.pkl    # TF-IDF 向量化器
├── run_sentiment_app.py    # Gradio 推理服务入口
└── 技术文档.md              # 详细技术文档
```

## 优化方向

- 全量数据训练、替换 SVM / XGBoost 提升准确率
- 引入 BERT 预训练模型增强语义理解
- 扩展为正面 / 中性 / 负面三分类
- Docker 容器化部署

## License

MIT
