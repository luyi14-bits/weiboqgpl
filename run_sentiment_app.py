import jieba
import joblib
import gradio as gr

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def predict_sentiment(text):
    if not text.strip():
        return "请输入有效文本"
    words = " ".join(jieba.lcut(str(text)))
    vec = vectorizer.transform([words])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "😊 正面评价" if pred == 1 else "😠 负面评价"
    confidence = f"置信度：{max(proba) * 100:.1f}%"
    return f"{label}\n{confidence}"


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="请输入一句中文评论", label="📝 输入文本"),
    outputs=gr.Textbox(label="🔍 分析结果"),
    title="微博评论情感分析系统",
    description="基于朴素贝叶斯算法 + TF-IDF 特征，测试准确率 78.65%",
    examples=[
        ["这部电影太好看了，强烈推荐！"],
        ["服务态度太差，再也不会来了"],
        ["今天天气真好，心情特别棒"]
    ]
)

demo.launch(server_name="127.0.0.1", server_port=9000, share=False)
