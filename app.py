import pickle
import gradio as gr

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Define prediction function
def predict(text):
    if not text.strip():
        return "❗ Please enter some news content."
    vector = tfidf.transform([text]).toarray()
    pred = model.predict(vector)[0]
    return "✅ REAL News" if pred == 1 else "❌ FAKE News"

# Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="News Content", lines=10, placeholder="Paste a news article here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector",
    description="Enter a news article to check if it's REAL or FAKE using a trained machine learning model."
).launch()
