# 🎥 YouTube Comment Sentiment Analyzer

A Streamlit web app that fetches comments from any YouTube video and analyzes their sentiment (Positive / Negative / Neutral) using two different NLP approaches — **TextBlob** and **BERT**. Top-level comments and replies are tracked and visualized separately.

---

## 🚀 Features

* 🔗 Analyze comments from any public YouTube video (just paste the URL)
* 🧪 Choose between 2 sentiment engines:
  * **TextBlob** – fast, lightweight, word-based sentiment scoring
  * **BERT (Multilingual)** – transformer-based deep learning model, more accurate, understands context
* 💬 Comments and replies are fetched and analyzed **separately**
* 📊 Live visualizations for both comments and replies:
  * Pie chart of sentiment distribution
  * Word cloud of frequently used words
* 🌍 Multilingual support (BERT model handles Hindi/Marathi comments reasonably well)

---

## 🛠️ Setup Instructions

### 1. Prerequisites

* Python 3.8+
* A YouTube Data API v3 key ([get one here](https://console.cloud.google.com/apis/library/youtube.googleapis.com))

### 2. Clone the repository

```bash
git clone https://github.com/kalevaishnavi04/youtube-comment-sentiment-analyzer.git
cd youtube-comment-sentiment-analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

**Never commit your `.env` file** — it's already excluded via `.gitignore`.

### 6. Run the app

```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

1. Paste a YouTube video URL into the app.
2. Click "Analyze Comments."
3. The app fetches all top-level comments and their replies using the YouTube Data API.
4. Each comment/reply is analyzed for sentiment using the selected method (TextBlob or BERT).
5. Results are displayed as separate pie charts and word clouds for comments vs. replies.

---

## 📌 Notes

* BERT is more accurate but slower — a video with many comments/replies will take longer to process.
* TextBlob is faster and works well for a quick overview.
* Comments with replies disabled or videos with comments turned off will show an appropriate error.

---

## 🧰 Tech Stack

* **Streamlit** – web dashboard
* **YouTube Data API v3** – fetching comments and replies
* **TextBlob** – lightweight sentiment analysis
* **Transformers (BERT)** – `nlptown/bert-base-multilingual-uncased-sentiment` for deep sentiment analysis
* **WordCloud + Matplotlib** – visualizations

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Vaishnavi Kale**
🔗 [github.com/kalevaishnavi04](https://github.com/kalevaishnavi04)