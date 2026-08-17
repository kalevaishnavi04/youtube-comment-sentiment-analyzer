# 🎥 YouTube Comment Sentiment Analyzer

> **Business question:** What is the overall audience sentiment toward a YouTube video, and how does sentiment differ between top-level comments and replies?

**Answer:** This project collects YouTube comments in real time, classifies them as **Positive, Negative, or Neutral**, and compares sentiment patterns between top-level comments and replies using **TextBlob and multilingual BERT**.

---

## 📌 1. Business Context

Understanding audience feedback is important for creators, marketers, and media teams because thousands of YouTube comments can contain valuable information about viewer opinions, reactions, and potential issues.

The goal of this project is to transform unstructured YouTube comments into measurable sentiment insights that can be used to understand audience perception.

The analysis focuses on two types of conversations separately:

* **Top-level comments** — direct reactions to the video
* **Replies** — follow-up discussions between viewers

Two NLP approaches are compared to understand the trade-off between **speed and contextual accuracy**.

---

## 📊 2. Data

### Data Source

The project uses the **YouTube Data API v3** to collect comments from any public YouTube video provided by the user.

### Data Grain

Each row represents **one individual comment or reply**.

### Data Coverage

The application dynamically collects comments from the selected YouTube video at analysis time.

Therefore, the dataset is **video-specific and dynamically generated**, rather than a fixed historical dataset.

### Data Volume

The number of records depends on the selected video and the available comments/replies.

The application supports analyzing potentially thousands of comments, subject to YouTube API limitations and the amount of discussion available on the video.

### Schema

| Column              | Description                                    | Type        |
| ------------------- | ---------------------------------------------- | ----------- |
| `comment_id`        | Unique identifier of the comment               | String      |
| `video_id`          | YouTube video identifier                       | String      |
| `comment_text`      | Text written by the viewer                     | String      |
| `author`            | Comment author identifier/name where available | String      |
| `published_at`      | Comment publication timestamp                  | Datetime    |
| `comment_type`      | Top-level comment or reply                     | Categorical |
| `parent_comment_id` | Parent comment ID for replies                  | String      |
| `sentiment`         | Positive, Negative, or Neutral                 | Categorical |
| `sentiment_score`   | Model-specific sentiment score                 | Numeric     |
| `model`             | Sentiment engine used                          | Categorical |

---

# 🧠 3. Methodology

## Step 1 — YouTube Data Collection

The user provides a YouTube video URL.

The application extracts the video ID and uses the **YouTube Data API v3** to retrieve:

1. Top-level comments
2. Replies associated with those comments

Comments and replies are retained as separate records because replies represent a different level of audience interaction.

---

## Step 2 — Data Cleaning

Before sentiment analysis, the text is cleaned to reduce noise.

Typical preprocessing includes:

* Removing unnecessary whitespace
* Handling empty comments
* Removing duplicate records where necessary
* Preserving the original comment text for analysis
* Separating comments from replies

The original text is retained because transformer-based models can benefit from contextual information that may be lost through aggressive preprocessing.

---

## Step 3 — Sentiment Analysis

Two different NLP approaches are implemented.

### Approach 1 — TextBlob

**TextBlob** provides a lightweight, lexicon-based sentiment approach.

It was selected because:

* It is fast
* It requires relatively little computational power
* It is easy to interpret
* It provides a useful baseline for comparison

However, TextBlob is primarily designed around English text and may struggle with multilingual comments, slang, sarcasm, and context.

### Approach 2 — Multilingual BERT

The project uses:

`nlptown/bert-base-multilingual-uncased-sentiment`

This transformer model was selected because it can capture contextual relationships within text and provides better support for multilingual user-generated content.

This is particularly useful for YouTube comments containing combinations of:

* English
* Hindi
* Marathi
* Informal language
* Mixed-language expressions

The BERT approach is computationally more expensive than TextBlob, but it can provide a stronger contextual baseline.

---

## Step 4 — Sentiment Classification

The model output is converted into three business-friendly categories:

| Sentiment   | Interpretation                                           |
| ----------- | -------------------------------------------------------- |
| 🟢 Positive | Favorable or supportive reaction                         |
| 🔴 Negative | Critical, dissatisfied, or unfavorable reaction          |
| ⚪ Neutral   | Informational, mixed, or relatively emotionless reaction |

---

## Step 5 — Comment vs. Reply Analysis

Instead of combining everything into one sentiment distribution, the application analyzes:

### Top-Level Comments

These represent direct audience reactions to the video.

### Replies

These represent conversations occurring underneath individual comments.

This distinction helps identify whether sentiment changes during audience discussions.

---

## Step 6 — Visualization

The application generates visual summaries for both comments and replies.

### Sentiment Distribution

Pie charts show the proportion of:

* Positive
* Negative
* Neutral

sentiment.

### Word Cloud

Word clouds highlight frequently occurring words within the analyzed text.

These visualizations make large volumes of unstructured comments easier to interpret.

---

# 📈 4. Key Findings

The dashboard is designed to report findings from the selected video rather than using predetermined numbers.

After running an analysis, the following metrics should be reported:

### Finding 1 — Overall Audience Sentiment

**[X%]** of analyzed top-level comments were classified as positive, **[Y%]** as negative, and **[Z%]** as neutral.

This provides an overall measure of audience perception.

### Finding 2 — Reply Sentiment

Replies showed **[X%] positive / [Y%] negative / [Z%] neutral** sentiment.

Comparing this with top-level comments helps determine whether audience discussions become more positive or negative.

### Finding 3 — Comment vs. Reply Difference

The difference between comment sentiment and reply sentiment was **[X percentage points]**.

A significant difference may indicate that audience discussions develop a different emotional tone from initial reactions.

### Finding 4 — Model Comparison

TextBlob and multilingual BERT produced different sentiment classifications for **[X%]** of analyzed text.

This demonstrates the impact of using a contextual transformer model compared with a lightweight lexicon-based baseline.

### Finding 5 — Dominant Discussion Topics

The most frequently occurring words included:

**[word 1] · [word 2] · [word 3] · [word 4] · [word 5]**

These terms provide an indication of the topics generating the most audience discussion.

> **Note:** Replace the `[X]` values with actual results generated by the application. Do not put invented numbers in the portfolio README.

---

# 💡 5. Recommendations

Based on the sentiment results, the analysis can support the following actions.

### For Content Creators

If negative sentiment exceeds the expected level, review the most frequently occurring negative topics and identify recurring viewer concerns.

### For Marketing Teams

Use highly positive topics and frequently mentioned themes to understand which aspects of the content resonate most strongly with the audience.

### For Community Managers

Prioritize highly negative or rapidly escalating discussion threads for manual review.

### For Product/Content Teams

Compare sentiment across multiple videos to identify recurring audience preferences and problems.

### Recommended Decision Framework

| Finding                               | Recommended Action                          | Owner                |
| ------------------------------------- | ------------------------------------------- | -------------------- |
| High positive sentiment               | Identify successful content themes          | Content Team         |
| High negative sentiment               | Investigate recurring complaints            | Content/Product Team |
| Negative replies significantly higher | Review discussion threads                   | Community Team       |
| Strong recurring keywords             | Use themes in future content                | Content Team         |
| Large TextBlob/BERT disagreement      | Prefer contextual model for deeper analysis | Data/ML Team         |

---

# ⚠️ 6. Limitations & Assumptions

### YouTube API Limitations

The analysis depends on comments accessible through the YouTube Data API and is subject to API quotas and availability.

### Sentiment Is Not Perfect

Neither TextBlob nor BERT can reliably understand every example of:

* Sarcasm
* Humor
* Slang
* Memes
* Cultural references
* Code-switching
* Ambiguous statements

### Multilingual Performance

Although the BERT model supports multiple languages, performance may vary across Hindi, Marathi, English, and mixed-language comments.

### Sentiment ≠ Opinion

A positive sentiment does not necessarily mean the viewer approves of every aspect of the video.

Similarly, negative sentiment does not automatically indicate a genuine product or content problem.

### No Causal Conclusions

Comment sentiment shows **audience reactions**, but it cannot establish why viewers behaved a certain way or whether sentiment caused changes in views, likes, subscriptions, or revenue.

### Sampling Constraints

The results depend on the comments available for the selected video and the API's accessible data.

---

# 🗂️ 7. Repository Guide

```text
youtube-comment-sentiment-analyzer/
│
├── main.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── .env
│   └── YouTube API credentials
│
├── .gitignore
│   └── Prevents secrets and unnecessary files from being committed
│
├── README.md
│   └── Project documentation
│
└── assets/
    └── Screenshots / project visuals
```

> **Important:** Never commit `.env` or your YouTube API key to GitHub.

---

# 🛠️ 8. Tech Stack

| Technology                | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| Python                    | Core programming language                  |
| Streamlit                 | Interactive web application                |
| YouTube Data API v3       | Comment collection                         |
| TextBlob                  | Lightweight sentiment baseline             |
| Hugging Face Transformers | Transformer-based NLP                      |
| BERT                      | Contextual multilingual sentiment analysis |
| WordCloud                 | Frequent-word visualization                |
| Matplotlib                | Data visualization                         |
| Pandas                    | Data manipulation                          |
| Python-dotenv             | Environment variable management            |

---

# 🚀 9. Reproduce the Project

## Prerequisites

* Python 3.8+
* YouTube Data API v3 key
* Internet connection
* Git

### Clone the Repository

```bash
git clone https://github.com/kalevaishnavi04/youtube-comment-sentiment-analyzer.git

cd youtube-comment-sentiment-analyzer
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

Do not commit this file.

### Run the Application

```bash
streamlit run main.py
```

The application will be available at:

`http://localhost:8501`

---

# 🔄 10. How the Application Works

```text
YouTube Video URL
       ↓
Extract Video ID
       ↓
YouTube Data API
       ↓
Fetch Comments + Replies
       ↓
Data Cleaning
       ↓
┌──────────────────────┐
│   Sentiment Engine   │
├──────────────────────┤
│ TextBlob             │
│ Multilingual BERT    │
└──────────────────────┘
       ↓
Sentiment Classification
       ↓
Comment / Reply Split
       ↓
┌─────────────────────────────┐
│ Pie Charts + Word Clouds    │
│ Sentiment Comparison        │
└─────────────────────────────┘
       ↓
Audience Insights
```

---

# 📊 11. Example Business Questions

The project can be used to answer questions such as:

* Is the overall audience reaction positive or negative?
* Are viewers more positive in comments or replies?
* What topics are generating negative reactions?
* Which keywords dominate audience discussions?
* How different are TextBlob and BERT predictions?
* Does multilingual BERT provide different insights for Hindi/Marathi comments?
* Which videos generate the most positive audience reactions?

---

# 🔮 12. Future Improvements

Potential improvements include:

* Aspect-based sentiment analysis
* Emotion classification
* Topic modeling
* Sentiment trends over time
* Comment engagement analysis
* Like-count vs. sentiment analysis
* Multilingual translation
* Sarcasm detection
* Automatic identification of toxic comments
* YouTube channel-level sentiment comparison
* Database storage for historical analysis
* Model evaluation using a manually labeled test dataset

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👩‍💻 Author

**Vaishnavi Kale**

GitHub: `github.com/kalevaishnavi04`

Built as an NLP and data analytics project to transform unstructured YouTube audience feedback into actionable sentiment insights.
