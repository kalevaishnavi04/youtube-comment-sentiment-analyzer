# 🎥 YouTube Comment Sentiment Analyzer

> **Business question:** What are viewers saying, feeling, and discussing about a YouTube video — and can we identify emerging topics, toxic content, language patterns, and bot-like behavior from their comments?

**Answer:** This project builds an end-to-end NLP analytics pipeline that collects YouTube comments and replies and transforms them into actionable audience insights using **TextBlob, multilingual BERT, TF-IDF, toxicity detection, language detection, and behavioral analysis**.

---

# 1. 🎯 Business Context

YouTube comments contain a large amount of unstructured audience feedback, but manually reviewing thousands of comments is time-consuming and difficult to scale.

This project treats YouTube comments as an audience-feedback dataset and applies multiple NLP techniques to understand **sentiment, topics, toxicity, language, engagement behavior, and discussion trends**.

The analysis is designed for **content creators, marketing teams, community managers, and analysts** who want to understand how audiences respond to videos at scale.

The project also separates **top-level comments from replies**, allowing the initial audience reaction to be compared with the sentiment and behavior that develops during discussions.

---

# 2. 📊 Data

## Data Source

Comments are collected dynamically using the **YouTube Data API v3** from a public YouTube video URL.

The application can analyze:

* Top-level comments
* Comment replies
* Comment timestamps
* Comment text
* Comment metadata available through the API

## Data Grain

**One row = one individual YouTube comment or reply.**

## Time Span

The time span is determined dynamically by the selected video.

For example, if a video was published in January 2026 and comments were collected in August 2026, the analysis can contain approximately seven months of audience discussion.

## Data Volume

The number of records depends on the selected video's comment activity and API availability.

The application is designed to handle **thousands of comments and replies**, subject to YouTube API quotas and the amount of publicly available discussion.

---

## Data Schema

| Column              | Description                                | Type        |
| ------------------- | ------------------------------------------ | ----------- |
| `comment_id`        | Unique comment identifier                  | String      |
| `video_id`          | YouTube video identifier                   | String      |
| `comment_text`      | Original viewer comment                    | String      |
| `author`            | Comment author information where available | String      |
| `published_at`      | Comment publication timestamp              | Datetime    |
| `comment_type`      | Top-level comment or reply                 | Categorical |
| `parent_comment_id` | Parent comment ID for replies              | String      |
| `sentiment`         | Positive / Negative / Neutral              | Categorical |
| `sentiment_score`   | Sentiment confidence/score                 | Numeric     |
| `sentiment_model`   | TextBlob or BERT                           | Categorical |
| `language`          | Detected comment language                  | Categorical |
| `toxicity`          | Toxic / non-toxic classification           | Categorical |
| `toxicity_score`    | Toxicity model score                       | Numeric     |
| `is_duplicate`      | Duplicate/repeated comment flag            | Boolean     |
| `is_spam_like`      | Generic spam/praise behavior flag          | Boolean     |
| `keywords`          | Extracted important terms                  | String/List |
| `emoji_score`       | Emoji-based sentiment adjustment           | Numeric     |

---

# 3. 🧠 Methodology

The project follows an end-to-end NLP pipeline:

```text
YouTube Video URL
        ↓
YouTube Data API
        ↓
Comments + Replies
        ↓
Data Cleaning
        ↓
┌──────────────────────────────────────┐
│          NLP ANALYSIS PIPELINE       │
├──────────────────────────────────────┤
│ Sentiment Analysis                   │
│ Topic / Keyword Extraction           │
│ Toxicity Detection                   │
│ Language Detection                   │
│ Emoji Analysis                       │
│ Bot-like / Spam Detection            │
│ Extractive Summarization              │
└──────────────────────────────────────┘
        ↓
Feature Engineering
        ↓
Trend & Comparative Analysis
        ↓
Visualizations
        ↓
Audience Insights
```

---

## 3.1 Data Collection

The user provides a YouTube video URL.

The application extracts the video ID and retrieves available comments using the YouTube Data API.

Top-level comments and replies are stored separately.

This distinction is important because a direct reaction to a video represents a different type of audience behavior from a reply occurring within an existing conversation.

---

# 4. 😊 Sentiment Analysis

Two sentiment engines are implemented.

## TextBlob

TextBlob provides a lightweight lexicon-based sentiment baseline.

### Why use it?

* Fast
* Simple
* Low computational cost
* Easy to interpret
* Useful as a baseline model

However, it has limitations with:

* Multilingual text
* Slang
* Sarcasm
* Context
* Mixed-language comments

---

## Multilingual BERT

The project uses:

`nlptown/bert-base-multilingual-uncased-sentiment`

The model is used to capture contextual information that a traditional lexicon-based method may miss.

It is particularly useful for comments containing:

* English
* Hindi
* Marathi
* Informal expressions
* Mixed-language text

### Why compare both models?

The comparison demonstrates the practical trade-off between:

**Speed vs. contextual understanding**

TextBlob provides a fast baseline, while BERT provides a more sophisticated transformer-based approach.

---

# 5. 😀 Emoji-Aware Sentiment

Emojis frequently carry emotional information that traditional text preprocessing can lose.

For example:

```text
"This was amazing 🔥🔥🔥"
```

The emoji information can strengthen the positive sentiment signal.

The application therefore incorporates emoji information as an adjustment to the sentiment score instead of completely removing emojis.

This allows visual emotional expressions to contribute to the final sentiment interpretation.

---

# 6. 🔑 Topic & Keyword Extraction

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to identify important words and phrases.

TF-IDF was selected because it is:

* Computationally lightweight
* Explainable
* Easy to reproduce
* Suitable for extracting important terms from a specific comment corpus

The output identifies words that are particularly important within the selected video's discussion.

Example:

```text
AI
video
technology
tutorial
Python
future
```

These keywords help identify **what viewers are actually discussing**, rather than only determining whether they feel positively or negatively.

---

# 7. 🚫 Toxicity Detection

The optional toxicity pipeline uses:

`unitary/toxic-bert`

The model identifies potentially abusive or toxic comments.

The feature is disabled by default because transformer-based toxicity detection can significantly increase processing time.

The application can flag comments containing potentially:

* Toxic language
* Abusive language
* Harassment
* Offensive expressions

### Important

A toxicity prediction is treated as a **model-generated signal**, not as a definitive judgment about a person or comment.

---

# 8. 🌍 Language Detection

The project uses `langdetect` to identify the language of comments.

The dashboard can provide a language distribution such as:

```text
English     62%
Hindi       21%
Marathi     11%
Other        6%
```

This allows analysts to understand the linguistic composition of the audience.

Because YouTube comments frequently contain mixed-language text, language detection should be interpreted as an approximate classification rather than perfect linguistic identification.

---

# 9. 🤖 Bot-Like & Spam Detection

The application uses lightweight behavioral heuristics rather than claiming to perform definitive bot detection.

Potential signals include:

### Duplicate comments

Repeated or nearly identical comments may indicate spam or automated behavior.

### Generic praise

Repeated patterns such as generic promotional or repetitive praise can be flagged as spam-like behavior.

### Why this approach?

A true bot-detection system would require richer behavioral data such as:

* Account history
* Posting frequency
* Network behavior
* Account age
* Cross-video activity

The YouTube comment dataset alone cannot reliably establish whether an account is actually a bot.

Therefore, this project reports **"bot-like" or "spam-like" behavior**, not confirmed bots.

---

# 10. 📈 Sentiment Trends

Comments are grouped by publication date to understand how audience sentiment changes over time.

The dashboard can reveal patterns such as:

```text
Video Release
     ↓
High Initial Interest
     ↓
Positive Reaction
     ↓
Controversy / Discussion
     ↓
Negative Sentiment Increase
     ↓
Long-Term Stabilization
```

This helps identify whether audience perception remains stable or changes as discussion develops.

---

# 11. 💬 Comment vs. Reply Analysis

Comments and replies are analyzed separately.

### Top-Level Comments

Represent direct audience reactions.

### Replies

Represent deeper audience discussions.

The project compares:

* Sentiment
* Toxicity
* Language
* Keywords
* Volume
* Trends

This can reveal whether discussions become more positive, negative, or toxic after the initial reaction.

---

# 12. 📝 Extractive Summarization

Instead of calling an external LLM API, the application uses TF-IDF to select the most representative comments.

The process:

```text
Comments
   ↓
TF-IDF Vectorization
   ↓
Sentence Importance Score
   ↓
Rank Comments
   ↓
Select Representative Comments
```

This produces an **extractive summary** using real comments from the dataset.

Unlike generative AI, the method does not create new statements.

This makes the summary:

* Lightweight
* Reproducible
* API-key independent
* Based directly on viewer comments

---

# 13. ⚖️ Multi-Video Comparison

The application supports comparing **2–3 YouTube videos**.

For each video, the dashboard can compare:

* Positive sentiment %
* Negative sentiment %
* Neutral sentiment %
* Comment volume
* Reply volume
* Toxicity
* Dominant keywords

Example:

| Metric         | Video A | Video B | Video C |
| -------------- | ------: | ------: | ------: |
| Positive       |     XX% |     XX% |     XX% |
| Neutral        |     XX% |     XX% |     XX% |
| Negative       |     XX% |     XX% |     XX% |
| Toxic comments |     XX% |     XX% |     XX% |
| Comments       |   X,XXX |   X,XXX |   X,XXX |

This makes the project useful for **content performance benchmarking**.

---

# 14. 📈 Key Findings

The final README should contain **real numbers generated by your analysis**, not invented values.

For example:

### Finding 1 — Audience Sentiment

**XX%** of top-level comments were classified as positive compared with **YY%** negative.

### Finding 2 — Replies vs. Comments

Negative sentiment in replies was **X percentage points higher** than in top-level comments.

### Finding 3 — Dominant Topics

The five most important TF-IDF keywords were:

**[Keyword 1] · [Keyword 2] · [Keyword 3] · [Keyword 4] · [Keyword 5]**

### Finding 4 — Toxicity

**X%** of analyzed comments were flagged by the toxicity model.

### Finding 5 — Model Agreement

TextBlob and BERT disagreed on **X%** of comments, demonstrating the difference between lexicon-based and transformer-based sentiment analysis.

---

# 15. 💡 Recommendations

The findings can be converted into concrete actions.

| Finding                     | Recommendation                                        | Owner            |
| --------------------------- | ----------------------------------------------------- | ---------------- |
| High negative sentiment     | Investigate recurring negative topics                 | Content Team     |
| High positive sentiment     | Replicate successful content themes                   | Content Team     |
| Increasing negative trend   | Review events/topics associated with sentiment change | Marketing Team   |
| High toxicity               | Prioritize moderation of flagged discussions          | Community Team   |
| Repeated spam-like comments | Strengthen moderation/filtering rules                 | Community Team   |
| Strong topic concentration  | Use popular themes in future content                  | Content Strategy |
| Language diversity          | Consider multilingual content/subtitles               | Content Team     |
| Large model disagreement    | Use BERT for deeper analysis                          | Data/ML Team     |

---

# 16. ⚠️ Limitations & Assumptions

## YouTube API Limitations

The project depends on data available through the YouTube Data API and is subject to API quota and accessibility limitations.

## Sentiment Limitations

Sentiment models can struggle with:

* Sarcasm
* Humor
* Slang
* Memes
* Cultural references
* Mixed-language expressions

## Language Detection

`langdetect` may produce unreliable results for:

* Very short comments
* Mixed-language comments
* Comments containing only emojis
* Names or abbreviations

## Toxicity Model

A toxicity score is a model prediction and should not be treated as a definitive moderation decision.

## Bot Detection

The application identifies **bot-like patterns**, not confirmed bots.

Reliable bot detection would require additional behavioral and account-level information.

## No Causal Inference

The analysis identifies associations and patterns but cannot prove that sentiment caused changes in:

* Views
* Likes
* Subscribers
* Revenue
* Watch time

## Sampling

Results depend on the comments available for the selected video and the data returned through the API.

---

# 17. 🗂️ Repository Structure

```text
youtube-comment-sentiment-analyzer/
│
├── main.py
│   └── Streamlit application and analysis pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── .env
│   └── YouTube API key
│
├── .gitignore
│   └── Excludes secrets and unnecessary files
│
├── README.md
│   └── Project documentation
│
└── assets/
    └── Dashboard screenshots
```

If your actual repository has additional Python modules, models, utilities, or datasets, update this structure to match the real repository rather than documenting files that don't exist.

---

# 18. 🛠️ Tech Stack

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| Python                    | Core development                |
| Streamlit                 | Interactive dashboard           |
| YouTube Data API v3       | Data collection                 |
| Pandas                    | Data manipulation               |
| TextBlob                  | Baseline sentiment analysis     |
| Hugging Face Transformers | Transformer models              |
| BERT                      | Multilingual sentiment analysis |
| Toxic-BERT                | Toxicity detection              |
| Scikit-learn              | TF-IDF and feature extraction   |
| langdetect                | Language detection              |
| emoji                     | Emoji processing                |
| WordCloud                 | Text visualization              |
| Matplotlib                | Charts and trends               |

---

# 19. 🚀 Setup & Reproduction

## Prerequisites

* Python 3.8+
* YouTube Data API v3 key
* Git
* Internet connection

## Clone Repository

```bash
git clone https://github.com/kalevaishnavi04/youtube-comment-sentiment-analyzer.git

cd youtube-comment-sentiment-analyzer
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create:

```text
.env
```

Add:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

Never commit your API key.

## Run Application

```bash
streamlit run main.py
```

Open:

```text
http://localhost:8501
```

---

# 20. 🔄 End-to-End Workflow

```text
                    YouTube URL
                         │
                         ▼
                 Extract Video ID
                         │
                         ▼
                YouTube Data API
                         │
                         ▼
              Comments + Replies
                         │
                         ▼
                 Data Cleaning
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Sentiment       Language       Emoji
      Analysis        Detection      Analysis
          │              │              │
          ▼              ▼              ▼
        BERT /       langdetect      Score
      TextBlob
          │
          ├──────────────┐
          ▼              ▼
      TF-IDF         Toxic-BERT
      Keywords       (Optional)
          │              │
          └───────┬──────┘
                  ▼
          Behavioral Analysis
                  │
          ┌───────┴────────┐
          ▼                ▼
       Trends          Bot-like /
                       Spam-like
                  │
                  ▼
            Streamlit Dashboard
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
     Charts    Keywords   Summary
        │
        ▼
   Audience Insights
```

---

# 21. 📊 Dashboard Outputs

The Streamlit dashboard provides:

### Sentiment

* Positive / Negative / Neutral distribution
* TextBlob vs. BERT comparison
* Comment vs. reply sentiment

### Topics

* TF-IDF keywords
* Keyword frequency
* Word cloud

### Audience Behavior

* Comment volume
* Reply volume
* Sentiment trends
* Duplicate comments
* Spam-like patterns

### Safety

* Toxicity detection
* Toxic comment percentage

### Language

* Language distribution
* Multilingual audience insights

### Summary

* Representative viewer comments
* Extractive summary

### Comparison

* Side-by-side analysis of 2–3 videos

---

# 22. 🔮 Future Improvements

Potential improvements include:

* Aspect-based sentiment analysis
* Emotion classification
* BERTopic topic modeling
* Advanced multilingual sentiment models
* Sarcasm detection
* Improved semantic duplicate detection
* More sophisticated bot detection
* Comment engagement prediction
* Like-count vs. sentiment analysis
* Sentiment vs. video views analysis
* Historical database storage
* Channel-level analytics
* Automated weekly sentiment reports
* LLM-based generative summaries
* Real-time comment monitoring

---

# 23. 📄 License

This project is licensed under the **MIT License**.

---

# 24. 👩‍💻 Author

**Vaishnavi Kale**

GitHub: `https://github.com/kalevaishnavi04`

Built as an end-to-end **NLP, Machine Learning, and Data Analytics project** for transforming large-scale YouTube audience feedback into structured and actionable insights.
