🎥 YouTube Comment Sentiment Analyzer

Business Question: What are viewers saying about a YouTube video, and how does sentiment differ between top-level comments and replies?
Answer: This project automatically collects YouTube comments and replies, classifies them as Positive, Negative, or Neutral, and converts unstructured audience feedback into visual sentiment insights using TextBlob and multilingual BERT.

📌 1. Business Context

Understanding audience sentiment is useful for creators, marketers, brands, and content teams when evaluating how viewers respond to video content. Manually reviewing thousands of comments is time-consuming and makes it difficult to identify overall audience trends. This project was built as an NLP-based audience feedback analysis system that automatically collects YouTube comments and analyzes their sentiment. The resulting insights can help content teams identify audience reactions, recurring topics, and areas requiring attention.

📊 2. Data Source & Data Structure
Data Source

The application retrieves data directly from the YouTube Data API v3 based on a public YouTube video URL.

Data Grain

Each record represents one YouTube comment or reply.

Data Collected
Field	Description
video_id	Unique YouTube video identifier
comment_id	Unique comment identifier
comment_text	Text of the comment
comment_type	Top-level comment or reply
author	Comment author information
published_at	Comment publication timestamp
sentiment	Positive, Negative, or Neutral
sentiment_score	Sentiment confidence/score from the selected model
Data Flow
YouTube Video URL
        ↓
YouTube Data API v3
        ↓
Comments + Replies
        ↓
Text Cleaning
        ↓
Sentiment Analysis
        ↓
Positive / Negative / Neutral
        ↓
Visualization & Insights
Data Volume

The application supports analyzing the comments and replies returned by the YouTube API for a selected public video. The actual number of records depends on the video's available comments, replies, API limits, and pagination.

Time Span

The system analyzes comments based on the data available from the selected YouTube video at the time of analysis.

🧠 3. Methodology
Step 1 — Video Identification

The user enters a public YouTube video URL.

The application extracts the YouTube Video ID, which is then used to request comment data through the YouTube Data API.

Step 2 — Data Collection

The application retrieves:

Top-level comments
Replies to comments
Comment metadata

Comments and replies are intentionally stored/analyzed separately because replies represent a different level of audience interaction.

Step 3 — Text Processing

Before sentiment analysis, comment text is prepared for NLP processing.

Typical preprocessing includes:

Removing unnecessary whitespace
Handling missing values
Preparing text for model input
Creating text suitable for word-cloud generation
Step 4 — Sentiment Analysis

Two approaches are available.

🟢 TextBlob

TextBlob provides a lightweight, rule/lexicon-based approach.

Why use it?

Fast
Lightweight
Easy to interpret
Suitable for quick sentiment analysis
Comment
   ↓
TextBlob
   ↓
Polarity Score
   ↓
Positive / Negative / Neutral
🔵 Multilingual BERT

The project also uses:

nlptown/bert-base-multilingual-uncased-sentiment

This transformer-based model is designed to understand sentiment using contextual language representations.

Why use BERT instead of only TextBlob?

TextBlob is fast but can struggle with:

Context
Complex sentences
Multilingual content
Informal language

BERT provides a more context-aware alternative and can handle multilingual comments more effectively.

Step 5 — Comment vs Reply Analysis

The application separates:

Top-Level Comments
        │
        └── Sentiment Analysis


Replies
        │
        └── Sentiment Analysis

This allows the user to understand whether sentiment changes between the original audience reaction and subsequent discussions.

Step 6 — Visualization

The processed results are presented using:

📊 Sentiment Distribution

Pie charts show the percentage of:

Positive comments
Negative comments
Neutral comments
☁️ Word Cloud

Word clouds highlight frequently occurring words in:

Top-level comments
Replies
📈 4. Key Findings

The application is designed to surface the following measurable insights for each analyzed video:

1. Sentiment Distribution

The dashboard identifies the percentage of comments classified as Positive, Negative, and Neutral, providing an overall view of audience reaction.

2. Comment vs Reply Sentiment

Separate sentiment distributions allow users to compare the tone of original comments with subsequent replies and discussions.

3. Dominant Audience Topics

Word clouds highlight frequently occurring terms, helping identify the subjects and themes most discussed by viewers.

4. Multilingual Audience Feedback

The multilingual BERT model can provide sentiment analysis for comments containing languages such as English, Hindi, and Marathi, making the system more suitable for diverse audiences.

5. Model Trade-off

The project demonstrates a practical trade-off between speed and contextual understanding:

Model	Speed	Context	Multilingual
TextBlob	⚡ Fast	Basic	Limited
Multilingual BERT	🧠 Slower	Stronger	Better
💡 5. Recommendations

Based on the sentiment and topic analysis, content teams can:

📈 For Positive Sentiment

Identify topics generating strong positive reactions and consider creating additional content around them.

Owner: Content / Creator Team

🚨 For Negative Sentiment

Review recurring negative themes and distinguish between genuine product/content issues and isolated comments.

Owner: Content + Community Management Team

💬 For High-Engagement Replies

Analyze reply sentiment separately to understand where audience discussions become more positive or negative.

Owner: Community Management Team

🌍 For Multilingual Audiences

Use multilingual sentiment analysis to avoid ignoring feedback from non-English-speaking viewers.

Owner: Content Strategy Team

⚠️ 6. Limitations & Assumptions
Sentiment classification is model-based and may not correctly understand sarcasm, slang, jokes, or ambiguous statements.
TextBlob is primarily designed for English and is less suitable for multilingual sentiment analysis.
BERT is more computationally expensive and may take longer for videos with large comment volumes.
YouTube API availability and quotas can limit the amount of data retrieved.
Deleted, disabled, or unavailable comments cannot be analyzed.
Sentiment classification does not establish the reason behind a viewer's opinion.
A positive or negative comment does not necessarily represent the opinion of the entire audience.
Word frequency does not always indicate topic importance.
The project analyzes publicly available YouTube comments and should not be treated as a formal market-research survey.
🏗️ 7. Project Architecture
                    ┌─────────────────────┐
                    │   YouTube Video URL │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ YouTube Data API v3 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Comments + Replies  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Processing   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌─────────────┐       ┌──────────────┐
             │  TextBlob   │       │ Multilingual │
             │             │       │     BERT     │
             └──────┬──────┘       └──────┬───────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Sentiment Results   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌─────────────┐       ┌──────────────┐
             │ Pie Charts  │       │  Word Clouds │
             └─────────────┘       └──────────────┘
📁 8. Repository Structure
youtube-comment-sentiment-analyzer/
│
├── main.py
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── assets/
    └── screenshots/
Important Files
File	Purpose
main.py	Main Streamlit application
requirements.txt	Python dependencies
.env.example	Environment variable template
.gitignore	Prevents sensitive/unnecessary files from being committed
README.md	Project documentation
⚙️ 9. Requirements
Python 3.8+
YouTube Data API v3 key
Internet connection
Required Python libraries

Install dependencies using:

pip install -r requirements.txt
🚀 10. Reproduce the Project Locally
Step 1 — Clone Repository
git clone https://github.com/kalevaishnavi04/youtube-comment-sentiment-analyzer.git
cd youtube-comment-sentiment-analyzer
Step 2 — Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Configure API Key

Create a .env file:

YOUTUBE_API_KEY=your_youtube_api_key_here

Never commit the .env file to GitHub.

Step 5 — Run Application
streamlit run main.py

Open:

http://localhost:8501
🔐 11. API Configuration

The application requires a YouTube Data API v3 key.

Create an API key through Google Cloud Console and enable:

YouTube Data API v3

Then add the key to .env:

YOUTUBE_API_KEY=YOUR_API_KEY
🛠️ 12. Tech Stack
Frontend / Application
Streamlit
Data Collection
YouTube Data API v3
NLP
TextBlob
Hugging Face Transformers
Multilingual BERT
Data Processing
Python
Pandas
Visualization
Matplotlib
WordCloud
Environment Management
Python venv
.env
📸 13. Application Screenshots

Add screenshots here after uploading them to the repository.

![Home Page](assets/screenshots/home.png)


![Sentiment Analysis](assets/screenshots/sentiment.png)


![Comment Analysis](assets/screenshots/comments.png)


![Word Cloud](assets/screenshots/wordcloud.png)

Recommended screenshots:

Application homepage
YouTube URL input
Sentiment selection
Comment sentiment chart
Reply sentiment chart
Word clouds
Final analysis dashboard
🔮 14. Future Improvements
Real-time sentiment monitoring
Emotion classification beyond positive/negative/neutral
Sentiment trends over time
Topic modeling using LDA/BERTopic
Aspect-based sentiment analysis
Better sarcasm detection
Automatic language detection
Sentiment comparison across multiple videos
Export results to CSV/Excel
YouTube channel-level analytics
AI-generated audience insight summaries
📄 15. License

This project is licensed under the MIT License.

👩‍💻 Author
Vaishnavi Kale

AI & Data Analytics | Machine Learning | NLP | Generative AI | Python

💻 GitHub: kalevaishnavi04
📧 Email: kalevaishanvi833@gmail.com

⭐ If you found this project useful, consider giving the repository a star!