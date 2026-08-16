import os
import streamlit as st
from googleapiclient.discovery import build
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# YouTube Data API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Separate counters for top-level comments vs replies
comment_sentiment_counts = defaultdict(int)
reply_sentiment_counts = defaultdict(int)
comments_processed = 0
replies_processed = 0
all_comments_text = []
all_replies_text = []

# Load BERT model (only when needed, to avoid slow startup every time)
_bert_tokenizer = None
_bert_model = None

def get_bert_model():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        _bert_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return _bert_tokenizer, _bert_model

# Sentiment Analysis
def analyze_sentiment_textblob(comment):
    blob = TextBlob(comment)
    return blob.sentiment.polarity

def analyze_sentiment_bert(comment):
    tokenizer, model = get_bert_model()
    inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment_score = torch.softmax(logits, dim=1).tolist()[0]
    return sentiment_score

def categorize_sentiment_textblob(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def categorize_sentiment_bert(score):
    positive, neutral, negative = score[4], score[2], score[0]
    if positive > max(neutral, negative):
        return 'Positive'
    elif negative > max(positive, neutral):
        return 'Negative'
    else:
        return 'Neutral'

def analyze_comment(comment, analysis_method):
    sentiment_score = None
    sentiment_category = None

    if analysis_method == "TextBlob":
        sentiment_score = analyze_sentiment_textblob(comment)
        sentiment_category = categorize_sentiment_textblob(sentiment_score)
    elif analysis_method == "BERT":
        sentiment_score = analyze_sentiment_bert(comment)
        sentiment_category = categorize_sentiment_bert(sentiment_score)

    return sentiment_score, sentiment_category

def generate_pie_chart(sentiment_counts, chart_container, title):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    if sum(sizes) == 0:
        chart_container.write(f"No {title.lower()} data available.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_title(title)
    chart_container.pyplot(fig)
    plt.close(fig)

def generate_word_cloud(text_list, cloud_container, title):
    if not text_list:
        cloud_container.write(f"No {title.lower()} text available.")
        return

    text = ' '.join(text_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    cloud_container.pyplot(fig)
    plt.close(fig)

def fetch_replies_for_thread(parent_id):
    """Fetch all replies for a given top-level comment thread (beyond what's inlined)."""
    replies = []
    request = youtube.comments().list(
        parentId=parent_id,
        part='snippet',
        maxResults=100
    )
    while request is not None:
        try:
            response = request.execute()
            for item in response.get('items', []):
                replies.append(item['snippet']['textDisplay'])
            request = youtube.comments().list_next(request, response)
        except HttpError:
            break
    return replies

def fetch_video_comments_and_analyze(video_id, analysis_method, progress_callback=None):
    global comments_processed, replies_processed, all_comments_text, all_replies_text

    request = youtube.commentThreads().list(
        videoId=video_id,
        part='snippet',
        maxResults=100,
        textFormat='plainText'
    )

    while request is not None:
        try:
            response = request.execute()
            for item in response.get('items', []):
                # --- Top-level comment ---
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                all_comments_text.append(comment)
                comments_processed += 1

                _, sentiment_category = analyze_comment(comment, analysis_method)
                comment_sentiment_counts[sentiment_category] += 1

                # --- Replies for this thread ---
                total_reply_count = item['snippet'].get('totalReplyCount', 0)
                if total_reply_count > 0:
                    reply_texts = fetch_replies_for_thread(item['snippet']['topLevelComment']['id'])
                    for reply_text in reply_texts:
                        all_replies_text.append(reply_text)
                        replies_processed += 1
                        _, reply_sentiment = analyze_comment(reply_text, analysis_method)
                        reply_sentiment_counts[reply_sentiment] += 1

                if progress_callback:
                    progress_callback(comments_processed, replies_processed)

            request = youtube.commentThreads().list_next(request, response)

        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                st.error("Comments are disabled for this video.")
            else:
                st.error(f"Error fetching comments: {e}")
            break

def extract_video_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def start_streamlit_dashboard():
    st.title("YouTube Comment Sentiment Analysis (TextBlob + BERT)")
    st.caption("Comments and replies are tracked and shown separately.")

    video_url = st.text_input("Enter YouTube Video URL:", "")
    analysis_method = st.selectbox("Choose Sentiment Analysis Method", ["TextBlob", "BERT"])
    run_button = st.button("Analyze Comments")

    status_container = st.empty()

    if run_button and video_url:
        global comments_processed, replies_processed, all_comments_text, all_replies_text
        global comment_sentiment_counts, reply_sentiment_counts

        comments_processed = 0
        replies_processed = 0
        all_comments_text = []
        all_replies_text = []
        comment_sentiment_counts = defaultdict(int)
        reply_sentiment_counts = defaultdict(int)

        video_id = extract_video_id(video_url)
        if video_id:
            def update_status(c, r):
                status_container.write(f"Comments processed: {c} | Replies processed: {r}")

            with st.spinner(f"Fetching and analyzing comments + replies using {analysis_method}..."):
                fetch_video_comments_and_analyze(video_id, analysis_method, progress_callback=update_status)

            status_container.write(
                f"**Total Comments Processed: {comments_processed}**  |  **Total Replies Processed: {replies_processed}**"
            )

            st.divider()
            st.subheader("Top-Level Comments")
            col1, col2 = st.columns(2)
            generate_pie_chart(comment_sentiment_counts, col1, "Comment Sentiment")
            generate_word_cloud(all_comments_text, col2, "Comment Word Cloud")

            st.divider()
            st.subheader("Replies")
            col3, col4 = st.columns(2)
            generate_pie_chart(reply_sentiment_counts, col3, "Reply Sentiment")
            generate_word_cloud(all_replies_text, col4, "Reply Word Cloud")
        else:
            st.error("Invalid YouTube URL. Make sure it looks like https://www.youtube.com/watch?v=VIDEO_ID")

if __name__ == "__main__":
    start_streamlit_dashboard()