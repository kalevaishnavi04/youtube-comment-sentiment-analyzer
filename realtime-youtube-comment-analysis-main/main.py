import os
import re
import string
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from textblob import TextBlob
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import emoji as emoji_lib
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ----------------------------------------------------------------------------
# Lazy-loaded models (loaded once, cached by Streamlit)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BERT sentiment model...")
def load_bert_sentiment_model():
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

@st.cache_resource(show_spinner="Loading toxicity detection model...")
def load_toxic_model():
    tokenizer = AutoTokenizer.from_pretrained('unitary/toxic-bert')
    model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')
    return tokenizer, model

# ----------------------------------------------------------------------------
# 1. Emoji-aware sentiment helper
# ----------------------------------------------------------------------------
POSITIVE_EMOJIS = {"😀", "😂", "😍", "❤️", "👍", "🔥", "😊", "😁", "🥰", "👏", "💯", "😄"}
NEGATIVE_EMOJIS = {"😡", "👎", "😢", "😭", "🙄", "💔", "😞", "😠", "😒", "🤮", "😤"}

def emoji_sentiment_adjustment(text):
    """Returns a small nudge (-0.3 to +0.3) based on emoji content."""
    if not EMOJI_AVAILABLE:
        return 0.0
    found = [c for c in text if c in emoji_lib.EMOJI_DATA]
    if not found:
        return 0.0
    pos = sum(1 for c in found if c in POSITIVE_EMOJIS)
    neg = sum(1 for c in found if c in NEGATIVE_EMOJIS)
    if pos == neg:
        return 0.0
    return 0.15 if pos > neg else -0.15

def strip_emojis(text):
    if not EMOJI_AVAILABLE:
        return text
    return emoji_lib.replace_emoji(text, replace='')

# ----------------------------------------------------------------------------
# Sentiment analysis (TextBlob / BERT), emoji-adjusted
# ----------------------------------------------------------------------------
def analyze_sentiment_textblob(comment):
    clean_text = strip_emojis(comment)
    polarity = TextBlob(clean_text).sentiment.polarity
    polarity += emoji_sentiment_adjustment(comment)
    return max(-1.0, min(1.0, polarity))

def analyze_sentiment_bert(comment):
    tokenizer, model = load_bert_sentiment_model()
    clean_text = strip_emojis(comment)
    inputs = tokenizer(clean_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    nudge = emoji_sentiment_adjustment(comment)
    if nudge > 0:
        scores[4] += nudge  # boost "very positive" bucket
    elif nudge < 0:
        scores[0] += abs(nudge)  # boost "very negative" bucket
    return scores

def categorize_sentiment_textblob(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    return 'Neutral'

def categorize_sentiment_bert(score):
    positive, neutral, negative = score[4], score[2], score[0]
    if positive > max(neutral, negative):
        return 'Positive'
    elif negative > max(positive, neutral):
        return 'Negative'
    return 'Neutral'

def analyze_comment(comment, analysis_method):
    if analysis_method == "TextBlob":
        score = analyze_sentiment_textblob(comment)
        return score, categorize_sentiment_textblob(score)
    else:
        score = analyze_sentiment_bert(comment)
        return score, categorize_sentiment_bert(score)

# ----------------------------------------------------------------------------
# 3. Toxic / hate comment detection
# ----------------------------------------------------------------------------
def is_toxic(comment, threshold=0.5):
    tokenizer, model = load_toxic_model()
    inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits).tolist()[0]
    # unitary/toxic-bert labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
    max_score = max(scores)
    return max_score >= threshold, max_score

# ----------------------------------------------------------------------------
# 2. Topic / keyword extraction (TF-IDF based, no heavy LDA dependency)
# ----------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    "the", "is", "and", "to", "a", "of", "in", "it", "this", "that", "i", "you",
    "for", "on", "was", "with", "as", "but", "are", "be", "have", "has", "not",
    "he", "she", "they", "we", "his", "her", "its", "so", "just", "very", "or",
    "video", "comment", "https", "www", "com"
}

def extract_top_keywords(texts, top_n=15):
    if not texts or len(texts) < 3:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=list(CUSTOM_STOPWORDS),
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = tfidf_matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    except ValueError:
        return []

# ----------------------------------------------------------------------------
# 7. Language detection
# ----------------------------------------------------------------------------
def detect_language(text):
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        clean = strip_emojis(text).strip()
        if len(clean) < 3:
            return "unknown"
        return detect(clean)
    except Exception:
        return "unknown"

# ----------------------------------------------------------------------------
# 8. Fake / bot comment detection (heuristic: near-duplicate & generic praise)
# ----------------------------------------------------------------------------
GENERIC_PRAISE_PATTERNS = [
    r'^(nice|good|great|awesome|amazing|superb|wow|best)\s*(video|content)?[!.]*$',
    r'^first[!.]*$',
    r'^(love|loved) (it|this)[!.]*$',
]

def normalize_for_dedup(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def detect_bot_like_comments(comments):
    """Returns (duplicate_count, generic_praise_count, flagged_examples)."""
    normalized = [normalize_for_dedup(c) for c in comments]
    counts = Counter(normalized)
    duplicate_count = sum(c for c in counts.values() if c > 1)

    generic_count = 0
    flagged_examples = []
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in GENERIC_PRAISE_PATTERNS]
    for original, norm in zip(comments, normalized):
        if any(p.match(norm) for p in compiled_patterns):
            generic_count += 1
            if len(flagged_examples) < 10:
                flagged_examples.append(original)

    return duplicate_count, generic_count, flagged_examples

# ----------------------------------------------------------------------------
# 6. Lightweight extractive summary (no external LLM API key required)
# ----------------------------------------------------------------------------
def generate_extractive_summary(texts, top_n=5):
    """Picks the most 'representative' comments using TF-IDF centrality —
    a simple stand-in for a true LLM summary. For a real AI-generated
    summary, plug in an LLM API call here using the collected `texts`."""
    if len(texts) < top_n:
        return texts
    try:
        vectorizer = TfidfVectorizer(stop_words=list(CUSTOM_STOPWORDS), max_features=300)
        tfidf_matrix = vectorizer.fit_transform(texts)
        centroid = np.asarray(tfidf_matrix.mean(axis=0))
        similarities = cosine_similarity(tfidf_matrix, centroid).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        return [texts[i] for i in top_indices]
    except ValueError:
        return texts[:top_n]

# ----------------------------------------------------------------------------
# Visualizations
# ----------------------------------------------------------------------------
def generate_pie_chart(sentiment_counts, container, title):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    if sum(sizes) == 0:
        container.write(f"No {title.lower()} data available.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_title(title)
    container.pyplot(fig)
    plt.close(fig)

def generate_word_cloud(text_list, container, title):
    if not text_list:
        container.write(f"No {title.lower()} text available.")
        return
    text = ' '.join(text_list)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    container.pyplot(fig)
    plt.close(fig)

def generate_trend_chart(timeline_data, container):
    if not timeline_data:
        container.write("No timeline data available.")
        return
    df = pd.DataFrame(timeline_data)
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in daily.columns:
            daily[col] = 0
    daily = daily[['Positive', 'Neutral', 'Negative']].sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily['Positive'], label='Positive', color='green', marker='o')
    ax.plot(daily.index, daily['Neutral'], label='Neutral', color='gray', marker='o')
    ax.plot(daily.index, daily['Negative'], label='Negative', color='red', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Sentiment Trend Over Time')
    ax.legend()
    fig.autofmt_xdate()
    container.pyplot(fig)
    plt.close(fig)

def generate_keyword_chart(keywords, container, title):
    if not keywords:
        container.write("Not enough data to extract keywords.")
        return
    terms = [k[0] for k in keywords][::-1]
    scores = [k[1] for k in keywords][::-1]
    fig, ax = plt.subplots(figsize=(6, max(4, len(terms) * 0.35)))
    ax.barh(terms, scores, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel('TF-IDF Score')
    container.pyplot(fig)
    plt.close(fig)

def generate_language_chart(lang_counts, container):
    if not lang_counts:
        container.write("No language data available.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    langs = list(lang_counts.keys())
    counts = list(lang_counts.values())
    ax.bar(langs, counts, color='coral')
    ax.set_title('Comments by Detected Language')
    ax.set_ylabel('Count')
    container.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# YouTube fetching
# ----------------------------------------------------------------------------
def fetch_replies_for_thread(parent_id):
    replies = []
    request = youtube.comments().list(parentId=parent_id, part='snippet', maxResults=100)
    while request is not None:
        try:
            response = request.execute()
            for item in response.get('items', []):
                replies.append(item['snippet']['textDisplay'])
            request = youtube.comments().list_next(request, response)
        except HttpError:
            break
    return replies

def fetch_and_analyze(video_id, analysis_method, check_toxicity, progress_callback=None):
    """Fetches comments+replies and runs all analyses. Returns a results dict."""
    results = {
        'comments_processed': 0,
        'replies_processed': 0,
        'all_comments_text': [],
        'all_replies_text': [],
        'comment_sentiment_counts': defaultdict(int),
        'reply_sentiment_counts': defaultdict(int),
        'sentiment_timeline': [],
        'language_counts': defaultdict(int),
        'toxic_comments': [],
    }

    request = youtube.commentThreads().list(
        videoId=video_id, part='snippet', maxResults=100, textFormat='plainText'
    )

    while request is not None:
        try:
            response = request.execute()
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                published_at = item['snippet']['topLevelComment']['snippet'].get('publishedAt')

                results['all_comments_text'].append(comment)
                results['comments_processed'] += 1

                _, sentiment_category = analyze_comment(comment, analysis_method)
                results['comment_sentiment_counts'][sentiment_category] += 1

                if published_at:
                    results['sentiment_timeline'].append({'date': published_at[:10], 'sentiment': sentiment_category})

                lang = detect_language(comment)
                results['language_counts'][lang] += 1

                if check_toxicity:
                    toxic, score = is_toxic(comment)
                    if toxic:
                        results['toxic_comments'].append((comment, round(score, 2)))

                total_reply_count = item['snippet'].get('totalReplyCount', 0)
                if total_reply_count > 0:
                    reply_texts = fetch_replies_for_thread(item['snippet']['topLevelComment']['id'])
                    for reply_text in reply_texts:
                        results['all_replies_text'].append(reply_text)
                        results['replies_processed'] += 1
                        _, reply_sentiment = analyze_comment(reply_text, analysis_method)
                        results['reply_sentiment_counts'][reply_sentiment] += 1

                if progress_callback:
                    progress_callback(results['comments_processed'], results['replies_processed'])

            request = youtube.commentThreads().list_next(request, response)

        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                st.error("Comments are disabled for this video.")
            else:
                st.error(f"Error fetching comments: {e}")
            break

    return results

def extract_video_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def get_video_title(video_id):
    try:
        response = youtube.videos().list(part='snippet', id=video_id).execute()
        items = response.get('items', [])
        if items:
            return items[0]['snippet']['title']
    except Exception:
        pass
    return video_id

# ----------------------------------------------------------------------------
# Dashboard for a single video's full analysis
# ----------------------------------------------------------------------------
def render_single_video_analysis(results, analysis_method):
    st.write(
        f"**Total Comments: {results['comments_processed']}**  |  "
        f"**Total Replies: {results['replies_processed']}**"
    )

    st.divider()
    st.subheader("💬 Top-Level Comments")
    col1, col2 = st.columns(2)
    generate_pie_chart(results['comment_sentiment_counts'], col1, "Comment Sentiment")
    generate_word_cloud(results['all_comments_text'], col2, "Comment Word Cloud")

    st.divider()
    st.subheader("↩️ Replies")
    col3, col4 = st.columns(2)
    generate_pie_chart(results['reply_sentiment_counts'], col3, "Reply Sentiment")
    generate_word_cloud(results['all_replies_text'], col4, "Reply Word Cloud")

    st.divider()
    st.subheader("📈 Sentiment Trend Over Time")
    st.caption("How comment sentiment changed by publish date.")
    trend_container = st.empty()
    generate_trend_chart(results['sentiment_timeline'], trend_container)

    st.divider()
    st.subheader("🔑 Top Topics / Keywords")
    st.caption("What people are actually talking about (TF-IDF based, not just word frequency).")
    keywords = extract_top_keywords(results['all_comments_text'])
    kw_container = st.empty()
    generate_keyword_chart(keywords, kw_container, "Top Keywords in Comments")

    st.divider()
    st.subheader("🌍 Language Breakdown")
    lang_container = st.empty()
    generate_language_chart(results['language_counts'], lang_container)
    if not LANGDETECT_AVAILABLE:
        st.warning("Install `langdetect` (`pip install langdetect`) to enable this feature.")

    st.divider()
    st.subheader("🚫 Toxic / Hate Comment Flags")
    if results['toxic_comments']:
        st.error(f"{len(results['toxic_comments'])} potentially toxic comments detected.")
        with st.expander("View flagged comments"):
            for comment, score in results['toxic_comments']:
                st.write(f"⚠️ ({score}) {comment}")
    else:
        st.success("No toxic comments detected (or toxicity check was skipped).")

    st.divider()
    st.subheader("🤖 Fake / Bot-like Comment Signals")
    dup_count, generic_count, examples = detect_bot_like_comments(results['all_comments_text'])
    c1, c2 = st.columns(2)
    c1.metric("Duplicate / repeated comments", dup_count)
    c2.metric("Generic praise comments", generic_count)
    if examples:
        with st.expander("Sample flagged comments"):
            for ex in examples:
                st.write(f"• {ex}")

    st.divider()
    st.subheader("📝 Quick Summary (extractive)")
    st.caption(
        "This picks the most representative real comments (TF-IDF based) as a lightweight stand-in "
        "for an AI-generated summary. Plug in an LLM API call in `generate_extractive_summary()` "
        "for a true AI-written summary."
    )
    summary_comments = generate_extractive_summary(results['all_comments_text'])
    for c in summary_comments:
        st.write(f"• {c}")

# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def start_streamlit_dashboard():
    st.set_page_config(page_title="YouTube Comment Sentiment Analyzer", layout="wide")
    st.title("🎥 YouTube Comment Sentiment Analyzer")
    st.caption(
        "Sentiment, topics, toxicity, trends, language, and bot detection — "
        "powered by TextBlob, BERT, and TF-IDF."
    )

    tab1, tab2 = st.tabs(["🔍 Single Video Analysis", "⚖️ Compare Multiple Videos"])

    analysis_method = st.sidebar.selectbox("Sentiment Method", ["TextBlob", "BERT"])
    check_toxicity = st.sidebar.checkbox(
        "Enable toxicity detection (slower, downloads a model on first use)", value=False
    )

    # --- Single video tab ---
    with tab1:
        video_url = st.text_input("Enter YouTube Video URL:", key="single_url")
        run_button = st.button("Analyze Comments", key="single_run")

        if run_button and video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                status = st.empty()

                def update_status(c, r):
                    status.write(f"Comments processed: {c} | Replies processed: {r}")

                with st.spinner(f"Fetching and analyzing comments using {analysis_method}..."):
                    results = fetch_and_analyze(video_id, analysis_method, check_toxicity, update_status)

                render_single_video_analysis(results, analysis_method)
            else:
                st.error("Invalid YouTube URL. Make sure it looks like https://www.youtube.com/watch?v=VIDEO_ID")

    # --- Multi-video comparison tab ---
    with tab2:
        st.write("Paste 2-3 YouTube video URLs to compare their sentiment side-by-side.")
        urls_text = st.text_area(
            "One URL per line:", height=100, key="multi_urls",
            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
        )
        compare_button = st.button("Compare Videos", key="compare_run")

        if compare_button and urls_text.strip():
            urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
            comparison_data = []

            for url in urls:
                vid = extract_video_id(url)
                if not vid:
                    st.warning(f"Skipping invalid URL: {url}")
                    continue
                title = get_video_title(vid)
                with st.spinner(f"Analyzing: {title}..."):
                    res = fetch_and_analyze(vid, analysis_method, check_toxicity=False)
                total = res['comments_processed']
                pos = res['comment_sentiment_counts'].get('Positive', 0)
                neu = res['comment_sentiment_counts'].get('Neutral', 0)
                neg = res['comment_sentiment_counts'].get('Negative', 0)
                comparison_data.append({
                    'Video': title[:40],
                    'Total Comments': total,
                    'Positive %': round(100 * pos / total, 1) if total else 0,
                    'Neutral %': round(100 * neu / total, 1) if total else 0,
                    'Negative %': round(100 * neg / total, 1) if total else 0,
                })

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                x = range(len(df))
                width = 0.25
                ax.bar([i - width for i in x], df['Positive %'], width, label='Positive', color='green')
                ax.bar(x, df['Neutral %'], width, label='Neutral', color='gray')
                ax.bar([i + width for i in x], df['Negative %'], width, label='Negative', color='red')
                ax.set_xticks(list(x))
                ax.set_xticklabels(df['Video'], rotation=20, ha='right')
                ax.set_ylabel('% of Comments')
                ax.set_title('Sentiment Comparison Across Videos')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    start_streamlit_dashboard()