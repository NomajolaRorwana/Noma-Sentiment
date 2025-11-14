import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import re
from collections import defaultdict

# Configure the page with pink theme
st.set_page_config(
    page_title="Noma Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pink and white theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e91e63;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        color: #ad1457;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #fce4ec;
    }
    .stButton button {
        background-color: #e91e63;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #ad1457;
        color: white;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e91e63;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .positive-sentiment {
        color: #e91e63;
        font-weight: bold;
        font-size: 1.2em;
    }
    .negative-sentiment {
        color: #880e4f;
        font-weight: bold;
        font-size: 1.2em;
    }
    .neutral-sentiment {
        color: #ad1457;
        font-weight: bold;
        font-size: 1.2em;
    }
    .success-message {
        background-color: #fce4ec;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e91e63;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing feedback history
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'sentiment_counts' not in st.session_state:
    st.session_state.sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

# Simple sentiment analysis using rule-based approach
def analyze_sentiment(text):
    """Advanced rule-based sentiment analysis"""
    text_lower = text.lower()
   
    # Enhanced word lists with weights
    positive_words = {
        'excellent': 2.0, 'outstanding': 2.0, 'amazing': 2.0, 'perfect': 2.0,
        'great': 1.5, 'good': 1.0, 'awesome': 1.5, 'fantastic': 1.5,
        'wonderful': 1.5, 'brilliant': 1.5, 'superb': 1.5, 'love': 2.0,
        'happy': 1.5, 'pleased': 1.0, 'satisfied': 1.0, 'quick': 1.0,
        'fast': 1.0, 'helpful': 1.0, 'responsive': 1.0, 'professional': 1.0,
        'thanks': 1.0, 'thank you': 1.5, 'appreciate': 1.0, 'recommend': 1.5
    }
   
    negative_words = {
        'terrible': 2.0, 'awful': 2.0, 'horrible': 2.0, 'disgusting': 2.0,
        'bad': 1.0, 'poor': 1.0, 'worst': 2.0, 'hate': 2.0, 'useless': 1.5,
        'waste': 1.5, 'frustrating': 1.5, 'annoying': 1.0, 'disappointing': 1.5,
        'slow': 1.0, 'broken': 1.5, 'failed': 1.5, 'issue': 1.0, 'problem': 1.0,
        'complaint': 1.0, 'unacceptable': 1.5, 'pathetic': 2.0, 'rubbish': 1.5
    }
   
    # Intensifiers and negations
    intensifiers = {'very': 1.5, 'really': 1.3, 'extremely': 1.7, 'absolutely': 1.7}
    negations = {'not', "n't", 'no', 'never', 'without'}
   
    # Calculate scores
    positive_score = 0
    negative_score = 0
   
    words = text_lower.split()
    for i, word in enumerate(words):
        word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
       
        # Check for intensifiers
        intensifier = 1.0
        if i > 0 and words[i-1] in intensifiers:
            intensifier = intensifiers[words[i-1]]
       
        # Check for negations
        negation = 1.0
        if i > 0 and words[i-1] in negations:
            negation = -1.0
       
        # Score positive words
        if word in positive_words:
            score = positive_words[word] * intensifier * negation
            positive_score += score
       
        # Score negative words
        if word in negative_words:
            score = negative_words[word] * intensifier * negation
            negative_score += score
   
    # Calculate overall polarity
    total_score = positive_score - negative_score
   
    # Determine sentiment
    if total_score > 1.0:
        sentiment = 'positive'
        emoji = '😊'
        polarity = min(total_score / 10.0, 1.0)  # Normalize
    elif total_score < -1.0:
        sentiment = 'negative'
        emoji = '😠'
        polarity = max(total_score / 10.0, -1.0)  # Normalize
    else:
        sentiment = 'neutral'
        emoji = '😐'
        polarity = 0.0
   
    return {
        'polarity': polarity,
        'sentiment': sentiment,
        'emoji': emoji,
        'positive_score': positive_score,
        'negative_score': negative_score,
        'subjectivity': min((positive_score + negative_score) / 10.0, 1.0)
    }

def extract_keywords(text, sentiment):
    """Extract potential keywords based on sentiment"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
   
    positive_keywords = {'good', 'great', 'excellent', 'awesome', 'helpful', 'fast', 'quick',
                        'thanks', 'thank', 'amazing', 'love', 'perfect', 'fantastic', 'outstanding',
                        'satisfied', 'happy', 'pleased', 'responsive', 'professional', 'brilliant'}
   
    negative_keywords = {'bad', 'terrible', 'awful', 'slow', 'useless', 'waste', 'horrible',
                        'frustrating', 'hate', 'disappointing', 'poor', 'broken', 'failed',
                        'issue', 'problem', 'complaint', 'angry', 'annoying', 'unacceptable'}
   
    if sentiment == 'positive':
        found_keywords = [word for word in words if word in positive_keywords]
    elif sentiment == 'negative':
        found_keywords = [word for word in words if word in negative_keywords]
    else:
        found_keywords = [word for word in words if word in positive_keywords.union(negative_keywords)]
   
    # Return unique keywords, max 5
    return list(set(found_keywords))[:5]

# Main dashboard with pink theme
st.markdown('<h1 class="main-header">🌸 Noma Customer Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Analyze customer feedback sentiment in real-time with our beautiful pink-themed dashboard")

# Sidebar for text input with pink theme
with st.sidebar:
    st.markdown('<div style="background-color: #fce4ec; padding: 1rem; border-radius: 10px;">', unsafe_allow_html=True)
    st.header("📝 Enter Customer Feedback")
    feedback_text = st.text_area(
        "Paste customer feedback text here:",
        height=150,
        placeholder="Example: 'The service was excellent and very quick! The staff was very helpful.'"
    )
   
    analyze_btn = st.button("Analyze Sentiment", type="primary")
   
    # Quick examples
    st.markdown("---")
    st.markdown("### Try these examples:")
    examples = {
        "Positive": "The service was excellent and very quick! The staff was very helpful and professional.",
        "Negative": "I'm very disappointed with the slow response and poor quality service. Absolutely terrible experience.",
        "Neutral": "The product is okay, but it could be better. Average experience overall."
    }
   
    for sentiment, example in examples.items():
        if st.button(f"{sentiment}: {example[:50]}...", key=example):
            feedback_text = example
   
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("""
    1. Enter customer feedback text
    2. Click 'Analyze Sentiment'
    3. View real-time sentiment analysis
    4. Track trends over time
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if analyze_btn and feedback_text:
    # Analyze the sentiment
    with st.spinner('Analyzing sentiment...'):
        result = analyze_sentiment(feedback_text)
        keywords = extract_keywords(feedback_text, result['sentiment'])
   
    # Store in history
    feedback_entry = {
        'text': feedback_text[:100] + "..." if len(feedback_text) > 100 else feedback_text,
        'full_text': feedback_text,
        'sentiment': result['sentiment'],
        'polarity': result['polarity'],
        'emoji': result['emoji'],
        'timestamp': datetime.datetime.now(),
        'keywords': keywords,
        'positive_score': result['positive_score'],
        'negative_score': result['negative_score']
    }
    st.session_state.feedback_history.append(feedback_entry)
   
    # Update sentiment counts
    st.session_state.sentiment_counts[result['sentiment']] += 1
   
    # Show immediate result with custom styling
    sentiment_color_class = f"{result['sentiment']}-sentiment"
    st.markdown(
        f"""
        <div class="success-message">
            <h3>✅ Sentiment Analysis Complete</h3>
            <p>Result: {result['emoji']} <span class='{sentiment_color_class}'>{result['sentiment'].upper()}</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    # Show sentiment details in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Polarity Score", f"{result['polarity']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Subjectivity", f"{result['subjectivity']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Positive Score", f"{result['positive_score']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Negative Score", f"{result['negative_score']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

# Create layout with columns for metrics
st.markdown("---")
st.markdown('<h2 class="sub-header">📈 Overall Metrics</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_feedback = len(st.session_state.feedback_history)
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Total Feedback Analyzed",
        value=total_feedback
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    positive_rate = (st.session_state.sentiment_counts['positive'] / max(total_feedback, 1)) * 100
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Positive Feedback Rate",
        value=f"{positive_rate:.1f}%"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    # Calculate NPS-like score (Positive % - Negative %)
    nps_score = ((st.session_state.sentiment_counts['positive'] - st.session_state.sentiment_counts['negative']) / max(total_feedback, 1)) * 100
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Net Sentiment Score",
        value=f"{nps_score:.1f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    negative_rate = (st.session_state.sentiment_counts['negative'] / max(total_feedback, 1)) * 100
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Negative Feedback Rate",
        value=f"{negative_rate:.1f}%"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Sentiment Breakdown Section
st.markdown("---")
st.markdown('<h2 class="sub-header">📊 Sentiment Analysis Results</h2>', unsafe_allow_html=True)

if st.session_state.feedback_history:
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
   
    with chart_col1:
        # Sentiment distribution pie chart with pink colors
        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [
                st.session_state.sentiment_counts['positive'],
                st.session_state.sentiment_counts['neutral'],
                st.session_state.sentiment_counts['negative']
            ]
        })
       
        # Pink color scheme for the pie chart
        pink_colors = ['#e91e63', '#f48fb1', '#880e4f']  # Pink shades
       
        fig_pie = px.pie(
            sentiment_df,
            values='Count',
            names='Sentiment',
            title='Customer Sentiment Distribution',
            color='Sentiment',
            color_discrete_sequence=pink_colors
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
   
    with chart_col2:
        # Sentiment trend over time with pink theme
        history_df = pd.DataFrame(st.session_state.feedback_history)
        history_df['sequence'] = range(1, len(history_df) + 1)
       
        fig_trend = px.line(
            history_df,
            x='sequence',
            y='polarity',
            title='Sentiment Trend (Polarity Over Time)',
            labels={'sequence': 'Feedback Sequence', 'polarity': 'Sentiment Polarity'},
            color_discrete_sequence=['#e91e63']
        )
        fig_trend.add_hline(y=0, line_dash="dash", line_color="#ad1457", annotation_text="Neutral")
        fig_trend.update_traces(mode='markers+lines', marker=dict(size=8, color='#e91e63'))
        fig_trend.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Keyword Analysis
    st.markdown('<h2 class="sub-header">🔍 Feedback Keywords Analysis</h2>', unsafe_allow_html=True)
   
    # Collect all keywords
    all_keywords = []
    for entry in st.session_state.feedback_history:
        all_keywords.extend(entry['keywords'])
   
    if all_keywords:
        keyword_counts = pd.Series(all_keywords).value_counts()
        keyword_df = pd.DataFrame({
            'Keyword': keyword_counts.index,
            'Frequency': keyword_counts.values
        })
       
        fig_keywords = px.bar(
            keyword_df.head(10),
            x='Frequency',
            y='Keyword',
            orientation='h',
            title='Most Frequent Feedback Keywords',
            color='Frequency',
            color_continuous_scale='pinkyl'
        )
        fig_keywords.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_keywords, use_container_width=True)
    else:
        st.info("Keywords will appear as more feedback is analyzed")

    # Recent Feedback Table
    st.markdown('<h2 class="sub-header">📋 Recent Feedback Analysis</h2>', unsafe_allow_html=True)
   
    display_df = pd.DataFrame(st.session_state.feedback_history[-10:])
    if not display_df.empty:
        display_df = display_df[['emoji', 'sentiment', 'text', 'timestamp', 'polarity']]
        display_df.columns = ['', 'Sentiment', 'Feedback Preview', 'Time', 'Polarity']
        display_df['Polarity'] = display_df['Polarity'].round(3)
       
        # Style the dataframe
        st.dataframe(display_df, use_container_width=True)
       
        # Export option
        if st.button("Export Analysis Data"):
            export_df = pd.DataFrame(st.session_state.feedback_history)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"noma_sentiment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

else:
    # Show placeholder when no feedback has been analyzed
    st.info("👆 Enter customer feedback in the sidebar to start analysis")
   
    # Sample dashboard layout
    sample_col1, sample_col2 = st.columns(2)
   
    with sample_col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("Customer Sentiment Distribution")
        st.markdown("Chart will appear here after analyzing feedback")
        st.markdown('</div>', unsafe_allow_html=True)
   
    with sample_col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("Sentiment Trend Over Time")
        st.markdown("Trend analysis will appear here after analyzing multiple feedback entries")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with pink theme
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #e91e63; background-color: #fce4ec; padding: 1rem; border-radius: 10px;'>"
    "<strong>🌸 Noma Customer Sentiment Analysis Dashboard • Real-time Text Analysis • Powered by Advanced Sentiment Analysis</strong>"
    "</div>",
    unsafe_allow_html=True
)
