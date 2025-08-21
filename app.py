import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time
import re

# Page configuration
st.set_page_config(
    page_title=" AI Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide"
)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# Simple rule-based sentiment analysis
def get_simple_sentiment(text):
    """Simple rule-based sentiment analysis"""
    try:
        text_lower = text.lower()
        
        # Define word lists
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'perfect', 'love', 'like', 'enjoy', 'happy',
            'pleased', 'satisfied', 'delighted', 'thrilled', 'excited', 'fantastic',
            'superb', 'outstanding', 'magnificent', 'marvelous', 'incredible',
            'best', 'better', 'nice', 'beautiful', 'sweet', 'cool', 'fun'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'angry', 'sad', 'upset', 'disappointed', 'frustrated', 'annoyed',
            'worried', 'concerned', 'stressed', 'depressed', 'miserable', 'unhappy',
            'worst', 'worse', 'ugly', 'stupid', 'dumb', 'boring', 'annoying',
            'pathetic', 'useless', 'worthless', 'disappointing', 'devastating'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Simple scoring
        total_words = len(text.split())
        positive_ratio = positive_count / max(total_words, 1)
        negative_ratio = negative_count / max(total_words, 1)
        
        if positive_count > negative_count:
            sentiment = "Positive"
            confidence = positive_ratio
        elif negative_count > positive_count:
            sentiment = "Negative"  
            confidence = negative_ratio
        else:
            sentiment = "Neutral"
            confidence = 0.1
            
        return {
            'sentiment': sentiment,
            'confidence': min(confidence * 5, 1.0),  # Scale confidence
            'raw_score': positive_count - negative_count,
            'details': f"Positive words: {positive_count}, Negative words: {negative_count}"
        }
    except Exception as e:
        return {'sentiment': 'Error', 'confidence': 0, 'raw_score': 0, 'details': str(e)}

def get_pattern_sentiment(text):
    """Pattern-based sentiment analysis using regex"""
    try:
        # Positive patterns
        positive_patterns = [
            r'\b(love|adore|enjoy)\b',
            r'\b(great|excellent|amazing|wonderful|fantastic|awesome|brilliant|perfect)\b',
            r'\b(good|nice|beautiful|sweet|cool|fun)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r':\)|:-\)|:D|😊|😃|😄|👍'  # Positive emoticons/emojis
        ]
        
        # Negative patterns  
        negative_patterns = [
            r'\b(hate|despise|detest)\b',
            r'\b(terrible|awful|horrible|disgusting|pathetic|useless)\b',
            r'\b(bad|worse|worst|ugly|stupid|boring)\b',
            r':\(|:-\(|😞|😠|👎'  # Negative emoticons/emojis
        ]
        
        positive_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in positive_patterns)
        negative_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in negative_patterns)
        
        if positive_score > negative_score:
            sentiment = "Positive"
            confidence = min(positive_score / 10, 1.0)
        elif negative_score > positive_score:
            sentiment = "Negative"
            confidence = min(negative_score / 10, 1.0)
        else:
            sentiment = "Neutral"
            confidence = 0.1
            
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_score': positive_score - negative_score,
            'details': f"Positive patterns: {positive_score}, Negative patterns: {negative_score}"
        }
    except Exception as e:
        return {'sentiment': 'Error', 'confidence': 0, 'raw_score': 0, 'details': str(e)}

def get_huggingface_sentiment(text):
    try:
        # This is a simplified demo - in practice you'd need proper authentication
        API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Mock response for demo (replace with actual API call when you have auth)
        # Uncomment below for real API call:
        # headers = {"Authorization": f"Bearer {your_hf_token}"}
        # response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
        
        # For now, return a mock analysis based on simple heuristics
        text_lower = text.lower()
        if any(word in text_lower for word in ['good', 'great', 'love', 'excellent', 'amazing']):
            return {
                'sentiment': 'Positive',
                'confidence': 0.85,
                'raw_score': 0.85,
                'details': 'Demo mode - detected positive keywords'
            }
        elif any(word in text_lower for word in ['bad', 'hate', 'terrible', 'awful', 'horrible']):
            return {
                'sentiment': 'Negative', 
                'confidence': 0.82,
                'raw_score': -0.82,
                'details': 'Demo mode - detected negative keywords'
            }
        else:
            return {
                'sentiment': 'Neutral',
                'confidence': 0.65,
                'raw_score': 0.0,
                'details': 'Demo mode - neutral content'
            }
            
    except Exception as e:
        return {'sentiment': 'API Error', 'confidence': 0, 'raw_score': 0, 'details': str(e)}

def create_comparison_chart(results):
    """Create a comparison chart of sentiment results"""
    methods = list(results.keys())
    sentiments = [results[method]['sentiment'] for method in methods]
    confidences = [results[method]['confidence'] for method in methods]
    
    # Create color map
    color_map = {
        'Positive': '#2E8B57', 
        'Negative': '#DC143C', 
        'Neutral': '#4682B4', 
        'Error': '#808080', 
        'API Error': '#FFA500'
    }
    colors = [color_map.get(s, '#808080') for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(x=methods, y=confidences, 
               marker_color=colors,
               text=sentiments,
               textposition='auto',
               hovertemplate='<b>%{x}</b><br>Sentiment: %{text}<br>Confidence: %{y:.3f}<extra></extra>')
    ])
    
    fig.update_layout(
        title='Sentiment Analysis Comparison',
        xaxis_title='Analysis Method',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def main():
    st.title("🎭 Sentiment Analysis Dashboard")
    st.markdown("Analyze text sentiment using multiple methods and compare their results")
    
    # Sidebar for settings
    st.sidebar.header("Analysis Methods")
    
    use_simple = st.sidebar.checkbox("Simple Rule-Based", value=True, help="Uses positive/negative word lists")
    use_pattern = st.sidebar.checkbox("Pattern-Based", value=True, help="Uses regex patterns and emoticons")
    use_huggingface = st.sidebar.checkbox("Hugging Face (Demo)", value=False, help="Demo version - requires API key for production")
    use_manual = st.sidebar.checkbox("Manual Assessment", value=False, help="Add your own sentiment rating")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Note**: This version works without external dependencies that require database access.")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Analysis")
        
        # Text input
        user_text = st.text_area(
            "Enter text to analyze:", 
            placeholder="Type or paste your text here...\n\nTry examples like:\n• 'I love this product! It's amazing!'\n• 'This is terrible and I hate it.'\n• 'The weather is okay today.'", 
            height=150
        )
        
        # Sample texts for testing
        st.markdown("**Quick Test Examples (click to load):**")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("😊 Positive Example"):
                user_text = "I absolutely love this product! It's fantastic and works perfectly. Highly recommended!"
                st.success("Positive example loaded!")
        with col_b:
            if st.button("😞 Negative Example"):
                user_text = "This is terrible! I hate it so much. Worst purchase ever. Complete waste of money."
                st.error("Negative example loaded!")
        with col_c:
            if st.button("😐 Neutral Example"):
                user_text = "The product arrived on time. It has basic features and standard quality. Nothing special."
                st.info("Neutral example loaded!")
        
        # Show current text length
        if user_text.strip():
            st.write(f"📝 Text length: {len(user_text)} characters")
        else:
            st.warning("👆 Please enter some text above or click an example button")
        
        # Manual assessment if enabled
        manual_sentiment = None
        if use_manual and user_text.strip():
            st.subheader("Manual Assessment")
            manual_sentiment = st.radio(
                "What sentiment do you think this text expresses?",
                ["Positive", "Negative", "Neutral"],
                horizontal=True
            )
        
        # Analysis button
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", disabled=not user_text.strip())
        
        if analyze_clicked and user_text.strip():
            # Show loading
            with st.spinner("Analyzing sentiment..."):
                results = {}
                
                # Run selected analyses
                if use_simple:
                    results['Simple Rule-Based'] = get_simple_sentiment(user_text)
                
                if use_pattern:
                    results['Pattern-Based'] = get_pattern_sentiment(user_text)
                
                if use_huggingface:
                    results['Hugging Face (Demo)'] = get_huggingface_sentiment(user_text)
                
                if use_manual and manual_sentiment:
                    results['Manual Assessment'] = {
                        'sentiment': manual_sentiment,
                        'confidence': 1.0,
                        'raw_score': 1.0 if manual_sentiment == 'Positive' else (-1.0 if manual_sentiment == 'Negative' else 0.0),
                        'details': 'Human assessment'
                    }
            
            # Store results
            result_entry = {
                'timestamp': datetime.now(),
                'text': user_text[:100] + "..." if len(user_text) > 100 else user_text,
                'results': results
            }
            st.session_state.results_history.append(result_entry)
            
            # Display results
            st.header("📊 Results")
            
            # Create metrics columns
            if results:
                cols = st.columns(len(results))
                for i, (method, result) in enumerate(results.items()):
                    with cols[i]:
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        # Choose emoji based on sentiment
                        emoji_map = {
                            'Positive': '😊',
                            'Negative': '😞', 
                            'Neutral': '😐',
                            'Error': '❌',
                            'API Error': '⚠️'
                        }
                        
                        emoji = emoji_map.get(sentiment, '❓')
                        
                        # Color coding
                        color_map = {
                            'Positive': 'normal',
                            'Negative': 'inverse', 
                            'Neutral': 'off',
                            'Error': 'off',
                            'API Error': 'off'
                        }
                        
                        st.metric(
                            label=f"{emoji} {method}",
                            value=sentiment,
                            delta=f"{confidence:.1%} confidence"
                        )
                
                # Comparison chart
                if len(results) > 1:
                    st.plotly_chart(create_comparison_chart(results), use_container_width=True)
                
                # Detailed results
                with st.expander("🔍 Detailed Results"):
                    for method, result in results.items():
                        st.markdown(f"**{method}:**")
                        col_detail1, col_detail2, col_detail3 = st.columns(3)
                        with col_detail1:
                            st.write(f"Sentiment: {result['sentiment']}")
                        with col_detail2:
                            st.write(f"Confidence: {result['confidence']:.3f}")
                        with col_detail3:
                            st.write(f"Raw Score: {result['raw_score']:.3f}")
                        
                        if 'details' in result:
                            st.write(f"Details: {result['details']}")
                        st.markdown("---")
    
    with col2:
        st.header("📈 Analysis History")
        
        if st.session_state.results_history:
            # Show recent analyses
            st.write(f"**Recent Analyses ({len(st.session_state.results_history)} total)**")
            
            for i, entry in enumerate(reversed(st.session_state.results_history[-5:])):
                with st.expander(f"#{len(st.session_state.results_history)-i} - {entry['timestamp'].strftime('%H:%M:%S')}"):
                    st.write("**Text:**")
                    st.write(f"_{entry['text']}_")
                    st.write("**Results:**")
                    for method, result in entry['results'].items():
                        emoji = '😊' if result['sentiment'] == 'Positive' else '😞' if result['sentiment'] == 'Negative' else '😐'
                        st.write(f"{emoji} **{method}**: {result['sentiment']} ({result['confidence']:.2f})")
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.results_history = []
                st.rerun()
        else:
            st.info("No analyses yet. Enter some text and click 'Analyze Sentiment' to get started!")
    
    # Statistics section
    if len(st.session_state.results_history) >= 2:
        st.header("📈 Statistics & Insights")
        
        # Collect all results for statistics
        all_results = []
        for entry in st.session_state.results_history:
            for method, result in entry['results'].items():
                all_results.append({
                    'Method': method,
                    'Sentiment': result['sentiment'],
                    'Confidence': result['confidence'],
                    'Timestamp': entry['timestamp']
                })
        
        if all_results:
            df_stats = pd.DataFrame(all_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = df_stats['Sentiment'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, 
                               names=sentiment_counts.index,
                               title='Overall Sentiment Distribution',
                               color_discrete_map={
                                   'Positive': '#2E8B57',
                                   'Negative': '#DC143C', 
                                   'Neutral': '#4682B4'
                               })
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Method comparison
                method_stats = df_stats.groupby('Method').agg({
                    'Confidence': 'mean',
                    'Sentiment': 'count'
                }).round(3)
                method_stats.columns = ['Avg Confidence', 'Total Analyses']
                st.write("**Method Performance:**")
                st.dataframe(method_stats, use_container_width=True)
            
            # Agreement analysis
            if len(df_stats['Method'].unique()) > 1:
                st.subheader("Method Agreement Analysis")
                
                # Calculate agreement between methods
                agreements = []
                methods = df_stats['Method'].unique()
                
                for entry in st.session_state.results_history:
                    if len(entry['results']) > 1:
                        sentiments = [result['sentiment'] for result in entry['results'].values()]
                        if len(set(sentiments)) == 1:  # All methods agree
                            agreements.append("Full Agreement")
                        elif len(set(sentiments)) == len(sentiments):  # All methods disagree
                            agreements.append("No Agreement")
                        else:
                            agreements.append("Partial Agreement")
                
                if agreements:
                    agreement_counts = pd.Series(agreements).value_counts()
                    fig_agreement = px.bar(x=agreement_counts.index, y=agreement_counts.values,
                                         title='Method Agreement Levels')
                    st.plotly_chart(fig_agreement, use_container_width=True)
    
    # Instructions and tips
    with st.expander("ℹ️ How to Use & Tips"):
        st.markdown("""
        ### How to Use:
        1. **Select analysis methods** in the sidebar
        2. **Enter text** in the text area or use quick examples
        3. **Add manual assessment** if you want to compare against your judgment
        4. **Click 'Analyze Sentiment'** to see results
        5. **View comparisons** and detailed breakdowns
        
        ### About the Methods:
        - **Simple Rule-Based**: Counts positive/negative words in the text
        - **Pattern-Based**: Uses regex patterns and emoticons for analysis  
        - **Hugging Face (Demo)**: Simulates API-based analysis (needs real API key for production)
        - **Manual Assessment**: Your own judgment for accuracy comparison
        
        ### Tips for Better Results:
        - Try texts with clear emotional language
        - Test with different types of content (reviews, social media, etc.)
        - Compare results across different methods
        - Use manual assessment to evaluate accuracy
        - Check the statistics to see method performance over time
        """)

if __name__ == "__main__":
    main()