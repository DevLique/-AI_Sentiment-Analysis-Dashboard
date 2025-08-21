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
    page_title="Sentiment Analysis Dashboard",
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