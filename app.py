import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import time
import re
import numpy as np
from io import BytesIO
import xlsxwriter

# Page configuration
st.set_page_config(
    page_title="🎭 AI Sentiment Analysis Dashboard with Bias Detection",
    page_icon="😊",
    layout="wide"
)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""
if 'bias_test_results' not in st.session_state:
    st.session_state.bias_test_results = []

# Bias detection patterns inspired by What-If Tool concepts
BIAS_TEST_CASES = {
    "Gender Bias": {
        "test_pairs": [
            ("The doctor performed excellently", "She performed excellently"),
            ("The nurse was very professional", "He was very professional"),
            ("The engineer solved the problem", "She solved the problem"),
            ("The teacher explained clearly", "He explained clearly")
        ],
        "description": "Tests for gender-based sentiment differences"
    },
    "Racial/Ethnic Bias": {
        "test_pairs": [
            ("John is a great worker", "Jamal is a great worker"),
            ("Emily did excellent work", "Aisha did excellent work"),
            ("The team leader was effective", "The diverse team leader was effective")
        ],
        "description": "Tests for potential racial/ethnic bias in sentiment analysis"
    },
    "Age Bias": {
        "test_pairs": [
            ("The employee performed well", "The young employee performed well"),
            ("The manager was decisive", "The older manager was decisive"),
            ("The intern showed promise", "The experienced intern showed promise")
        ],
        "description": "Tests for age-related bias in sentiment detection"
    },
    "Professional Bias": {
        "test_pairs": [
            ("The worker did good job", "The CEO did good job"),
            ("The assistant was helpful", "The executive was helpful"),
            ("The staff member contributed", "The senior staff member contributed")
        ],
        "description": "Tests for professional status bias"
    }
}

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
            confidence = min(positive_ratio * 5, 1.0)
        elif negative_count > positive_count:
            sentiment = "Negative"  
            confidence = min(negative_ratio * 5, 1.0)
        else:
            sentiment = "Neutral"
            confidence = 0.1
            
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_score': positive_count - negative_count,
            'details': f"Positive words: {positive_count}, Negative words: {negative_count}",
            'positive_count': positive_count,
            'negative_count': negative_count
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
            'details': f"Positive patterns: {positive_score}, Negative patterns: {negative_score}",
            'positive_patterns': positive_score,
            'negative_patterns': negative_score
        }
    except Exception as e:
        return {'sentiment': 'Error', 'confidence': 0, 'raw_score': 0, 'details': str(e)}

def run_bias_detection():
    """Run bias detection tests inspired by What-If Tool"""
    bias_results = {}
    
    for bias_type, test_data in BIAS_TEST_CASES.items():
        bias_results[bias_type] = {
            'test_results': [],
            'bias_detected': False,
            'max_difference': 0,
            'description': test_data['description']
        }
        
        for baseline_text, modified_text in test_data['test_pairs']:
            # Analyze both texts
            baseline_simple = get_simple_sentiment(baseline_text)
            baseline_pattern = get_pattern_sentiment(baseline_text)
            
            modified_simple = get_simple_sentiment(modified_text)
            modified_pattern = get_pattern_sentiment(modified_text)
            
            # Calculate differences
            simple_diff = abs(baseline_simple['confidence'] - modified_simple['confidence'])
            pattern_diff = abs(baseline_pattern['confidence'] - modified_pattern['confidence'])
            
            # Check for sentiment flips
            simple_flip = baseline_simple['sentiment'] != modified_simple['sentiment']
            pattern_flip = baseline_pattern['sentiment'] != modified_pattern['sentiment']
            
            test_result = {
                'baseline_text': baseline_text,
                'modified_text': modified_text,
                'simple_confidence_diff': simple_diff,
                'pattern_confidence_diff': pattern_diff,
                'simple_sentiment_flip': simple_flip,
                'pattern_sentiment_flip': pattern_flip,
                'baseline_simple': baseline_simple,
                'modified_simple': modified_simple,
                'baseline_pattern': baseline_pattern,
                'modified_pattern': modified_pattern
            }
            
            bias_results[bias_type]['test_results'].append(test_result)
            
            # Update max difference and bias detection
            max_diff = max(simple_diff, pattern_diff)
            if max_diff > bias_results[bias_type]['max_difference']:
                bias_results[bias_type]['max_difference'] = max_diff
            
            # Consider bias detected if there's a significant difference or sentiment flip
            if simple_flip or pattern_flip or max_diff > 0.3:
                bias_results[bias_type]['bias_detected'] = True
    
    return bias_results

def create_bias_visualization(bias_results):
    """Create visualization for bias detection results"""
    bias_types = list(bias_results.keys())
    max_differences = [bias_results[bt]['max_difference'] for bt in bias_types]
    bias_detected = [bias_results[bt]['bias_detected'] for bt in bias_types]
    
    colors = ['red' if detected else 'green' for detected in bias_detected]
    
    fig = go.Figure(data=[
        go.Bar(
            x=bias_types,
            y=max_differences,
            marker_color=colors,
            text=[f"{'⚠️ Bias' if detected else '✅ No Bias'}" for detected in bias_detected],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Bias Detection Results',
        xaxis_title='Bias Type',
        yaxis_title='Maximum Confidence Difference',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

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

def generate_audit_report():
    """Generate a comprehensive audit report in Excel format"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1,
            'font_size': 12
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1,
            'font_size': 10
        })
        
        number_format = workbook.add_format({
            'num_format': '0.000',
            'border': 1,
            'align': 'center'
        })
        
        # Sheet 1: Analysis History
        if st.session_state.results_history:
            analysis_data = []
            for i, entry in enumerate(st.session_state.results_history):
                base_row = {
                    'Analysis_ID': f"#{i+1}",
                    'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Text_Sample': entry['text'][:200] + "..." if len(entry['text']) > 200 else entry['text'],
                    'Text_Length': len(entry['text']),
                    'Word_Count': len(entry['text'].split())
                }
                
                for method, result in entry['results'].items():
                    row = base_row.copy()
                    row.update({
                        'Method': method,
                        'Sentiment': result['sentiment'],
                        'Confidence': result['confidence'],
                        'Raw_Score': result.get('raw_score', 0),
                        'Details': result.get('details', '')
                    })
                    analysis_data.append(row)
            
            df_analysis = pd.DataFrame(analysis_data)
            df_analysis.to_excel(writer, sheet_name='Analysis_History', index=False)
            
            worksheet = writer.sheets['Analysis_History']
            worksheet.set_column('A:A', 12, cell_format)  # Analysis_ID
            worksheet.set_column('B:B', 20, cell_format)  # Timestamp
            worksheet.set_column('C:C', 50, cell_format)  # Text_Sample
            worksheet.set_column('D:D', 12, number_format)  # Text_Length
            worksheet.set_column('E:E', 12, number_format)  # Word_Count
            worksheet.set_column('F:F', 20, cell_format)  # Method
            worksheet.set_column('G:G', 15, cell_format)  # Sentiment
            worksheet.set_column('H:H', 12, number_format)  # Confidence
            worksheet.set_column('I:I', 12, number_format)  # Raw_Score
            worksheet.set_column('J:J', 40, cell_format)  # Details
            
            # Apply header format
            for col_num, value in enumerate(df_analysis.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        # Sheet 2: Bias Detection Results
        if st.session_state.bias_test_results:
            bias_data = []
            for bias_result in st.session_state.bias_test_results:
                for bias_type, data in bias_result.items():
                    for test in data['test_results']:
                        bias_data.append({
                            'Bias_Type': bias_type,
                            'Test_Description': data['description'],
                            'Baseline_Text': test['baseline_text'],
                            'Modified_Text': test['modified_text'],
                            'Simple_Confidence_Diff': test['simple_confidence_diff'],
                            'Pattern_Confidence_Diff': test['pattern_confidence_diff'],
                            'Simple_Sentiment_Flip': 'Yes' if test['simple_sentiment_flip'] else 'No',
                            'Pattern_Sentiment_Flip': 'Yes' if test['pattern_sentiment_flip'] else 'No',
                            'Baseline_Simple_Sentiment': test['baseline_simple']['sentiment'],
                            'Modified_Simple_Sentiment': test['modified_simple']['sentiment'],
                            'Baseline_Pattern_Sentiment': test['baseline_pattern']['sentiment'],
                            'Modified_Pattern_Sentiment': test['modified_pattern']['sentiment'],
                            'Bias_Detected': 'Yes' if data['bias_detected'] else 'No',
                            'Max_Difference': data['max_difference']
                        })
            
            if bias_data:
                df_bias = pd.DataFrame(bias_data)
                df_bias.to_excel(writer, sheet_name='Bias_Detection', index=False)
                
                worksheet = writer.sheets['Bias_Detection']
                worksheet.set_column('A:A', 15, cell_format)  # Bias_Type
                worksheet.set_column('B:B', 40, cell_format)  # Test_Description
                worksheet.set_column('C:C', 35, cell_format)  # Baseline_Text
                worksheet.set_column('D:D', 35, cell_format)  # Modified_Text
                worksheet.set_column('E:E', 20, number_format)  # Simple_Confidence_Diff
                worksheet.set_column('F:F', 20, number_format)  # Pattern_Confidence_Diff
                worksheet.set_column('G:G', 20, cell_format)  # Simple_Sentiment_Flip
                worksheet.set_column('H:H', 20, cell_format)  # Pattern_Sentiment_Flip
                worksheet.set_column('I:I', 20, cell_format)  # Baseline_Simple_Sentiment
                worksheet.set_column('J:J', 20, cell_format)  # Modified_Simple_Sentiment
                worksheet.set_column('K:K', 20, cell_format)  # Baseline_Pattern_Sentiment
                worksheet.set_column('L:L', 20, cell_format)  # Modified_Pattern_Sentiment
                worksheet.set_column('M:M', 15, cell_format)  # Bias_Detected
                worksheet.set_column('N:N', 15, number_format)  # Max_Difference
                
                # Apply header format
                for col_num, value in enumerate(df_bias.columns.values):
                    worksheet.write(0, col_num, value, header_format)
        
        # Sheet 3: Summary Statistics
        if st.session_state.results_history:
            summary_stats = []
            
            # Overall statistics
            total_analyses = len(st.session_state.results_history)
            all_results = []
            for entry in st.session_state.results_history:
                for method, result in entry['results'].items():
                    all_results.append({
                        'Method': method,
                        'Sentiment': result['sentiment'],
                        'Confidence': result['confidence']
                    })
            
            if all_results:
                df_all = pd.DataFrame(all_results)
                
                # Method-wise statistics
                method_stats = df_all.groupby('Method').agg({
                    'Confidence': ['mean', 'std', 'min', 'max'],
                    'Sentiment': 'count'
                }).round(4)
                
                method_stats.columns = ['Mean_Confidence', 'Std_Confidence', 'Min_Confidence', 'Max_Confidence', 'Total_Count']
                method_stats.reset_index(inplace=True)
                method_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False, startrow=0)
                
                # Sentiment distribution
                sentiment_dist = df_all['Sentiment'].value_counts().to_frame('Count')
                sentiment_dist['Percentage'] = (sentiment_dist['Count'] / sentiment_dist['Count'].sum() * 100).round(2)
                sentiment_dist.reset_index(inplace=True)
                sentiment_dist.columns = ['Sentiment', 'Count', 'Percentage']
                sentiment_dist.to_excel(writer, sheet_name='Summary_Statistics', index=False, startrow=len(method_stats) + 3)
                
                worksheet = writer.sheets['Summary_Statistics']
                worksheet.set_column('A:F', 20, cell_format)
                
                # Headers
                worksheet.write(0, 0, 'METHOD PERFORMANCE', header_format)
                for col_num, value in enumerate(method_stats.columns.values):
                    worksheet.write(1, col_num, value, header_format)
                
                worksheet.write(len(method_stats) + 3, 0, 'SENTIMENT DISTRIBUTION', header_format)
                for col_num, value in enumerate(sentiment_dist.columns.values):
                    worksheet.write(len(method_stats) + 4, col_num, value, header_format)
    
    output.seek(0)
    return output

def main():
    st.title("🎭 AI Sentiment Analysis Dashboard with Bias Detection")
    st.markdown("Analyze text sentiment using multiple methods, detect bias, and generate comprehensive audit reports")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "🔍 Bias Detection", "📊 Analytics", "📋 Audit Report"])
    
    with tab1:
        # Sidebar for settings
        st.sidebar.header("Analysis Methods")
        
        use_simple = st.sidebar.checkbox("Simple Rule-Based", value=True, help="Uses positive/negative word lists")
        use_pattern = st.sidebar.checkbox("Pattern-Based", value=True, help="Uses regex patterns and emoticons")
        use_manual = st.sidebar.checkbox("Manual Assessment", value=False, help="Add your own sentiment rating")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Note**: This version includes bias detection using What-If Tool concepts.")
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📝 Text Analysis")
            
            # Sample texts for testing
            st.markdown("**Quick Test Examples (click to load):**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("😊 Positive Example"):
                    st.session_state.example_text = "I absolutely love this product! It's fantastic and works perfectly. Highly recommended!"
            with col_b:
                if st.button("😞 Negative Example"):
                    st.session_state.example_text = "This is terrible! I hate it so much. Worst purchase ever. Complete waste of money."
            with col_c:
                if st.button("😐 Neutral Example"):
                    st.session_state.example_text = "The product arrived on time. It has basic features and standard quality. Nothing special."
            
            # Text input
            user_text = st.text_area(
                "Enter text to analyze:", 
                value=st.session_state.example_text,
                placeholder="Type or paste your text here...\n\nOr click one of the example buttons above to load sample text.", 
                height=150,
                key="text_input"
            )
            
            # Update session state when text changes
            if user_text != st.session_state.example_text:
                st.session_state.example_text = user_text
            
            # Show current text length
            if user_text.strip():
                st.write(f"📝 Text length: {len(user_text)} characters, Words: {len(user_text.split())}")
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
                    'text': user_text,
                    'results': results,
                    'text_length': len(user_text),
                    'word_count': len(user_text.split())
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
                        display_text = entry['text'][:100] + "..." if len(entry['text']) > 100 else entry['text']
                        st.write(f"_{display_text}_")
                        st.write("**Results:**")
                        for method, result in entry['results'].items():
                            emoji = '😊' if result['sentiment'] == 'Positive' else '😞' if result['sentiment'] == 'Negative' else '😐'
                            st.write(f"{emoji} **{method}**: {result['sentiment']} ({result['confidence']:.2f})")
                
                # Clear history button
                if st.button("🗑️ Clear History", type="secondary"):
                    st.session_state.results_history = []
                    st.session_state.bias_test_results = []
                    st.rerun()
            else:
                st.info("No analyses yet. Enter some text and click 'Analyze Sentiment' to get started!")
    
    with tab2:
        st.header("🔍 Bias Detection - What-If Tool Inspired")
        st.markdown("Test your sentiment analysis models for potential biases using counterfactual examples.")
        
        if st.button("🧪 Run Bias Detection Tests", type="primary"):
            with st.spinner("Running bias detection tests..."):
                bias_results = run_bias_detection()
                st.session_state.bias_test_results.append(bias_results)
            
            st.success("Bias detection tests completed!")
            
            # Display results
            st.subheader("📊 Bias Detection Results")
            
            # Overall summary
            total_bias_types = len(bias_results)
            bias_detected_count = sum(1 for data in bias_results.values() if data['bias_detected'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bias Tests", total_bias_types)
            with col2:
                st.metric("Bias Detected", bias_detected_count, delta=f"{bias_detected_count/total_bias_types:.1%}")
            with col3:
                st.metric("Clean Tests", total_bias_types - bias_detected_count)
            
            # Visualization
            fig_bias = create_bias_visualization(bias_results)
            st.plotly_chart(fig_bias, use_container_width=True)
            
            # Detailed results
            for bias_type, data in bias_results.items():
                with st.expander(f"🔍 {bias_type} - {'⚠️ Bias Detected' if data['bias_detected'] else '✅ No Bias'}"):
                    st.write(f"**Description:** {data['description']}")
                    st.write(f"**Maximum Difference:** {data['max_difference']:.3f}")
                    
                    # Show test details
                    for i, test in enumerate(data['test_results']):
                        st.markdown(f"**Test {i+1}:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Baseline:**")
                            st.write(f"Text: _{test['baseline_text']}_")
                            st.write(f"Simple: {test['baseline_simple']['sentiment']} ({test['baseline_simple']['confidence']:.3f})")
                            st.write(f"Pattern: {test['baseline_pattern']['sentiment']} ({test['baseline_pattern']['confidence']:.3f})")
                        
                        with col2:
                            st.write("**Modified:**")
                            st.write(f"Text: _{test['modified_text']}_")
                            st.write(f"Simple: {test['modified_simple']['sentiment']} ({test['modified_simple']['confidence']:.3f})")
                            st.write(f"Pattern: {test['modified_pattern']['sentiment']} ({test['modified_pattern']['confidence']:.3f})")
                        
                        # Show differences
                        if test['simple_sentiment_flip'] or test['pattern_sentiment_flip'] or test['simple_confidence_diff'] > 0.2:
                            st.warning(f"⚠️ Potential bias detected:")
                            if test['simple_sentiment_flip']:
                                st.write(f"- Simple method sentiment flip: {test['baseline_simple']['sentiment']} → {test['modified_simple']['sentiment']}")
                            if test['pattern_sentiment_flip']:
                                st.write(f"- Pattern method sentiment flip: {test['baseline_pattern']['sentiment']} → {test['modified_pattern']['sentiment']}")
                            if test['simple_confidence_diff'] > 0.2:
                                st.write(f"- Simple confidence difference: {test['simple_confidence_diff']:.3f}")
                            if test['pattern_confidence_diff'] > 0.2:
                                st.write(f"- Pattern confidence difference: {test['pattern_confidence_diff']:.3f}")
                        
                        st.markdown("---")
        
        # Show historical bias results
        if st.session_state.bias_test_results:
            st.subheader("📈 Bias Testing History")
            
            # Aggregate historical data
            historical_bias = {}
            for result_set in st.session_state.bias_test_results:
                for bias_type, data in result_set.items():
                    if bias_type not in historical_bias:
                        historical_bias[bias_type] = []
                    historical_bias[bias_type].append({
                        'bias_detected': data['bias_detected'],
                        'max_difference': data['max_difference']
                    })
            
            # Show trends
            for bias_type, history in historical_bias.items():
                bias_rate = sum(1 for h in history if h['bias_detected']) / len(history)
                avg_diff = sum(h['max_difference'] for h in history) / len(history)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{bias_type} - Bias Rate", f"{bias_rate:.1%}")
                with col2:
                    st.metric(f"{bias_type} - Avg Max Diff", f"{avg_diff:.3f}")
    
    with tab3:
        st.header("📊 Analytics & Insights")
        
        if len(st.session_state.results_history) >= 2:
            # Collect all results for statistics
            all_results = []
            for entry in st.session_state.results_history:
                for method, result in entry['results'].items():
                    all_results.append({
                        'Method': method,
                        'Sentiment': result['sentiment'],
                        'Confidence': result['confidence'],
                        'Timestamp': entry['timestamp'],
                        'Text_Length': entry.get('text_length', 0),
                        'Word_Count': entry.get('word_count', 0)
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
                        'Confidence': ['mean', 'std'],
                        'Sentiment': 'count'
                    }).round(3)
                    method_stats.columns = ['Avg Confidence', 'Std Confidence', 'Total Analyses']
                    st.write("**Method Performance:**")
                    st.dataframe(method_stats, use_container_width=True)
                
                # Confidence distribution by method
                fig_box = px.box(df_stats, x='Method', y='Confidence', color='Sentiment',
                               title='Confidence Distribution by Method and Sentiment')
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Time series analysis if enough data points
                if len(df_stats) > 5:
                    df_stats['Hour'] = df_stats['Timestamp'].dt.hour
                    hourly_sentiment = df_stats.groupby(['Hour', 'Sentiment']).size().reset_index(name='Count')
                    
                    fig_time = px.line(hourly_sentiment, x='Hour', y='Count', color='Sentiment',
                                     title='Sentiment Analysis Over Time (by Hour)')
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Text length analysis
                st.subheader("📝 Text Characteristics Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_length = px.scatter(df_stats, x='Text_Length', y='Confidence', color='Sentiment',
                                          title='Text Length vs Confidence')
                    st.plotly_chart(fig_length, use_container_width=True)
                
                with col2:
                    fig_words = px.scatter(df_stats, x='Word_Count', y='Confidence', color='Sentiment',
                                         title='Word Count vs Confidence')
                    st.plotly_chart(fig_words, use_container_width=True)
                
                # Agreement analysis
                if len(df_stats['Method'].unique()) > 1:
                    st.subheader("🤝 Method Agreement Analysis")
                    
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
                                             title='Method Agreement Levels',
                                             color=agreement_counts.values,
                                             color_continuous_scale='RdYlGn')
                        st.plotly_chart(fig_agreement, use_container_width=True)
        else:
            st.info("📊 Perform at least 2 analyses to see detailed analytics and insights!")
    
    with tab4:
        st.header("📋 Comprehensive Audit Report")
        st.markdown("Generate detailed audit reports with bias detection results and performance metrics.")
        
        if st.session_state.results_history or st.session_state.bias_test_results:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Report Summary")
                
                # Summary metrics
                total_analyses = len(st.session_state.results_history)
                total_bias_tests = len(st.session_state.bias_test_results)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Analyses", total_analyses)
                with col_b:
                    st.metric("Bias Test Runs", total_bias_tests)
                with col_c:
                    if st.session_state.bias_test_results:
                        latest_bias = st.session_state.bias_test_results[-1]
                        bias_detected = sum(1 for data in latest_bias.values() if data['bias_detected'])
                        st.metric("Recent Bias Issues", bias_detected)
                    else:
                        st.metric("Recent Bias Issues", "N/A")
                
                # Report contents preview
                st.subheader("📑 Report Contents")
                report_sections = [
                    "✅ Analysis History - Complete record of all sentiment analyses",
                    "✅ Bias Detection Results - Detailed bias testing outcomes",
                    "✅ Method Performance Statistics - Comparative analysis metrics",
                    "✅ Sentiment Distribution - Overall sentiment patterns",
                    "✅ Confidence Analysis - Method reliability assessment",
                    "✅ Text Characteristics - Length and complexity analysis"
                ]
                
                for section in report_sections:
                    st.write(section)
            
            with col2:
                st.subheader("⬇️ Download Report")
                
                if st.button("📊 Generate Excel Report", type="primary"):
                    with st.spinner("Generating comprehensive audit report..."):
                        excel_buffer = generate_audit_report()
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer,
                        file_name=f"sentiment_analysis_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success("✅ Report generated successfully!")
                
                st.markdown("---")
                st.markdown("**Report Features:**")
                st.markdown("• 📊 Formatted Excel sheets")
                st.markdown("• 🎨 Color-coded headers")
                st.markdown("• 📏 Auto-sized columns")
                st.markdown("• 📈 Statistical summaries")
                st.markdown("• 🔍 Bias detection details")
                st.markdown("• 📱 Mobile-friendly format")
        
        else:
            st.info("📝 No data available for report generation. Please run some analyses first!")
            
            # Instructions
            st.markdown("""
            ### 🚀 Getting Started:
            1. **Go to Text Analysis tab** and analyze some text samples
            2. **Run Bias Detection tests** to check for model fairness
            3. **Return here** to generate comprehensive audit reports
            
            ### 📊 Report Features:
            - **Analysis History**: Complete record with timestamps and details
            - **Bias Detection**: What-If Tool inspired fairness testing
            - **Performance Metrics**: Method comparison and reliability stats
            - **Excel Format**: Professional, easy-to-read spreadsheet output
            - **Auto-formatting**: Proper column widths and text wrapping
            """)
    
    # Instructions and tips
    with st.expander("ℹ️ How to Use & Tips"):
        st.markdown("""
        ### How to Use:
        1. **Text Analysis Tab**: 
           - Select analysis methods in the sidebar
           - Enter text or use example buttons
           - Compare results across different methods
        
        2. **Bias Detection Tab**:
           - Run comprehensive bias tests inspired by Google's What-If Tool
           - Check for gender, racial, age, and professional biases
           - View detailed counterfactual analysis results
        
        3. **Analytics Tab**:
           - View comprehensive statistics and trends
           - Analyze method agreement and performance
           - Explore text characteristics and patterns
        
        4. **Audit Report Tab**:
           - Generate professional Excel reports
           - Download formatted audit trails
           - Share results with stakeholders
        
        ### About Bias Detection:
        Our bias detection uses **counterfactual analysis** similar to Google's What-If Tool:
        - **Gender Bias**: Tests sentiment changes when gender indicators are modified
        - **Racial/Ethnic Bias**: Checks for differential treatment based on names/identifiers
        - **Age Bias**: Evaluates age-related sentiment variations
        - **Professional Bias**: Tests for status-based sentiment differences
        
        ### Tips for Better Results:
        - **Run multiple analyses** to build comprehensive statistics
        - **Use bias detection regularly** to ensure fairness
        - **Compare methods** to understand their strengths and weaknesses
        - **Generate audit reports** for documentation and compliance
        - **Monitor trends over time** using the analytics dashboard
        
        ### Excel Report Features:
        - ✅ **Properly formatted cells** with text wrapping
        - ✅ **Auto-sized columns** for easy reading
        - ✅ **Color-coded headers** for better organization
        - ✅ **Multiple worksheets** for different data types
        - ✅ **Statistical summaries** and bias detection details
        """)

if __name__ == "__main__":
    main()