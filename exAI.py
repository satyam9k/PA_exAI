import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv
import os
import io

class ExplainableAIAnalyzer:
    def __init__(self):
        """Initialize Gemini API connection and Explainable AI Analyzer."""
        load_dotenv()
        api_key = "AIzaSyB-8BpC4oEsK0LF9Ap_2OTGM9hRLWA4nS4"
        
        if not api_key:
            raise ValueError("GEMINI API KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_insights(self, performance_metrics: dict):
        """Generate insights from performance metrics and analysis."""
        prompt = f"""
        Provide a comprehensive analysis of the machine learning model's performance:

        Performance Metrics:
        {performance_metrics}

        Please structure your response with:
        1. Model Performance Overview
        2. Key Performance Indicators
        3. Potential Improvements
        4. Actionable Recommendations

        Maintain a professional, data-driven tone with clear, concise explanations.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating insights: {e}")
            return None

    def chat_with_ai(self, user_query: str):
        """
        AI chat function to handle technical queries across all domains.
        Provides comprehensive technical assistance.
        """
        prompt = f"""
        You are a comprehensive technical AI assistant. 
        Provide a clear, detailed, and expert-level response to the following query:

        Query: {user_query}

        Guidelines for response:
        - Deliver technically accurate information
        - Explain concepts clearly and comprehensively
        - Use appropriate technical depth
        - Provide practical insights and explanations
        - Include relevant examples or code snippets if applicable
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error in chat functionality: {e}")
            return None


def initialize_session_state():
    """Initialize or reset session state variables."""
    # Ensure all session state variables are initialized
    session_keys = [
        'dataframe', 
        'page', 
        'uploaded_file', 
        'regression_results', 
        'ai_insights', 
        'chat_history',
        'data_insights'
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            if key == 'chat_history':
                st.session_state[key] = []
            else:
                st.session_state[key] = None

def load_data(uploaded_file):
    """Load data from CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def page_information_and_guide():
    """Page 01: Information and Usage Guide"""
    st.title("🧠 Explainable AI Data Analysis Toolkit")
    
    st.markdown("""
    ### Welcome to the Advanced Data Analysis Platform

    #### Key Features:
    1. **Statistical Analysis**
       - Descriptive statistics
       - Data distribution visualization

    2. **Linear Regression**
       - Automated model training
       - Performance metrics evaluation
       - Coefficient analysis

    3. **Explainable AI**
       - AI-generated model insights
       - Interactive technical chat

    #### How to Use:
    - Upload a CSV or Excel file from the sidebar
    - Select target variable for regression
    - Explore statistical insights
    - Generate AI-powered explanations
    - Chat with our specialized AI assistant
    """)

    st.info("💡 Tip: Start by uploading a dataset with numeric columns for best results!")

def page_data_upload_and_analysis():
    """Page 02: Data Upload and Statistical Analysis"""
    st.title("📊 Data Upload & Statistical Exploration")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel File", 
                                     type=["csv", "xlsx", "xls"])

    # Update session state with uploaded file
    st.session_state.uploaded_file = uploaded_file

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Update session state
            st.session_state.dataframe = df
            
            st.subheader("Dataset Overview")
            st.write(df.head())
            
            # Descriptive Statistics
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe())

            # Numeric Columns Distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                st.subheader("Distribution of Numeric Columns")
                
                # Select columns for visualization
                selected_cols = st.multiselect(
                    "Select columns to visualize", 
                    list(numeric_cols)
                )
                
                if selected_cols:
                    fig, axes = plt.subplots(
                        nrows=len(selected_cols), 
                        ncols=1, 
                        figsize=(10, 4*len(selected_cols))
                    )
                    
                    for i, col in enumerate(selected_cols):
                        ax = axes[i] if len(selected_cols) > 1 else axes
                        sns.histplot(df[col], kde=True, ax=ax)
                        ax.set_title(f'Distribution of {col}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

            else:
                st.warning("No numeric columns found for analysis.")

def page_linear_regression():
    """Page 03: Linear Regression and Evaluation Metrics"""
    st.title("📈 Linear Regression Analysis")

    # Check if dataframe exists
    if st.session_state.dataframe is None:
        st.warning("Please upload a dataset first")
        return

    df = st.session_state.dataframe

    # Select features and target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    target_col = st.selectbox("Select Target Variable", list(numeric_cols))
    feature_cols = st.multiselect("Select Feature Variables", 
                                   [col for col in numeric_cols if col != target_col])

    if st.button("Run Linear Regression"):
        if not feature_cols:
            st.error("Please select at least one feature column")
            return

        # Prepare data
        X = df[feature_cols]
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store regression results
        regression_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_cols': feature_cols,
            'coefficients': model.coef_
        }
        st.session_state.regression_results = regression_results

        # Display results
        st.subheader("Regression Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
        with col3:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")

        # Coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)

        st.subheader("Feature Importance")
        st.dataframe(coef_df)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)

def page_explainable_ai(analyzer):
    """Page 04: Explainable AI Insights"""
    st.title("🧠 Explainable AI Insights")
    
    # Check if regression results exist
    if st.session_state.regression_results is None:
        st.warning("Please run Linear Regression first")
        return

    # Generate AI Insights if not already generated
    if st.session_state.ai_insights is None:
        if st.button("Generate Explainable AI Insights"):
            insights = analyzer.generate_insights(st.session_state.regression_results)
            st.session_state.ai_insights = insights
    
    # Display Insights
    if st.session_state.ai_insights:
        st.subheader("🤖 AI Model Insights")
        st.markdown(st.session_state.ai_insights)

def page_ai_chat(analyzer):
    """Page 05: AI Chat Interface"""
    st.title("💬 Technical AI Assistant")
    
    # Chat input
    user_query = st.text_input("Ask a technical question:")
    
    # Ensure chat history is initialized
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Send query
    if st.button("Get Answer"):
        if user_query:
            # Get AI response
            response = analyzer.chat_with_ai(user_query)
            
            # Update chat history
            st.session_state.chat_history.append({
                'query': user_query,
                'response': response
            })
    
    # Display chat history
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['query']}")
            st.markdown(f"**AI:** {chat['response']}")
            st.markdown("---")

def main():
    # Initialize session state
    initialize_session_state()

    # Create analyzer
    analyzer = ExplainableAIAnalyzer()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "Information & Guide", 
        "Data Upload & Analysis", 
        "Linear Regression",
        "Explainable AI",
        "AI Chat"
    ]
    
    # Page selection with session state persistence
    selected_page = st.sidebar.radio("Select Page", pages)

    # Routing based on selected page
    if selected_page == "Information & Guide":
        page_information_and_guide()
    
    elif selected_page == "Data Upload & Analysis":
        page_data_upload_and_analysis()
    
    elif selected_page == "Linear Regression":
        page_linear_regression()
    
    elif selected_page == "Explainable AI":
        page_explainable_ai(analyzer)
    
    elif selected_page == "AI Chat":
        page_ai_chat(analyzer)

if __name__ == "__main__":
    main()