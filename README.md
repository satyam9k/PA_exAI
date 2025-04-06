# Explainable AI Data Analysis Toolkit 

This Streamlit application provides a multi-page toolkit for performing data analysis, running linear regression models, and generating AI-powered insights and explanations using Google's Gemini API.

## Features

*   **Multi-Page Interface:** Navigate easily between different analysis stages using the sidebar.
*   **Data Upload:** Load datasets from CSV or Excel files.
*   **Statistical Analysis:**
    *   View dataset overview (head).
    *   Calculate and display descriptive statistics.
    *   Visualize the distribution of numeric columns using histograms.
*   **Linear Regression:**
    *   Select target and feature variables from numeric columns.
    *   Automatically split data, scale features (StandardScaler), and train a Linear Regression model.
    *   Display key performance metrics: MSE, RMSE, MAE, R² Score.
    *   Show feature coefficients (importance).
    *   Visualize actual vs. predicted values with a scatter plot.
*   **Explainable AI Insights:**
    *   Leverage Google Gemini (`gemini-1.5-flash`) to generate natural language insights based on the regression model's performance metrics.
    *   Provides structured explanations covering performance overview, KPIs, potential improvements, and actionable recommendations.
*   **AI Technical Chat:**
    *   Interact with a specialized AI assistant (powered by Gemini) for technical questions across various domains.
    *   Get clear, detailed, and expert-level responses.
*   **Session State Management:** Persists uploaded data, analysis results, and chat history across different pages within a user session.
*   **Configuration:** Requires Google Gemini API key configured via environment variables.

## 🛠Technology Stack

*   **Frontend / UI:** Streamlit
*   **AI / LLM:** Google Gemini API (`gemini-1.5-flash`) via `google-generativeai` library
*   **Machine Learning:** Scikit-learn (`LinearRegression`, `train_test_split`, metrics, `StandardScaler`)
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Core Language:** Python 3

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher installed.
    *   Access to Google Gemini API and an API Key.

2.  **Clone the Repository (Optional):**
    ```bash
    git clone https://github.com/satyam9k/PA_exAI.git
    cd PA_exAI
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    google-generativeai
    # openpyxl  # Add if you need Excel (.xlsx) support for pandas
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    # If you need Excel support and didn't add it above:
    # pip install openpyxl 
    ```

5.  **Configure Environment Variable:**
    Set the `GEMINI_API_KEY` environment variable with your Google Gemini API Key. You can do this in your system environment or by creating a `.env` file (if you uncomment the `python-dotenv` lines in the code and install it: `pip install python-dotenv`).

    *Example using terminal (Linux/macOS):*
    ```bash
    export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    ```
    *Example using terminal (Windows CMD):*
    ```bash
    set GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    ```
    *Example using terminal (Windows PowerShell):*
    ```powershell
    $env:GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    ```
    *Example `.env` file (requires `python-dotenv` installed and uncommented code):*
    ```dotenv
    GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    ```
    **Note:** The script currently requires the API key to be available when the `ExplainableAIAnalyzer` is initialized. If it's not found, it will raise a `ValueError`.

## How to Run

1.  Ensure your virtual environment is activated and the `GEMINI_API_KEY` environment variable is set.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run exAI.py
    ```


3.  The application will open in your web browser.
4.  **Navigate:** Use the sidebar to move between pages:
    *   **Information & Guide:** Start here for an overview.
    *   **Data Upload & Analysis:** Upload your CSV/Excel file and view basic stats/plots.
    *   **Linear Regression:** Select features/target and run the regression model.
    *   **Explainable AI:** Generate and view AI insights about the regression results.
    *   **AI Chat:** Ask the AI assistant technical questions.

## How It Works

1.  **Initialization:** Sets up the Streamlit UI, initializes session state variables, and creates an instance of `ExplainableAIAnalyzer` which configures the connection to the Google Gemini API.
2.  **Data Handling:** Uses Pandas to load data from uploaded files. Session state (`st.session_state['dataframe']`) stores the loaded data for use across different pages.
3.  **Statistical Analysis:** Calculates descriptive statistics using Pandas and generates distribution plots using Matplotlib/Seaborn based on user selections.
4.  **Linear Regression:** Takes user input for features and target, performs data splitting and scaling using Scikit-learn, trains a `LinearRegression` model, calculates performance metrics, and visualizes results. Regression outcomes are stored in `st.session_state['regression_results']`.
5.  **Explainable AI:** The `ExplainableAIAnalyzer.generate_insights` method formats the stored regression metrics into a prompt for the Gemini model, asking for a structured analysis. The response is stored in `st.session_state['ai_insights']`.
6.  **AI Chat:** The `ExplainableAIAnalyzer.chat_with_ai` method takes the user's query, wraps it in a prompt defining the AI's role as a technical assistant, sends it to Gemini, and returns the response. Chat history is maintained in `st.session_state['chat_history']`.
7.  **Navigation & State:** Streamlit's sidebar controls page selection, and `st.session_state` ensures data and results persist as the user navigates between pages within a single session.

---
