# Real-Time AI Fraud Analyst Agent

## Project Overview

This project implements a sophisticated, full-stack AI agent designed for real-time credit card fraud analysis. Moving beyond simple binary classification, this system acts as an "AI Fraud Analyst" that provides nuanced, actionable decisions (`APPROVE`, `CHALLENGE`, `BLOCK`) and can engage in a conversational follow-up to explain its reasoning.

The system is architected to combine the predictive power of supervised machine learning with the contextual understanding of unsupervised learning, creating a robust and intelligent fraud detection framework. The entire application is deployed via a professional, stateful FastAPI backend.


## Key Features

* **Hybrid AI System:** Combines a supervised RandomForest classifier for fraud prediction with an unsupervised K-Means clustering model for data-driven customer segmentation.
* **Data-Driven Personas:** Automatically generates "spending personas" from transaction data to provide a baseline for normal behavior, allowing for powerful contextual analysis.
* **Multi-Tool Agent:** Utilizes a **LangChain** agent equipped with two custom tools: one for ML model inference and another for contextual analysis against the generated personas.
* **Conversational AI:** The agent maintains a memory of the conversation, allowing users to ask specific follow-up questions about an analysis (e.g., "Why was that a challenge?", "What was the fraud score?").
* **Robust API Backend:** The entire system is served via a **FastAPI** application with two distinct endpoints (`/analyze` and `/chat`) for managing analysis sessions and conversations.
* **End-to-End MLOps:** The project includes scripts for the entire machine learning lifecycle, from data preprocessing and model training (`train_model.py`) to persona generation (`create_profiles.py`).

## Model Performance

The final RandomForest model was evaluated on a hold-out test set, demonstrating robust performance in identifying fraudulent transactions within a highly imbalanced dataset.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Test Set Size** | 85,443 | Number of transactions the model was tested on. |
| **ROC-AUC Score** | 0.9493 | The model's ability to distinguish between genuine and fraudulent transactions. |
| **PR-AUC Score** | 0.8360 | A key metric for imbalanced data, showing the trade-off between precision and recall. |
| **Fraud Recall** | 0.83 | **(At 0.40 Threshold)** - The model correctly identified 83% of all actual frauds. |
| **Fraud Precision**| 0.80 | **(At 0.40 Threshold)** - When the model flags a transaction, it is correct 80% of the time. |
| **Fraud F1-Score** | 0.81 | **(At 0.40 Threshold)** - The balanced score between precision and recall for the fraud class. |

## Agent Performance Enhancement

While the base RandomForest model performs well, this project's key innovation is the multi-tool AI agent that enriches the ML prediction with contextual analysis. To quantify this value-add, a targeted evaluation was performed on high-ambiguity transactions (where the model's fraud score was between 0.40 and 0.80).

In this critical "grey area," the AI agent's contextual logic **correctly escalated 37.50%** of confirmed fraudulent transactions from a simple `CHALLENGE` to an immediate `BLOCK`. This demonstrates a significant improvement in risk mitigation over a system relying on a single ML model, proving the architectural value of the hybrid AI approach.

## Technologies Used

* **Backend:** Python, FastAPI, Uvicorn
* **AI & Machine Learning:** LangChain, Groq (for Llama 3), Scikit-learn, Imbalanced-learn, Pandas, NumPy, XGBoost
* **Data:** K-Means Clustering, SMOTE (for oversampling)
* **Tooling:** VS Code, Git & GitHub

## Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vtejj/AI_Fraud_analyst_agent.git
    cd credit_agent
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the next section.)*

4.  **Set up your API Key:**
    * Create a file named `.env` in the root directory.
    * Add your Groq API key to this file: `GROQ_API_KEY="your-secret-key"`

5.  **Prepare the Models and Personas:**
    * First, run the persona creation script. You will be prompted to enter the optimal number of personas based on the generated `elbow_plot.png`.
        ```bash
        python create_profiles.py
        ```
    * Next, run the model training script to generate the `fraud_model.joblib` file.
        ```bash
        python train_model.py
        ```

6.  **Run the API Server:**
    ```bash
    uvicorn main:app --reload --port 8003
    ```
    The API will be available at `http://127.0.0.1:8003/docs`.

## How to Use the API

You can interact with the API via the auto-generated FastAPI documentation at `http://127.0.0.1:8003/docs`.

1.  **Start an Analysis (`/analyze` endpoint):**
    * This is the first call for any new transaction.
    * Provide a unique `session_id` and the full `transaction` object.
    * The agent will perform its full analysis and return the initial decision.

2.  **Ask a Follow-up Question (`/chat` endpoint):**
    * Use the same `session_id` from the analysis step.
    * Provide your follow-up question in the `question` field.
    * The agent will use its memory to provide a context-aware answer.

## Data Source

This project uses the "Credit Card Fraud Detection" dataset from Kaggle. It is a highly imbalanced dataset containing anonymized transaction data.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


