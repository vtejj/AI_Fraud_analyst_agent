from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import json

from agent import create_chat_agent
from tools import get_ml_model_analysis, get_contextual_analysis
from schemas import TransactionInput
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory

# API and Model Setup 
app = FastAPI(title="AI Fraud Analyst API v6.0")
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Load the user persona database at startup
with open('user_database.json', 'r') as f:
    PERSONAS_DB = json.load(f)

# This dictionary will store session data
chat_sessions: Dict[str, Dict] = {}

# Request/Response Models 
class AnalyzeRequest(BaseModel):
    session_id: str
    transaction: TransactionInput

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ApiResponse(BaseModel):
    session_id: str
    response: str

#  API Endpoints 

@app.post("/analyze", response_model=ApiResponse)
async def analyze_transaction(request: AnalyzeRequest):
    """
    Analyzes a transaction, makes a decision, stores the results, and returns the initial analysis.
    """
    session_id = request.session_id
    transaction_data = request.transaction.model_dump()

    ml_results = get_ml_model_analysis.invoke({"transaction_data": transaction_data})
    context_results = get_contextual_analysis.invoke({"transaction_data": transaction_data})

    score = ml_results.get("fraud_probability_score", 0)
    decision = "APPROVE"
    if score >= 0.80: decision = "BLOCK"
    elif score >= 0.40: decision = "CHALLENGE"

    analysis_results = {
        "final_decision": decision,
        "ml_analysis": ml_results,
        "contextual_analysis": context_results
    }
    
    memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history")
    chat_agent = create_chat_agent(llm, memory)
    
    # We now also store the persona name found in the context analysis
    persona_name = context_results.get("persona_name", "unknown")
    
    chat_sessions[session_id] = {
        "analysis": json.dumps(analysis_results, indent=2),
        "agent": chat_agent,
        "memory": memory,
        "persona_name": persona_name # Store the persona name for later
    }

    initial_question = "Please provide a complete summary of your analysis and your final decision."
    response = await chat_agent.ainvoke({
        "analysis_results": chat_sessions[session_id]["analysis"],
        "question": initial_question,
        "persona_details": "{}" # Empty for the first turn
    })
    
    chat_sessions[session_id]["memory"].save_context({"input": initial_question}, {"output": response})
    return ApiResponse(session_id=session_id, response=response)

@app.post("/chat", response_model=ApiResponse)
async def chat_with_agent(request: ChatRequest):
    """Asks a follow-up question to an existing analysis session."""
    session_id = request.session_id
    question = request.question

    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please call /analyze first.")
    
    session = chat_sessions[session_id]
    chat_agent = session["agent"]
    memory = session["memory"]
    analysis = session["analysis"]
    
    # We retrieve the persona name and look up its full details from our database
    persona_name = session["persona_name"]
    persona_details = PERSONAS_DB.get(persona_name, {}) # Get the "textbook" page
    
    response = await chat_agent.ainvoke({
        "analysis_results": analysis,
        "question": question,
        "persona_details": json.dumps(persona_details, indent=2) # Give the "textbook" page to the agent
    })

    memory.save_context({"input": question}, {"output": response})
    return ApiResponse(session_id=session_id, response=response)