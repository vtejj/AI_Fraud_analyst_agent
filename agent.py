# agent.py (Version 7.0 - The Final, Knowledgeable Version)

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

# --- The Final Answering Agent Architecture ---
def create_chat_agent(llm, memory):
    """
    Creates the conversational agent that answers questions based on analysis
    and a detailed persona profile.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a witty and helpful AI Fraud Analyst. You must answer the user's questions based *only* on the two pieces of context provided below: the transaction analysis and the detailed persona characteristics.\n\n"
         "--- CONTEXT 1: TRANSACTION ANALYSIS ---\n"
         "{analysis_results}"
         "\n--- END OF CONTEXT 1 ---\n\n"
         "--- CONTEXT 2: DETAILED PERSONA CHARACTERISTICS ---\n"
         "{persona_details}"
         "\n--- END OF CONTEXT 2 ---\n\n"
         "IMPORTANT: Your entire response must be a single block of plain text. Do not use any markdown formatting like *, **, #, or newline characters (\\n)."),
        
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{question}")
    ])
    
    chain = RunnablePassthrough.assign(
        chat_history=lambda x: memory.load_memory_variables(x)["chat_history"]
    ) | prompt | llm | StrOutputParser()
    
    return chain