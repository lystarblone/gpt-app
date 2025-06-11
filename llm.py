from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import pipeline

def get_llm():
    hf_pipeline = pipeline(
        task="text-generation",
        model="distilgpt2",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        pad_token_id=50256,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def get_conversation_chain():
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Grok, a helpful AI assistant. Answer in a concise and accurate manner."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return ChatMessageHistory()
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    return chain_with_history