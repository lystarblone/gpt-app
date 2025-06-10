from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline

def get_llm():
    hf_pipeline = pipeline(
        task="text-generation",
        model="distilgpt2",
        max_length=512,
        do_sample=True,
        temperature=0.1
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def get_conversation_chain():
    llm = get_llm()
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)