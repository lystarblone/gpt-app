from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import torch
import time
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_llm():
    logger.info("Начало загрузки модели...")
    try:
        model_name = "Qwen/Qwen2-1.5B-Instruct"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Используемое устройство: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        logger.info("Модель загружена успешно")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Токенизатор загружен")
        
        hf_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            device=device,
            return_full_text=False
        )
        logger.info("Пайплайн создан")
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise

def get_conversation_chain():
    logger.info("Создание цепочки разговора...")
    try:
        llm = get_llm()
        logger.info("LLM получен, создание промпта...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты — Grok, ИИ-ассистент, созданный xAI. Отвечай только на вопрос пользователя, без повторения запроса или системного промпта. Давай точные, краткие и полезные ответы на русском языке. Если в запросе есть некорректные данные, четко укажи все ошибки и предоставь правильную информацию, основываясь на исторических фактах. Проверяй факты и избегай выдумок."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        logger.info("Промпт создан, сборка цепочки...")
        chain = prompt | llm
        
        session_histories = defaultdict(ChatMessageHistory)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            history = session_histories[session_id]
            if len(history.messages) > 8:
                history.messages = history.messages[-8:]
            return history
        
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        logger.info("Цепочка разговора создана")
        return chain_with_history
    except Exception as e:
        logger.error(f"Ошибка создания цепочки: {str(e)}")
        raise

def clean_response(response):
    """Очистка ответа от системного промпта, меток и лишнего текста."""
    if not isinstance(response, str):
        response = str(response)
    
    markers = ["System:", "Human:", "Assistant:"]
    cleaned_response = response
    for marker in markers:
        if marker in cleaned_response:
            parts = cleaned_response.split(marker)
            cleaned_response = parts[-1]
    
    cleaned_response = cleaned_response.strip()
    
    if "Проверь следующий текст" in cleaned_response:
        cleaned_response = cleaned_response.split("Проверь следующий текст")[-1].strip()
    
    return cleaned_response

if __name__ == "__main__":
    try:
        logger.info(f"MPS доступен: {torch.backends.mps.is_available()}")
        chain = get_conversation_chain()
        start_time = time.time()
        response = chain.invoke(
            {
                "input": (
                    "Проверь следующий текст на исторические ошибки и дай краткий исправленный вариант: "
                    "Kavtsov (1929), who was killed by the Soviet Union in 1941 during a battle with its Russian allies "
                    "and returned to Russia after World War II for his studies on German national politics. "
                    "In 2012 Khrushchev took office again seeking reëlection within Vladimir Lenin Center..."
                )
            },
            config={"configurable": {"session_id": "test_session"}}
        )
        end_time = time.time()
        cleaned_response = clean_response(response)
        logger.info(f"Время генерации: {end_time - start_time:.2f} секунд")
        print("Ответ модели:", cleaned_response)
    except Exception as e:
        logger.error(f"Ошибка при вызове цепочки: {str(e)}")