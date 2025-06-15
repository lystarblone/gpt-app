from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_llm():
    logger.info("Начало загрузки модели...")
    try:
        model_name = "gpt2"  # Легкая модель для теста
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Модель загружена успешно")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Токенизатор загружен")
        hf_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Уменьшено для экономии ресурсов
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
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
            ("system", """Ты — Grok, ИИ-ассистент, созданный xAI. Твоя задача — давать максимально полезные, точные и информативные ответы на русском языке. Следуй этим принципам:
            Полезность: Всегда стремись дать ответ, который решает проблему пользователя или отвечает на его вопрос.
            Точность: Проверяй факты и избегай выдумок.
            Ясность: Пиши простым языком.
            Контекст: Учитывай историю диалога.
            Тон: Будь дружелюбным и профессиональным.
            Эффективность: Давай лаконичные ответы."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        logger.info("Промпт создан, сборка цепочки...")
        chain = prompt | llm
        
        from collections import defaultdict
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