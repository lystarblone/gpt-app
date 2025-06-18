from typing import List
from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import get_async_session, init_db
from contextlib import asynccontextmanager
from llm import get_conversation_chain
from models import Chat, Message
from schemas import MessageCreate, MessageResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

# Настройка статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'show_auth_buttons': 'true',
            'show_search_block': 'true'
        }
    )

@app.post('/api/chat')
async def create_chat(message: MessageCreate, db: AsyncSession = Depends(get_async_session)):
    try:
        new_chat = Chat()
        db.add(new_chat)
        await db.commit()
        await db.refresh(new_chat)

        user_message = Message(content=message.content, role="user", chat_id=new_chat.id)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        return {"chat_id": new_chat.id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка создания чата: {str(e)}")

@app.get('/chat/{chat_id}', response_class=HTMLResponse)
async def chat_page(request: Request, chat_id: int, db: AsyncSession = Depends(get_async_session)):
    result = await db.execute(select(Chat).filter(Chat.id == chat_id))
    chat = result.scalars().first()
    if not chat:
        return templates.TemplateResponse('error.html', {'request': request, 'message': 'Чат не найден'}, status_code=404)
    return templates.TemplateResponse(
        'chat.html',
        {
            'request': request,
            'chat_id': chat_id,
            'show_auth_buttons': 'true',
            'show_search_block': 'true'
        }
    )

@app.post('/api/chat/{chat_id}/message', response_model=MessageResponse)
async def add_message(chat_id: int, message: MessageCreate, db: AsyncSession = Depends(get_async_session)):
    try:
        logger.info(f"Получен запрос на добавление сообщения в чат {chat_id}: {message.content}")
        result = await db.execute(select(Chat).filter(Chat.id == chat_id))
        chat = result.scalars().first()
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        user_message = Message(content=message.content, role="user", chat_id=chat_id)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        conversation = get_conversation_chain()
        logger.info(f"Цепочка разговора создана, вызов ainvoke с: {message.content}")
        response = await conversation.ainvoke(
            {"input": message.content},
            config={"configurable": {"session_id": str(chat_id)}}
        )
        logger.info(f"Сырой ответ от модели: {response}")
        response_text = str(response) if hasattr(response, '__str__') else "Нет ответа от модели"
        logger.info(f"Обработанный ответ: {response_text}")

        llm_message = Message(content=response_text, role="assistant", chat_id=chat_id)
        db.add(llm_message)
        await db.commit()
        await db.refresh(llm_message)

        return MessageResponse(
            content=llm_message.content,
            role=llm_message.role,
            timestamp=llm_message.timestamp
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка добавления сообщения: {str(e)}")

@app.get('/api/chat/{chat_id}/messages', response_model=List[MessageResponse])
async def get_chat_messages(chat_id: int, db: AsyncSession = Depends(get_async_session)):
    result = await db.execute(select(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp))
    messages = result.scalars().all()
    return messages

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True)