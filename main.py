from typing import List
from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from database import get_async_session, init_db
from contextlib import asynccontextmanager
from llm import get_conversation_chain
from models import Chat, Message
from schemas import MessageCreate, MessageResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/api/chat')
async def create_chat(message: MessageCreate, db: AsyncSession = Depends(get_async_session)):
    try:
        # Создаем новый чат
        new_chat = Chat()
        db.add(new_chat)
        await db.commit()
        await db.refresh(new_chat)

        # Сохраняем сообщение пользователя
        user_message = Message(content=message.content, role="user", chat_id=new_chat.id)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Возвращаем chat_id для перенаправления
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
    return templates.TemplateResponse('chat.html', {'request': request, 'chat_id': chat_id})

@app.post('/api/chat/{chat_id}/message', response_model=MessageResponse)
async def add_message(chat_id: int, message: MessageCreate, db: AsyncSession = Depends(get_async_session)):
    try:
        result = await db.execute(select(Chat).filter(Chat.id == chat_id))
        chat = result.scalars().first()
        if not chat:
            raise HTTPException(status_code=404, detail="Чат не найден")

        # Сохраняем сообщение пользователя
        user_message = Message(content=message.content, role="user", chat_id=chat_id)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Получаем ответ от LLM
        conversation = get_conversation_chain()
        response = await conversation.ainvoke(
            {"input": message.content},
            config={"configurable": {"session_id": str(chat_id)}}
        )

        # Сохраняем ответ LLM
        llm_message = Message(content=response, role="assistant", chat_id=chat_id)
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
        raise HTTPException(status_code=500, detail=f"Ошибка добавления сообщения: {str(e)}")

@app.get('/api/chat/{chat_id}/messages', response_model=List[MessageResponse])
async def get_chat_messages(chat_id: int, db: AsyncSession = Depends(get_async_session)):
    result = await db.execute(select(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp))
    messages = result.scalars().all()
    return messages

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True)