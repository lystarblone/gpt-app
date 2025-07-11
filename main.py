from typing import List, Optional
from fastapi import Depends, FastAPI, Request, HTTPException, BackgroundTasks, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import get_async_session, init_db
from contextlib import asynccontextmanager
from llm import get_conversation_chain, clean_response
from models import Chat, Message, User, RefreshToken
from schemas import MessageCreate, MessageResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from auth import hash_password, verify_password, create_access_token, create_refresh_token, verify_token
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

async def get_current_user(request: Request, db: AsyncSession = Depends(get_async_session)) -> Optional[dict]:
    token = request.cookies.get("access_token")
    if not token:
        logger.debug("Токен отсутствует в cookie")
        return None
    try:
        payload = verify_token(token)
        logger.debug(f"Токен проверен, пользователь: {payload.get('sub')}")
        return payload
    except HTTPException:
        logger.warning("Недействительный токен")
        return None

@app.get('/', response_class=HTMLResponse)
async def index_page(request: Request, current_user: Optional[dict] = Depends(get_current_user)):
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'is_authenticated': bool(current_user),
            'show_search_block': 'true'
        }
    )

@app.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        'login.html',
        {'request': request}
    )

@app.post('/login')
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember: bool = Form(False),
    db: AsyncSession = Depends(get_async_session)
):
    logger.info(f"Попытка входа: {email}, remember: {remember}")
    
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    
    if not user or not verify_password(password, user.hashed_password):
        logger.warning(f"Неудачная попытка входа: {email}")
        return templates.TemplateResponse(
            'login.html',
            {'request': request, 'error': 'Неверный email или пароль'},
            status_code=401
        )
    
    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token()
    
    # Сохраняем refresh-токен в базе данных
    refresh_token_expires = datetime.utcnow() + timedelta(days=7)
    db_refresh_token = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=refresh_token_expires
    )
    db.add(db_refresh_token)
    await db.commit()
    
    response = RedirectResponse(url='/', status_code=303)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,  # Для разработки, в продакшене True
        samesite="strict",
        max_age=15 * 60 if not remember else 7 * 24 * 60 * 60
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False,  # Для разработки, в продакшене True
        samesite="strict",
        max_age=7 * 24 * 60 * 60
    )
    logger.info(f"Успешный вход: {email}")
    return response

@app.get('/register', response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        'register.html',
        {'request': request}
    )

@app.post('/register')
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_async_session)
):
    logger.info(f"Попытка регистрации: {email}")
    
    result = await db.execute(select(User).filter(User.email == email))
    existing_user = result.scalars().first()
    
    if existing_user:
        logger.warning(f"Пользователь уже существует: {email}")
        return templates.TemplateResponse(
            'register.html',
            {'request': request, 'error': 'Email уже зарегистрирован'},
            status_code=400
        )
    
    hashed_password = hash_password(password)
    new_user = User(
        email=email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        logger.info(f"Пользователь зарегистрирован: {email}")
        
        access_token = create_access_token(data={"sub": new_user.email})
        refresh_token = create_refresh_token()
        
        # Сохраняем refresh-токен в базе данных
        refresh_token_expires = datetime.utcnow() + timedelta(days=7)
        db_refresh_token = RefreshToken(
            token=refresh_token,
            user_id=new_user.id,
            expires_at=refresh_token_expires
        )
        db.add(db_refresh_token)
        await db.commit()
        
        response = RedirectResponse(url='/', status_code=303)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False,  # Для разработки, в продакшене True
            samesite="strict",
            max_age=7 * 24 * 60 * 60
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=False,  # Для разработки, в продакшене True
            samesite="strict",
            max_age=7 * 24 * 60 * 60
        )
        return response
    except Exception as e:
        await db.rollback()
        logger.error(f"Ошибка при регистрации: {str(e)}")
        return templates.TemplateResponse(
            'register.html',
            {'request': request, 'error': 'Ошибка при регистрации. Попробуйте снова.'},
            status_code=500
        )

@app.post('/refresh')
async def refresh_token(request: Request, db: AsyncSession = Depends(get_async_session)):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh-токен отсутствует")
    
    result = await db.execute(
        select(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.expires_at > datetime.utcnow()
        )
    )
    db_refresh_token = result.scalars().first()
    
    if not db_refresh_token:
        raise HTTPException(status_code=401, detail="Недействительный или истёкший refresh-токен")
    
    result = await db.execute(select(User).filter(User.id == db_refresh_token.user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    
    access_token = create_access_token(data={"sub": user.email})
    response = Response()
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,  # Для разработки, в продакшене True
        samesite="strict",
        max_age=15 * 60
    )
    return {"access_token": access_token}

@app.get('/logout', response_class=RedirectResponse)
async def logout(request: Request, db: AsyncSession = Depends(get_async_session)):
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        result = await db.execute(select(RefreshToken).filter(RefreshToken.token == refresh_token))
        db_refresh_token = result.scalars().first()
        if db_refresh_token:
            await db.delete(db_refresh_token)
            await db.commit()
    response = RedirectResponse(url='/', status_code=303)
    response.delete_cookie(key="access_token")
    response.delete_cookie(key="refresh_token")
    logger.info("Пользователь вышел из системы")
    return response

@app.get('/chat/{chat_id}', response_class=HTMLResponse)
async def chat_page(request: Request, chat_id: int, db: AsyncSession = Depends(get_async_session), current_user: Optional[dict] = Depends(get_current_user)):
    result = await db.execute(select(Chat).filter(Chat.id == chat_id))
    chat = result.scalars().first()
    if not chat:
        return templates.TemplateResponse('error.html', {'request': request, 'message': 'Чат не найден'}, status_code=404)
    return templates.TemplateResponse(
        'chat.html',
        {
            'request': request,
            'chat_id': chat_id,
            'is_authenticated': bool(current_user),
            'show_search_block': 'true'
        }
    )

@app.post('/api/chat')
async def create_chat(
    message: MessageCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: Optional[dict] = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    try:
        logger.info("Создание нового чата")
        new_chat = Chat()
        db.add(new_chat)
        await db.commit()
        await db.refresh(new_chat)
        logger.info(f"Чат создан, ID: {new_chat.id}")

        user_message = Message(content=message.content, role="user", chat_id=new_chat.id)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)
        logger.info(f"Сообщение пользователя сохранено: {message.content}")

        async def process_model_response(chat_id: int, content: str):
            try:
                conversation = get_conversation_chain()
                logger.info(f"Вызов модели для чата {chat_id} с сообщением: {content}")
                response = await conversation.ainvoke(
                    {"input": content},
                    config={"configurable": {"session_id": str(chat_id)}}
                )
                logger.info(f"Сырой ответ модели: {response}")
                response_text = clean_response(response)
                logger.info(f"Обработанный ответ: {response_text}")

                llm_message = Message(content=response_text, role="assistant", chat_id=chat_id)
                db.add(llm_message)
                await db.commit()
                await db.refresh(llm_message)
                logger.info("Ответ модели сохранён")
            except Exception as e:
                logger.error(f"Ошибка при вызове модели: {str(e)}")
                error_message = Message(
                    content=f"Ошибка обработки сообщения моделью: {str(e)}",
                    role="assistant",
                    chat_id=chat_id
                )
                db.add(error_message)
                await db.commit()

        background_tasks.add_task(process_model_response, new_chat.id, message.content)

        return {"chat_id": new_chat.id}
    except Exception as e:
        await db.rollback()
        logger.error(f"Ошибка создания чата: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка создания чата: {str(e)}")

@app.post('/api/chat/{chat_id}/message', response_model=MessageResponse)
async def add_message(chat_id: int, message: MessageCreate, db: AsyncSession = Depends(get_async_session), current_user: Optional[dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
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
        response_text = clean_response(response)
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
async def get_chat_messages(chat_id: int, db: AsyncSession = Depends(get_async_session), current_user: Optional[dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    result = await db.execute(select(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp))
    messages = result.scalars().all()
    return messages

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True)