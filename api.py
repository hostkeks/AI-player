from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from typing import Optional
import logging
from minecraft_rag import MinecraftRAG
import time

# Инициализация RAG системы
rag = MinecraftRAG('minecraft_wiki_full.json')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minecraft NPC API", version="1.0.0")


# Модель для запроса
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9


# Модель для ответа
class ChatResponse(BaseModel):
    response: str
    status: str


# Глобальные переменные для модели
model = None
tokenizer = None


def load_model():
    """Загрузка модели при старте сервера"""
    global model, tokenizer
    try:
        model_name = "./gemma"
        logger.info("Загружаем модель Gemma...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Модель успешно загружена!")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске сервера"""
    if not load_model():
        raise RuntimeError("Не удалось загрузить модель")


@app.get("/")
async def root():
    return {"message": "Minecraft NPC API работает!", "status": "success"}


@app.get("/health")
async def health_check():
    """Проверка статуса API"""
    if model is None or tokenizer is None:
        return {"status": "error", "message": "Модель не загружена"}
    return {"status": "healthy", "message": "API работает нормально"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        start_time = time.time()
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")

        # 1. Ищем релевантную информацию
        relevant_info = rag.search(request.message, top_k=3)

        # ПРОВЕРЯЕМ КАЧЕСТВО ПОИСКА
        good_matches = [item for item in relevant_info if item['score'] > 0.4]

        if good_matches:
            # Есть хорошие совпадения - используем их
            context_parts = []
            for item in good_matches:
                content_preview = item['content'][:400] + "..." if len(item['content']) > 400 else item['content']
                context_parts.append(f"{item['title']}: {content_preview}")
            context = "\n\n".join(context_parts)
            knowledge_available = True
        else:
            # Нет хороших совпадений - используем общие знания
            context = "No specific information found in knowledge base about this exact question."
            knowledge_available = False

        # 2. ДЕТАЛЬНЫЙ АНГЛИЙСКИЙ ПРОМПТ
        if knowledge_available:
            system_instruction = f"""You are Arkady - the ultimate Minecraft companion and guide. You're a true friend to every player.

CONTEXT FROM MINECRAFT WIKI:
{context}

STRICT RULES:
- Answer in RUSSIAN language only
- Use the information above as your primary knowledge source
- Be conversational and friendly like you're talking to a friend
- Keep responses 2-4 sentences maximum
- If information is incomplete, say what you know based on the context
- Never make up stories or continue player's text
- Show genuine interest in helping the player

PERSONALITY:
- Warm, patient, knowledgeable with friendly humor
- Celebrate player achievements
- Offer encouragement when they struggle
- Mix practical advice with friendly banter"""
        else:
            system_instruction = """You are Arkady - the ultimate Minecraft companion and guide. You're a true friend to every player.

STRICT RULES:
- Answer in RUSSIAN language only  
- Be conversational and friendly like you're talking to a friend
- Keep responses 2-4 sentences maximum
- Provide general Minecraft advice based on your knowledge
- Never make up stories or continue player's text
- If unsure, say "Я не уверен в деталях, но в Майнкрафте обычно..." and give helpful suggestions
- Show genuine interest in helping the player

PERSONALITY:
- Warm, patient, knowledgeable with friendly humor
- Celebrate player achievements
- Offer encouragement when they struggle
- Mix practical advice with friendly banter

EXAMPLE STYLE:
"Привет! Отличный вопрос! В Майнкрафте обычно... [практический совет]. Кстати, я помню ты говорил про свой проект, как продвигается?" """

        prompt = f"{system_instruction}\n\nPlayer: {request.message}\n\nArkady:"

        # 3. Генерация ответа
        inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Извлекаем и очищаем ответ
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Arkady:" in full_response:
            response = full_response.split("Arkady:")[1].strip()
        else:
            response = full_response.replace(prompt, "").strip()

        response = response.split('\n')[0].split('Player:')[0].strip()

        end_time = time.time()
        resp_time = end_time - start_time
        print(f"Время ответа: {resp_time:.2f} секунд")

        return ChatResponse(response=response, status="success")

    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)