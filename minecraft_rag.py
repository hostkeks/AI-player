# minecraft_rag.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MinecraftRAG:
    def __init__(self, json_path):
        # Загружаем твой JSON файл
        with open(json_path, 'r', encoding='utf-8') as f:
            self.wiki_data = json.load(f)

        print(f"✅ Загружено {len(self.wiki_data)} страниц из вики")

        # Модель для поиска
        print("🔄 Загружаем модель для поиска...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Подготавливаем данные для поиска
        print("🔍 Создаем поисковый индекс...")
        self.prepare_search_index()
        print("✅ Поисковый индекс готов!")

    def prepare_search_index(self):
        """Создает поисковый индекс"""
        self.texts = []
        self.metadata = []

        for item in self.wiki_data:
            text = f"{item['title']}: {item['content']}"
            self.texts.append(text)
            self.metadata.append(item)

        # Создаем эмбеддинги
        self.embeddings = self.encoder.encode(self.texts)

    def search(self, query, top_k=3):
        """Ищет релевантную информацию"""
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'title': self.metadata[idx]['title'],
                'content': self.metadata[idx]['content'],
                'score': similarities[idx]
            })
        return results


# Инициализация и ТЕСТ - ДОБАВЬ ЭТУ ЧАСТЬ!
if __name__ == "__main__":
    print("🎮 Запускаем Minecraft RAG систему...")

    # Убедись что путь к файлу правильный!
    rag = MinecraftRAG('minecraft_wiki_ru_categories.json')  # путь к твоему файлу

    # Тестируем поиск
    test_queries = [
        "creeper explosion",
        "how to make diamond pickaxe",
        "redstone circuit basics",
        "ender dragon fight"
    ]

    for query in test_queries:
        print(f"\n🔍 Поиск: '{query}'")
        results = rag.search(query, top_k=2)

        for i, r in enumerate(results):
            print(f"  {i + 1}. {r['title']} (score: {r['score']:.3f})")
            print(f"     {r['content'][:150]}...")
            print()