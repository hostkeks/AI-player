import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_gemma_model():
    """Загрузка модели Gemma-2-2b"""
    model_name = "./gemma"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Добавляем padding token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def clean_response(response, prompt):
    """Очистка ответа от лишних частей"""
    # Удаляем промпт из ответа
    if prompt in response:
        response = response.replace(prompt, "")

    # Удаляем лишние теги и код
    response = response.split("<end_of_turn>")[0]
    response = response.split("```")[0]
    response = response.strip()

    return response


def main():
    """Основная функция чата"""
    print("Загружаем модель Gemma-2-2b...")
    model, tokenizer = load_gemma_model()
    print("Модель загружена!")

    print("\n🤖 Gemma Chat готов к общению!")
    print("Напишите 'выход' чтобы завершить разговор\n")

    while True:
        user_input = input("Вы: ")

        if user_input.lower() in ['выход', 'exit', 'quit', 'bye']:
            print("До свидания! 👋")
            break

        if not user_input.strip():
            continue

        # Форматируем промпт для чата
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

        # Токенизируем входные данные
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(model.device)

        print("Gemma думает...")
        try:
            # Генерируем ответ
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Декодируем ответ
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Очищаем ответ
            response = clean_response(full_response, prompt)

            print(f"Gemma: {response}\n")

        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()