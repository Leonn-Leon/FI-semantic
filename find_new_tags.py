import pandas as pd
from openai import OpenAI
import os

# --- Конфигурация OpenAI ---
try:
    client = OpenAI() # Убедись, что OPENAI_API_KEY установлен в окружении
except Exception as e:
    print(f"Не удалось инициализировать OpenAI client: {e}")
    exit()

# --- Функция запроса к LLM (остается такой же, как в предыдущем ответе) ---
def suggest_additional_single_tags_from_transactions(
    sample_transaction_descriptions,
    existing_pydantic_tags_description,
    client_name_for_context="Не указан",
    model="gpt-4.1-2025-04-14"
):
    if not sample_transaction_descriptions:
        return "Не предоставлены примеры описаний транзакций для анализа."
    descriptions_text = "\n".join([f"- {desc}" for desc in sample_transaction_descriptions])
    prompt = f"""
    Я анализирую банковские транзакции компании (МСБ) '{client_name_for_context}' для ее семантической сегментации.
    С помощью LLM я уже извлекаю информацию о следующих аспектах (на основе Pydantic моделей):
    {existing_pydantic_tags_description}

    Вот примеры **последних исходящих транзакций (поле ENTRY_DESCR)** для этой компании:
    ---
    {descriptions_text}
    ---

    Пожалуйста, внимательно изучи эти примеры описаний транзакций.
    Предложи список **новых, дополнительных ОДИНОЧНЫХ тегов**, которые можно было бы извлекать из подобных текстовых описаний.
    Эти теги должны характеризовать:
    1.  Специфические виды операционных расходов (например, 'расходы_на_аренду_оборудования', 'оплата_маркетинговых_услуг_Х', 'покупка_лицензий_ПО', 'командировочные_расходы').
    2.  Особенности финансовых операций (например, 'регулярный_автоплатеж_за_сервис_Y', 'перевод_между_своими_счетами', 'получение_возврата_средств').
    3.  Взаимодействия с конкретными типами контрагентов или сервисов (например, 'частые_платежи_логистической_компании_Z', 'оплата_облачных_сервисов_AWS_Azure').
    4.  Признаки определенных событий или активностей (например, 'участие_в_выставке_конференции', 'крупная_разовая_закупка_оборудования').

    Для каждого предложенного нового тега:
    - Дай ему короткое, понятное имя в формате 'название_тега_маленькими_буквами_через_подчеркивание'.
    - Кратко поясни его значение.
    - Укажи, на какие ключевые слова, фразы или паттерны в *предоставленных транзакциях* он мог бы опираться.

    Представь свой ответ в виде списка, где каждый элемент:
    Тег: [имя_тега]
    Значение: [пояснение значения тега]
    Основание (ключевые слова/паттерны из транзакций): [примеры]

    Важно: Предлагай именно **одиночные теги**, а не новые сложные категории.
    Цель - найти специфичные маркеры поведения или расходов.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты - опытный бизнес-аналитик, специализирующийся на выявлении специфических поведенческих тегов из текстовых описаний финансовых операций МСБ."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при обращении к OpenAI для клиента '{client_name_for_context}': {e}")
        return None

# --- Функция для чтения уже обработанных клиентов ---
def get_processed_clients(processed_file_path):
    """Читает файл и возвращает множество CLI_ID или имен уже обработанных клиентов."""
    processed_identifiers = set()
    if not os.path.exists(processed_file_path):
        return processed_identifiers
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Ищем CLI_ID или CLN_NAME в файле.
            # Можно улучшить парсинг, если формат файла mb_new_tags.md известен
            # Здесь простой поиск по строке.
            # Для большей надежности лучше сохранять CLI_ID в файл и искать по ним.
            # Если мы ищем по CLN_NAME, нужно быть осторожным с совпадениями.
            # Пока будем предполагать, что если имя клиента есть в файле, он обработан.
            # Для примера, будем извлекать то, что похоже на "Клиент: ИМЯ (CLI_ID: ID)"
            import re
            matches_name = re.findall(r"Анализируем клиента: (.*?) \(CLI_ID:", content)
            matches_id = re.findall(r"\(CLI_ID: ([\d\w\s,-]+)\)", content) # Более общий паттерн для ID
            
            for name in matches_name:
                processed_identifiers.add(name.strip())
            for client_id in matches_id:
                processed_identifiers.add(client_id.strip())
                
    except Exception as e:
        print(f"Ошибка при чтении файла обработанных клиентов '{processed_file_path}': {e}")
    return processed_identifiers

# --- Основная функция для обработки данных из Excel ---
def analyze_clients_for_additional_single_tags(
    products_file, 
    outgoing_ops_file, 
    processed_tags_file, # Путь к файлу mb_new_tags.md
    num_clients_to_process=None, 
    num_transactions_per_client=30
    ):
    try:
        df_products = pd.read_excel(products_file)
        df_outgoing_ops = pd.read_excel(outgoing_ops_file)
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден. {e}")
        return
    except Exception as e:
        print(f"Ошибка при чтении Excel файла: {e}")
        return

    df_products['CLI_ID'] = df_products['CLI_ID'].astype(str).str.replace(r'\.00$', '', regex=True)
    df_outgoing_ops['CLI_ID'] = df_outgoing_ops['CLI_ID'].astype(str).str.replace(r'\.00$', '', regex=True)

    try:
        df_outgoing_ops['DT_ENTRY_Parsed'] = pd.to_datetime(df_outgoing_ops['DT_ENTRY'], dayfirst=True, errors='coerce')
    except Exception:
        try:
            df_outgoing_ops['DT_ENTRY_Parsed'] = pd.to_datetime(df_outgoing_ops['DT_ENTRY'], errors='coerce')
        except Exception as e:
            print(f"Не удалось преобразовать DT_ENTRY в дату: {e}. Пропускаем сортировку по дате.")
            df_outgoing_ops['DT_ENTRY_Parsed'] = None

    # 1. Подсчет количества транзакций для каждого клиента
    client_transaction_counts = df_outgoing_ops.groupby('CLI_ID').size().rename('transaction_count')
    
    # Объединяем с df_products, чтобы получить имена и отсортировать
    df_products_with_counts = df_products.merge(client_transaction_counts, on='CLI_ID', how='left')
    df_products_with_counts['transaction_count'] = df_products_with_counts['transaction_count'].fillna(0).astype(int)
    
    # Сортируем клиентов: сначала те, у кого больше транзакций
    sorted_clients_df = df_products_with_counts.sort_values(by='transaction_count', ascending=False)

    # 2. Получаем список уже обработанных клиентов
    processed_identifiers = get_processed_clients(processed_tags_file)
    print(f"Найдены следующие уже обработанные идентификаторы: {processed_identifiers if processed_identifiers else 'Нет'}")


    existing_pydantic_tags_info = """
    - Основные типы платежей: payments_to_suppliers (true/false), payments_salary_related (true/false), payments_tax (true/false).
    - Операции с наличными: cash_activity_level ('high' или 'low').
    - Признаки ВЭД: has_ved_signs (true/false).
    """
    
    clients_to_iterate_df = sorted_clients_df
    if num_clients_to_process is not None:
        clients_to_iterate_df = sorted_clients_df.head(num_clients_to_process)

    processed_count_in_session = 0
    for index, client_row in clients_to_iterate_df.iterrows():
        cli_id = client_row['CLI_ID']
        cln_name = client_row.get('CLN_NAME', f"Клиент ID {cli_id}")
        
        # Проверяем, обработан ли клиент (по ID или по имени)
        if cli_id in processed_identifiers or cln_name in processed_identifiers:
            print(f"Клиент {cln_name} (CLI_ID: {cli_id}) уже обработан. Пропускаем.")
            continue
        
        print(f"\n\n{'='*25}\nАнализируем клиента: {cln_name} (CLI_ID: {cli_id}), транзакций: {client_row['transaction_count']}\n{'='*25}")

        client_ops_df = df_outgoing_ops[df_outgoing_ops['CLI_ID'] == cli_id]

        if client_ops_df.empty:
            print("Исходящие транзакции для этого клиента не найдены.")
            # Записываем в файл, что клиент был рассмотрен, но без транзакций (чтобы не проверять снова)
            with open(processed_tags_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'='*25}\nАнализируем клиента: {cln_name} (CLI_ID: {cli_id})\n{'='*25}\n")
                f.write("Исходящие транзакции для этого клиента не найдены.\n")
            continue
            
        if 'DT_ENTRY_Parsed' in client_ops_df.columns and client_ops_df['DT_ENTRY_Parsed'].notna().any():
            sorted_ops = client_ops_df.sort_values(by='DT_ENTRY_Parsed', ascending=False)
        else:
            print("Предупреждение: Не удалось отсортировать транзакции по дате. Используется исходный порядок.")
            sorted_ops = client_ops_df 

        last_n_ops = sorted_ops.head(num_transactions_per_client)
        transaction_descriptions = last_n_ops['ENTRY_DESCR'].dropna().astype(str).tolist()

        if not transaction_descriptions:
            print("Описания транзакций для анализа не найдены.")
            with open(processed_tags_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'='*25}\nАнализируем клиента: {cln_name} (CLI_ID: {cli_id})\n{'='*25}\n")
                f.write("Описания транзакций для анализа не найдены.\n")
            continue
            
        print(f"Передаем {len(transaction_descriptions)} последних исходящих транзакций в LLM...")
        
        suggestions = suggest_additional_single_tags_from_transactions(
            transaction_descriptions, 
            existing_pydantic_tags_info,
            client_name_for_context=cln_name
        )

        # Записываем результат (или сообщение об ошибке) в файл mb_new_tags.md
        # Запись происходит сразу после получения ответа от LLM для данного клиента
        with open(processed_tags_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*25}\nАнализируем клиента: {cln_name} (CLI_ID: {cli_id}), транзакций: {client_row['transaction_count']}\n{'='*25}\n")
            if suggestions:
                f.write(f"\n--- Предложения по ДОПОЛНИТЕЛЬНЫМ ОДИНОЧНЫМ тегам для клиента {cln_name} ---\n")
                f.write(suggestions)
                f.write("\n--- Конец предложений ---\n")
                print(f"\n--- Предложения по ДОПОЛНИТЕЛЬНЫМ ОДИНОЧНЫМ тегам для клиента {cln_name} ---")
                print(suggestions)
                print("--- Конец предложений ---")
            else:
                f.write("Не удалось получить предложения по дополнительным тегам от LLM для этого клиента.\n")
                print("Не удалось получить предложения по дополнительным тегам от LLM для этого клиента.")
        
        processed_count_in_session += 1
        # Если num_clients_to_process был задан и мы обработали нужное количество *новых* клиентов
        if num_clients_to_process is not None and processed_count_in_session >= num_clients_to_process:
            print(f"\nОбработано {processed_count_in_session} новых клиентов, как было указано. Завершение.")
            break


if __name__ == "__main__":
    products_file_path = "data/1. Продукты.xlsx"
    outgoing_ops_file_path = "data/2. Исходящие операции.xlsx"
    processed_tags_md_file = "mb_new_tags.md" # Имя файла для записи результатов и проверки

    if not os.path.exists(products_file_path):
        print(f"Ошибка: Файл продуктов не найден - {products_file_path}")
        exit()
    if not os.path.exists(outgoing_ops_file_path):
        print(f"Ошибка: Файл исходящих операций не найден - {outgoing_ops_file_path}")
        exit()

    NUMBER_OF_NEW_CLIENTS_TO_PROCESS = 10
    TRANSACTIONS_PER_CLIENT = 30

    analyze_clients_for_additional_single_tags(
        products_file_path, 
        outgoing_ops_file_path,
        processed_tags_md_file, # Передаем путь к файлу
        num_clients_to_process=NUMBER_OF_NEW_CLIENTS_TO_PROCESS,
        num_transactions_per_client=TRANSACTIONS_PER_CLIENT
    )