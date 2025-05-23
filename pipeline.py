import os
import pandas as pd
from datetime import datetime, date
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import json
from fetch_tags import FetchTags

# --- Пример использования ---
if __name__ == "__main__":
    # Укажи пути к твоим Excel файлам
    products_file_path = "data/" + "1. Продукты.xlsx"  # Замени на реальное имя файла
    outgoing_ops_file_path = "data/" + "2. Исходящие операции.xlsx" # Замени
    incoming_ops_file_path = "data/" + "3.Входящие операции.xlsx" # Замени
    # dynamics_file_path = "data/" + "4. Динамика остатков.xlsx" # Пока не используется для этих тегов
    contracts_file_path = "data/" + "5. Договора.xlsx" # Замени
    
    fecth_tags = FetchTags()
    client_tagged_data = fecth_tags.process_excel_files(
        products_file_path,
        outgoing_ops_file_path,
        incoming_ops_file_path,
        contracts_file_path
    )

    if client_tagged_data:
        print("\n\n--- Итоговые результаты тегирования ---")
        for client_info in client_tagged_data:
            print(f"Клиент: {client_info['CLN_NAME']} (CLI_ID: {client_info['CLI_ID']})")
            print(f"Теги: {', '.join(client_info['TAGS']) if client_info['TAGS'] else 'Нет тегов'}")
            print("-" * 30)
        
        # Опционально: сохранение результатов в новый Excel или CSV
        df_results = pd.DataFrame(client_tagged_data)
        try:
            df_results.to_csv("client_tags_results_csv.csv", index=False)
            print("\nРезультаты сохранены в client_tags_results_csv.csv")
        except Exception as e:
            print(f"Не удалось сохранить результаты в CSV: {e}")