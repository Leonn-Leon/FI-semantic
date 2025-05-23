import os
import pandas as pd
from datetime import datetime, date
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import json
import yaml
from loguru import logger

class PaymentTypes(BaseModel):
    payments_to_suppliers: bool = Field(default=False, description="True, если есть платежи поставщикам (оплата по счету, за товары/услуги, за материалы)")
    payments_salary_related: bool = Field(default=False, description="True, если есть выплаты, похожие на зарплату (перечисление зп, аванс)")
    payments_tax: bool = Field(default=False, description="True, если есть налоговые платежи (оплата налога, пени ФНС, взнос ПФР)")

class CashOperations(BaseModel):
    # TODO: на будущее задать что считать high/low
    cash_activity_level: Literal['high', 'low'] = Field(description='"high", если есть частые/крупные операции с наличными; "low", если преобладают безналичные расчеты.')

class VedSigns(BaseModel):
    has_ved_signs: bool = Field(default=False, description="True, если найдены признаки ВЭД, иначе false.")

class FetchTags:
    def __init__(self, config_path="config.yaml"): # Изменяем путь по умолчанию на .yaml
        load_dotenv(override=True)
        self.client = OpenAI()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) # Используем yaml.safe_load
        except Exception as e: # Общий обработчик на случай других проблем
            logger.error(f"Неожиданная ошибка при загрузке конфигурации '{config_path}': {e}. Используются значения по умолчанию.")

    def get_llm_structured_output_with_pydantic(
        self,
        tags_context: str, # Измененный параметр для контекста задачи
        pydantic_model: type[BaseModel],    # Тип Pydantic модели, которую ожидаем
    ) -> Optional[BaseModel]:
        """
        Отправляет промпт в OpenAI и ожидает структурированный ответ,
        соответствующий предоставленной Pydantic модели, используя 'tools'.
        Использует шаблоны промптов из конфигурационного файла (self.config).
        """
        try:
            tool_name = pydantic_model.__name__ # Используем имя класса модели как имя функции

            # Получаем значения из конфига или используем переданные переопределения
            # self.config предполагается загруженным в __init__
            system_prompt_content = self.config.get("default_system_prompt")
            model_to_use = self.config.get("openai_model",)

            # Формируем пользовательский промпт из шаблона в конфиге
            user_prompt_template = self.config.get("user_prompt_template")
            user_prompt_content_final = user_prompt_template.format(
                tags_context=tags_context,
                tool_name=tool_name
            )
            completion = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content_final}
                ],
                tools=[{"type": "function", "function": openai.pydantic_function_tool(pydantic_model, name=tool_name)}],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=0.1,
            )

            message = completion.choices[0].message
            
            if message.tool_calls and message.tool_calls[0].function.name == tool_name:
                arguments_json_str = message.tool_calls[0].function.arguments
                try:
                    parsed_args_dict = json.loads(arguments_json_str)
                    return pydantic_model(**parsed_args_dict)
                except json.JSONDecodeError as e_json_args:
                    print(f"Ошибка декодирования JSON аргументов функции от OpenAI: {e_json_args}")
                    print(f"Полученные аргументы (строка): {arguments_json_str}")
                    return None
                except Exception as e_pydantic: # Например, pydantic.ValidationError
                    print(f"Ошибка при создании Pydantic модели из аргументов: {e_pydantic}")
                    print(f"Распарсенные аргументы (словарь), вызвавшие ошибку Pydantic: {parsed_args_dict if 'parsed_args_dict' in locals() else 'Не удалось распарсить JSON аргументы'}")
                    return None
            else:
                error_message = f"LLM не вызвала ожидаемый инструмент '{tool_name}'."
                if message.content:
                    error_message += f" Ответ LLM: {message.content}"
                if message.tool_calls:
                    error_message += f" Вызванные инструменты: {[tc.function.name for tc in message.tool_calls]}"
                print(error_message)
                return None

        except openai.APIError as e: # Более специфичная обработка ошибок API OpenAI
            print(f"Ошибка OpenAI API: {e}")
            return None
        except Exception as e: # Любые другие непредвиденные ошибки
            print(f"Непредвиденная ошибка при обращении к OpenAI (Pydantic tools): {e}")
            return None

    def parse_boolean_flag(self, value):
        """Преобразует значения флагов (1.00, 0.00, "да", "нет") в булевы."""
        if pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ['да', 'yes', 'true', '1', '1.0']
        return False

    def parse_date_value(self, value):
        """Преобразует значение в объект date, если возможно."""
        if pd.isna(value):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                # Попытка распарсить популярные форматы
                return pd.to_datetime(value, errors='coerce').date()
            except Exception:
                pass # Если не удалось, вернем None ниже
        return None


    def get_company_size_tags(self, staff_group_value):
        tags = []
        if pd.notna(staff_group_value):
            staff_group_value = str(staff_group_value).lower()
            # Логика может потребовать уточнения на основе реальных значений в STAFF_GROUP
            if "1-24" in staff_group_value or "до 15" in staff_group_value or "микро" in staff_group_value:
                tags.append("company_size_micro")
            elif "25-100" in staff_group_value or "16-100" in staff_group_value or "малое" in staff_group_value: # Примерные значения
                tags.append("company_size_small")
            elif "101-250" in staff_group_value or "среднее" in staff_group_value:
                tags.append("company_size_medium")
        return tags

    def get_company_age_tags(self, dt_bank_open_value):
        tags = []
        dt_bank_open_date = self.parse_date_value(dt_bank_open_value)
        if dt_bank_open_date:
            today = date.today()
            age_years = (today - dt_bank_open_date).days / 365.25
            if age_years < 3:
                tags.append("company_age_new")
            else:
                tags.append("company_age_established")
        return tags

    def get_payment_type_tags_llm(self, transactions_descriptions):
        tags = []
        if not transactions_descriptions:
            return tags
        
        sample_descriptions = "\n".join(transactions_descriptions[:20])
        
        payments_context = self.config["payments_context"].format(sample_descriptions=sample_descriptions)
        
        # Pydantic модель PaymentTypes уже описана выше
        structured_response: Optional[PaymentTypes] = self.get_llm_structured_output_with_pydantic(
            tags_context=payments_context,
            pydantic_model=PaymentTypes
        )
        
        if structured_response:
            if structured_response.payments_to_suppliers:
                tags.append("payments_to_suppliers")
            if structured_response.payments_salary_related:
                tags.append("payments_salary_related")
            if structured_response.payments_tax:
                tags.append("payments_tax")
        return tags

    def get_cash_operations_tags_llm(self, transactions_descriptions, kassa_comis_total):
        tags = []
        has_cash_indicators_from_data = bool(kassa_comis_total and kassa_comis_total > 0)

        if not transactions_descriptions and not has_cash_indicators_from_data:
            tags.append("cash_operations_low")
            return tags

        sample_descriptions = "\n".join(transactions_descriptions[:10]) if transactions_descriptions else "Нет описаний транзакций для анализа."
        
        additional_cash_info_str = f"Дополнительная информация: {'есть данные о комиссиях по кассовым операциям на общую сумму ' + str(kassa_comis_total) if has_cash_indicators_from_data else ""}."

        # Формируем tags_context_cash:
        tags_context_cash = self.config["tags_context_cash"].format(
            sample_descriptions=sample_descriptions,
            additional_cash_info_str=additional_cash_info_str
        )
        
        structured_response: Optional[CashOperations] = self.get_llm_structured_output_with_pydantic(
            user_prompt_content=tags_context_cash,
            pydantic_model=CashOperations
        )
        
        if structured_response:
            if structured_response.cash_activity_level == "high":
                tags.append("cash_operations_high")
            elif structured_response.cash_activity_level == "low":
                tags.append("cash_operations_low")
            elif not tags and has_cash_indicators_from_data: # Дополнительная логика, если LLM не определила четко
                tags.append("cash_operations_high")
        
        if not tags:
            tags.append("cash_operations_low" if not has_cash_indicators_from_data else "cash_operations_high")

        return tags

    def get_geo_tags(self, city_value):
        tags = []
        if pd.notna(city_value):
            city_value_str = str(city_value).lower()
            if "москва" in city_value_str:
                tags.append("geo_moscow_smb")
            else:
                tags.append("geo_region_smb")
        return tags

    # Вспомогательная функция, если она нужна для is_ved_flag_value
    def parse_boolean_flag(self, value):
        if pd.isna(value) if 'pd' in globals() else value is None: # Проверка на pandas, если используется
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ['да', 'yes', 'true', '1', '1.0']
        return False

    def get_ved_tags(self, is_ved_flag_value, transactions_descriptions=None):
        tags = []
        is_ved_explicitly_true = self.parse_boolean_flag(is_ved_flag_value)

        if is_ved_explicitly_true:
            tags.append("ved_active")
            return tags

        if transactions_descriptions:
            sample_descriptions = "\n".join(transactions_descriptions[:10])
            
            tags_context_ved = self.config["tags_context_ved"].format(
                sample_descriptions=sample_descriptions)
            # TODO: Добавить возможность писать reason почему этот тэг
            # Общий промпт сделать, вынести в конфиг, задать место в промпте для контекста
            # Передавать не только транзакции, но и профиль клиента
            structured_response: Optional[VedSigns] = self.get_llm_structured_output_with_pydantic(
                user_prompt_content=tags_context_ved,
                pydantic_model=VedSigns
            )
            
            if structured_response and structured_response.has_ved_signs:
                tags.append("ved_active")
            else:
                tags.append("ved_absent")
        else:
            tags.append("ved_absent")
            
        return tags

    def get_acquiring_tags(self, is_acq_flag_value):
        tags = []
        if self.parse_boolean_flag(is_acq_flag_value):
            tags.append("acquiring_user_active")
        else:
            tags.append("acquiring_absent_or_low")
        return tags

    def get_debt_load_tags(self, is_credit_flag_value, client_contracts_df=None):
        tags = []
        has_credit = self.parse_boolean_flag(is_credit_flag_value)
        
        if not has_credit and client_contracts_df is not None and not client_contracts_df.empty:
            # Проверяем типы контрактов из таблицы "5. Договора"
            if client_contracts_df['CON_TYPE'].astype(str).str.lower().str.contains('кредит|credit|loan').any():
                has_credit = True
                
        if has_credit:
            tags.append("debt_load_present")
        else:
            tags.append("debt_load_absent")
        return tags

    def get_salary_project_tag(self, is_sal_flag_value):
        tags = []
        if self.parse_boolean_flag(is_sal_flag_value):
            tags.append("salary_project_user")
        return tags

    def get_loyalty_tags(self, dt_bank_open_value):
        tags = []
        age_tags = self.get_company_age_tags(dt_bank_open_value) # Используем уже существующую функцию
        if "company_age_established" in age_tags:
            tags.append("loyalty_long_term_client_smb")
        return tags

    # --- Основная функция для обработки данных из Excel ---
    def process_excel_files(self, products_file, outgoing_ops_file, incoming_ops_file, contracts_file):
        """
        Читает данные из Excel, обрабатывает их и извлекает теги для каждого клиента.
        """
        try:
            df_products = pd.read_excel(products_file)
            df_outgoing_ops = pd.read_excel(outgoing_ops_file)
            df_incoming_ops = pd.read_excel(incoming_ops_file)
            df_contracts = pd.read_excel(contracts_file) # Добавляем чтение договоров
        except FileNotFoundError as e:
            logger.error(f"Ошибка: Файл не найден. {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при чтении Excel файла: {e}")
            return None

        # Объединяем исходящие и входящие операции для удобства
        df_all_ops = pd.concat([df_outgoing_ops, df_incoming_ops], ignore_index=True)

        # Убедимся, что CLI_ID имеет одинаковый тип для мержа
        df_products['CLI_ID'] = df_products['CLI_ID'].astype(str).str.replace(r'\.00$', '', regex=True)
        df_all_ops['CLI_ID'] = df_all_ops['CLI_ID'].astype(str).str.replace(r'\.00$', '', regex=True)
        df_contracts['CLI_ID'] = df_contracts['CLI_ID'].astype(str).str.replace(r'\.00$', '', regex=True)


        results = [] # Список для хранения результатов (CLI_ID, теги)

        # Итерация по уникальным клиентам из таблицы продуктов
        for index, client_row in df_products.iterrows():
            cli_id = client_row['CLI_ID']
            logger.info(f"\n--- Обработка клиента CLI_ID: {cli_id} ({client_row.get('CLN_NAME', 'N/A')}) ---")
            
            client_tags = set()

            # 1. Данные из таблицы "Продукты" (company_data)
            company_data = client_row.to_dict()

            # 2. Транзакции для этого клиента
            client_transactions_df = df_all_ops[df_all_ops['CLI_ID'] == cli_id]
            transaction_descriptions = client_transactions_df['ENTRY_DESCR'].dropna().astype(str).tolist()
            
            kassa_comis_total_client = company_data.get('KASSA_COMIS', 0)
            if pd.isna(kassa_comis_total_client): kassa_comis_total_client = 0

            # 3. Контракты для этого клиента (для уточнения debt_load)
            client_contracts_df_filtered = df_contracts[df_contracts['CLI_ID'] == cli_id]

            # Извлечение тегов
            client_tags.update(self.get_company_size_tags(company_data.get("STAFF_GROUP")))
            client_tags.update(self.get_company_age_tags(company_data.get("DT_BANK_OPEN")))
            
            client_tags.update(self.get_payment_type_tags_llm(transaction_descriptions))
            client_tags.update(self.get_cash_operations_tags_llm(transaction_descriptions, kassa_comis_total_client))
            
            client_tags.update(self.get_geo_tags(company_data.get("CITY")))
            client_tags.update(self.get_ved_tags(company_data.get("IS_VED"), transaction_descriptions))
            
            client_tags.update(self.get_acquiring_tags(company_data.get("IS_ACQ")))
            # Передаем отфильтрованные контракты клиента
            client_tags.update(self.get_debt_load_tags(company_data.get("IS_CREDIT"), client_contracts_df_filtered))
            client_tags.update(self.get_salary_project_tag(company_data.get("IS_SAL")))
            
            client_tags.update(self.get_loyalty_tags(company_data.get("DT_BANK_OPEN")))

            results.append({
                "CLI_ID": cli_id,
                "CLN_NAME": company_data.get('CLN_NAME', 'N/A'),
                "TAGS": list(client_tags)
            })
            
            logger.info(f"Извлеченные теги для {cli_id}: {list(client_tags)}")

        return results