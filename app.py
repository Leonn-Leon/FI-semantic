import streamlit as st
import pandas as pd
import ast
import altair as alt # Для графиков

# --- Конфигурация страницы Streamlit ---
st.set_page_config(
    page_title="Анализ Тегов Клиентов МСБ",
    page_icon="📊",
    layout="wide"
)

# --- Загрузка и кэширование данных ---
@st.cache_data
def load_data(file_path="client_tags_results_csv.csv"):
    try:
        df = pd.read_csv(file_path)
        # df.to_csv("client_tags_results_csv.csv", index=False)
        if not df.empty and 'TAGS' in df.columns:
            # Преобразование колонки TAGS, если она сохранена как строка-список
            if isinstance(df['TAGS'].iloc[0], str) and df['TAGS'].iloc[0].strip().startswith('['):
                try:
                    df['TAGS_List'] = df['TAGS'].apply(
                        lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']') else x
                    )
                except (ValueError, SyntaxError):
                     df['TAGS_List'] = df['TAGS'].apply(
                        lambda x: [tag.strip() for tag in x.split(',')] if pd.notna(x) and isinstance(x, str) else []
                    )
            elif isinstance(df['TAGS'].iloc[0], str):
                 df['TAGS_List'] = df['TAGS'].apply(
                        lambda x: [tag.strip() for tag in x.split(',')] if pd.notna(x) and isinstance(x, str) else []
                    )
            else:
                df['TAGS_List'] = df['TAGS']
            
            df['TAGS_List'] = df['TAGS_List'].apply(lambda x: x if isinstance(x, list) else [])
            # Создаем строку тегов для отображения в st.dataframe
            df['TAGS_String'] = df['TAGS_List']#.apply(lambda x: ", ".join(x) if x else "Нет тегов")
        else:
             st.warning("Колонка 'TAGS' отсутствует или файл пуст.")
             df['TAGS_List'] = pd.Series([[] for _ in range(len(df))])
             df['TAGS_String'] = pd.Series(["Нет тегов" for _ in range(len(df))])
        return df
    except FileNotFoundError:
        st.error(f"Ошибка: Файл '{file_path}' не найден.")
        return pd.DataFrame(columns=['CLI_ID', 'CLN_NAME', 'TAGS_List', 'TAGS_String'])
    except Exception as e:
        st.error(f"Ошибка при загрузке данных из Excel: {e}")
        return pd.DataFrame(columns=['CLI_ID', 'CLN_NAME', 'TAGS_List', 'TAGS_String'])

# --- Основная часть приложения ---
st.title("📊 Анализ Тегов Клиентов МСБ")
st.markdown("Интерактивное представление тегов, присвоенных клиентам малого и среднего бизнеса.")

data_df = load_data()

if not data_df.empty:
    st.sidebar.header("Фильтры и Настройки")

    search_name = st.sidebar.text_input("Поиск по наименованию клиента (CLN_NAME):", "")
    
    all_tags_list = []
    if 'TAGS_List' in data_df.columns: # Используем TAGS_List для получения уникальных тегов
        all_tags_list = sorted(list(set(tag for sublist in data_df['TAGS_List'] for tag in sublist)))

    selected_tags_filter = st.sidebar.multiselect(
        "Фильтр по тегам (клиенты с ВСЕМИ выбранными тегами):",
        options=all_tags_list,
        default=[]
    )

    filtered_df = data_df.copy()
    if search_name:
        filtered_df = filtered_df[filtered_df['CLN_NAME'].str.contains(search_name, case=False, na=False)]
    if selected_tags_filter:
        filtered_df = filtered_df[filtered_df['TAGS_List'].apply(lambda client_tags: all(tag in client_tags for tag in selected_tags_filter))]

    st.header("Данные по Клиентам и Их Тегам")

    if filtered_df.empty:
        st.warning("По вашему запросу клиенты не найдены.")
    else:
        st.info(f"Найдено клиентов: {len(filtered_df)}")
        show_cli_id = st.checkbox("Показывать CLI_ID", value=False, key="show_cli_id_checkbox")
        
        columns_to_display_in_table = ['CLN_NAME', 'TAGS_String'] # Используем TAGS_String
        if show_cli_id:
            columns_to_display_in_table.insert(0, 'CLI_ID')
        
        # Переименовываем колонку для отображения
        display_df_renamed = filtered_df[columns_to_display_in_table].rename(columns={'TAGS_String': 'Теги'})
        
        st.dataframe(
            display_df_renamed,
            hide_index=True,
            use_container_width=False
        )

        st.sidebar.markdown("---")
        st.sidebar.header("Аналитика")

        if st.sidebar.checkbox("Показать распределение тегов", value=True):
            st.header("Распределение Тегов по Клиентам")
            if not filtered_df.empty and 'TAGS_List' in filtered_df.columns:
                tag_counts_series = pd.Series([tag for sublist in filtered_df['TAGS_List'] for tag in sublist]).value_counts()
                if not tag_counts_series.empty:
                    source = pd.DataFrame({'Тег': tag_counts_series.index, 'Количество': tag_counts_series.values})
                    max_tags_to_show = st.slider("Количество тегов на графике:", 5, len(source) if len(source)>5 else 6, min(20, len(source) if len(source)>0 else 1 ), key="tags_slider")
                    source = source.head(max_tags_to_show)

                    base_chart = alt.Chart(source).encode(
                        x=alt.X('Количество:Q', title='Количество клиентов', axis=alt.Axis(grid=False)), # Убираем сетку по X для чистоты
                        y=alt.Y('Тег:N', title=None, sort='-x', axis=alt.Axis(labels=False, ticks=False, domain=False)) # Скрываем метки и ось Y слева
                    )

                    # Слой с полосами
                    bars = base_chart.mark_bar().encode(
                        tooltip=['Тег', 'Количество'] # Оставляем всплывающую подсказку
                    )

                    # Слой с текстом на полосах
                    text_on_bars = base_chart.mark_text(
                        align='left',
                        baseline='middle',
                        dx=5,
                        color='white',
                        fontSize=10
                    ).encode(
                        text='Тег:N'
                    )
                    
                    # Слой с текстом количества клиентов в конце полосы (опционально)
                    text_count_end_of_bar = base_chart.mark_text(
                        align='left',
                        baseline='middle',
                        dx=5,
                        color='black',
                        fontSize=9
                    ).encode(
                        text=alt.Text('Количество:Q', format='.0f'), 
                    )
                    final_chart = bars + text_on_bars


                    st.altair_chart(final_chart, use_container_width=True)
                else:
                    st.write("Нет тегов для анализа в отфильтрованных данных.")
            else:
                st.write("Нет данных для анализа распределения тегов.")

        if st.sidebar.checkbox("Показать клиентов по конкретному тегу", value=False):
            st.header("Поиск Клиентов по Одному Тегу")
            single_tag_select = st.selectbox(
                "Выберите тег для просмотра клиентов:",
                options=[""] + all_tags_list
            )
            if single_tag_select:
                clients_with_single_tag_df = data_df[data_df['TAGS_List'].apply(lambda tags: single_tag_select in tags)]
                if not clients_with_single_tag_df.empty:
                    st.write(f"Клиенты с тегом '{single_tag_select}':")

                    display_single_tag_df = clients_with_single_tag_df[['CLN_NAME', 'TAGS_String']].rename(columns={'TAGS_String': 'Теги'})
                    st.dataframe(display_single_tag_df, hide_index=True, use_container_width=False)
                else:
                    st.write(f"Нет клиентов с тегом '{single_tag_select}'.")

    st.sidebar.markdown("---")
    st.sidebar.info(f"Загружено {len(data_df)} записей о клиентах.")
    if selected_tags_filter or search_name:
        st.sidebar.info(f"Отображается {len(filtered_df)} клиентов после фильтрации.")
else:
    st.warning("Не удалось загрузить данные.")

st.markdown("---")
st.markdown("Разработано для анализа клиентских тегов МСБ.")