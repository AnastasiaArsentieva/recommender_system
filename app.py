import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Book Recommender System", layout="wide")


@st.cache_resource
def load_all():
    # Проверяем наличие файлов
    if not (os.path.exists('model.pkl') and os.path.exists('book_pivot.pkl') and os.path.exists('final_df.pkl')):
        st.error("Критическая ошибка: Файлы модели (.pkl) не найдены. Сначала запустите main.py!")
        st.stop()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('book_pivot.pkl', 'rb') as f:
        pivot = pickle.load(f)

    # Загружаем датасет и принудительно приводим ключевые колонки к строкам
    df = pd.read_pickle('final_df.pkl')
    df['User-ID'] = df['User-ID'].astype(str)
    df['Book-Title'] = df['Book-Title'].astype(str)

    return model, pivot, df


# Загрузка данных
try:
    model, pivot, df = load_all()
except Exception as e:
    st.error(f"Ошибка при чтении кэша: {e}")
    st.info("Решение: Удалите все .pkl файлы и заново запустите main.py")
    st.stop()

st.title("📚 Умная система рекомендаций книг")
st.markdown("---")

# Боковая панель для навигации
option = st.sidebar.selectbox("Выберите режим поиска:", ("По названию книги", "По ID пользователя"))


def display_posters(suggestions, pivot_df, full_df):
    """Вспомогательная функция для отрисовки обложек в ряд"""
    cols = st.columns(5)
    for i in range(1, len(suggestions)):
        book_title = pivot_df.index[suggestions[i]]

        # Получаем URL обложки (берем первый попавшийся из дубликатов)
        poster_data = full_df[full_df['Book-Title'] == book_title]['Image-URL-M']
        if not poster_data.empty:
            # Исправлено: извлекаем строку URL из Series
            poster_url = poster_data.iloc[0]
        else:
            poster_url = "https://via.placeholder.com"

        with cols[i - 1]:
            st.image(poster_url, use_container_width=True)
            st.caption(f"**{book_title[:50]}...**")


if option == "По названию книги":
    st.subheader("Найдите книги, похожие на вашу любимую")
    book_list = pivot.index.values
    selected_book = st.selectbox("Введите название книги:", book_list)

    if st.button('Найти похожие'):
        try:
            # Получаем индекс выбранной книги
            idx = np.where(pivot.index == selected_book)[0][0]
            distances, suggestions = model.kneighbors(pivot.iloc[idx, :].values.reshape(1, -1), n_neighbors=6)

            st.success(f"Пользователи, читавшие '{selected_book}', также оценили:")
            display_posters(suggestions[0], pivot, df)
        except Exception as e:
            st.error(f"Не удалось найти рекомендации: {e}")

elif option == "По ID пользователя":
    st.subheader("Персональные рекомендации для читателя")
    # Используем текстовый ввод, так как ID в базе теперь строки
    user_id_input = st.text_input("Введите ID пользователя:", value="276847")

    if st.button('Получить подборку'):
        user_id = str(user_id_input).strip()

        # Фильтруем данные пользователя
        user_ratings = df[df['User-ID'] == user_id].sort_values(by='Book-Rating', ascending=False)

        if not user_ratings.empty:
            # Берем самую высокооцененную книгу пользователя
            fav_book = user_ratings.iloc[0]['Book-Title']
            st.info(f"Фаворит этого пользователя: **{fav_book}**")

            try:
                # Ищем похожие на фаворита
                idx = np.where(pivot.index == fav_book)[0][0]
                distances, suggestions = model.kneighbors(pivot.iloc[idx, :].values.reshape(1, -1), n_neighbors=6)

                st.subheader("Вам может понравиться:")
                display_posters(suggestions[0], pivot, df)
            except Exception as e:
                st.warning("Книга-фаворит слишком редкая для расчета сходства по соседям.")
        else:
            st.error(f"Пользователь с ID {user_id} не найден или у него нет оценок в нашей базе.")

# Футер
st.markdown("---")
st.caption("Проект гибридной рекомендательной системы | Анализ данных и ML")


