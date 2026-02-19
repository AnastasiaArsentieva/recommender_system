import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def load_and_prepare_data():
    # Если файл есть, пытаемся прочитать. Если ошибка - пересобираем.
    if os.path.exists('final_df.pkl'):
        try:
            return pd.read_pickle('final_df.pkl')
        except Exception:
            print("[WARNING] Старый кэш несовместим. Пересоздаю...")
            os.remove('final_df.pkl')

    print("[PROCESS] Первичная обработка CSV...")
    # Принудительно читаем все как обычные объекты (строки)
    books = pd.read_csv('Books.csv', low_memory=False, dtype=str)
    ratings = pd.read_csv('Ratings.csv', dtype=str)
    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce').fillna(0)

    # Очистка
    user_counts = ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts.astype(int) > 50].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]

    ratings_with_books = ratings.merge(books, on='ISBN')

    book_counts = ratings_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
    popular_books = book_counts[book_counts['Book-Rating'] >= 10]['Book-Title']

    final_rating = ratings_with_books[ratings_with_books['Book-Title'].isin(popular_books)]

    # КРИТИЧЕСКИЙ МОМЕНТ: принудительно переводим всё в стандартные типы Python
    for col in final_rating.columns:
        if final_rating[col].dtype.name == 'string' or final_rating[col].dtype == object:
            final_rating[col] = final_rating[col].astype(str)

    final_rating.to_pickle('final_df.pkl')
    return final_rating


df = load_and_prepare_data()

print("[PROCESS] Создание матрицы...")
# Переводим индексы в простые строки ПЕРЕД pivot_table
df['Book-Title'] = df['Book-Title'].astype(str)
book_pivot = df.pivot_table(columns='User-ID', index='Book-Title', values='Book-Rating')
book_pivot.index = book_pivot.index.map(str)
book_pivot.fillna(0, inplace=True)

print("[PROCESS] Обучение модели...")
model = NearestNeighbors(algorithm='brute', metric='cosine')
model.fit(csr_matrix(book_pivot))

# Сохраняем модель
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('book_pivot.pkl', 'wb') as f:
    pickle.dump(book_pivot, f)

print("Готово! Все файлы обновлены. Теперь можно запускать Streamlit.")







