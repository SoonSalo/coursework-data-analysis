import os
from collections import Counter
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import pymorphy3
import re
import urllib.request
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Скачиваем список стоп-слов, если его нет
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ==========================================
# ШАГ 0: Автоматическая загрузка и чтение реального датасета с GitHub
# ==========================================
# ==========================================
# ШАГ 0: Чтение локального датасета
# ==========================================
local_filename = "premise_question_answer4.txt"
print("Читаем локальный датасет обращений...")


# Читаем и парсим файл, распределяя строки на Вопросы (Категория Q) и Ответы (Категория A)
questions, answers = [], []
with open(local_filename, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("Q:"):
            questions.append(line.replace("Q:", "").strip())
        elif line.startswith("A:"):
            answers.append(line.replace("A:", "").strip())
        # Ограничиваем выборку первыми 300 парами для оптимальной скорости вычислений
        if len(questions) >= 300:
            break

# Формируем итоговую таблицу данных под наш Кейс №68 (Обращения граждан/клиентов)
data = {
    "category": ["Вопросы клиентов"] * len(questions) + ["Ответы системы"] * len(answers),
    "text": questions + answers
}
df = pd.DataFrame(data)
print(f"Загружен реальный корпус текстов! Всего строк для анализа: {len(df)}")

# ==========================================
# ЭТАП 1: Очистка текста
# ==========================================
def clean_text(text):
    text = text.lower()                    # Всё в нижний регистр
    text = re.sub(r'[^а-яё ]', '', text)   # Удаляем всё, кроме русских букв и пробелов
    text = ' '.join(text.split())          # Убираем лишние пробелы
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ==========================================
# ЭТАП 2: Лемматизация (приведение к начальной форме)
# ==========================================
morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

df['lemmas'] = df['clean_text'].apply(lemmatize_text)

# ==========================================
# ЭТАП 3: Подсчёт частоты слов и визуализация Топ-10
# ==========================================
all_words = ' '.join(df['lemmas']).split()
word_counts = Counter(all_words)
top_words = word_counts.most_common(10)
words, counts = zip(*top_words)

plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='#008080', alpha=0.85, edgecolor='black')
plt.title('Топ-10 самых частых слов в реальном датасете (до удаления стоп-слов)')
plt.xlabel('Слова')
plt.ylabel('Частота встречи')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figure9.png", dpi=300)
plt.show()

# ==========================================
# ЭТАП 4: Удаление стоп-слов и построение Облаков слов (WordCloud)
# ==========================================
stop_words = set(stopwords.words('russian'))

def remove_stopwords(text, stop_words):
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['no_stopwords'] = df['lemmas'].apply(lambda x: remove_stopwords(x, stop_words))

# Разделяем облака слов по категориям: Вопросы vs Ответы
q_text = ' '.join(df[df['category'] == 'Вопросы клиентов']['no_stopwords'])
a_text = ' '.join(df[df['category'] == 'Ответы системы']['no_stopwords'])

wc_q = WordCloud(width=400, height=300, background_color='white', max_words=20).generate(q_text)
wc_a = WordCloud(width=400, height=300, background_color='white', max_words=20).generate(a_text)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(wc_q, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Облако слов: Вопросы клиентов', fontsize=14)

axes[1].imshow(wc_a, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Облако слов: Ответы автоматики', fontsize=14)

plt.tight_layout()
plt.savefig("figure10.png", dpi=300)
plt.show()

# ==========================================
# ЭТАП 5: TF-IDF Векторизация
# ==========================================
vectorizer = TfidfVectorizer(stop_words=list(stop_words))
tfidf_matrix = vectorizer.fit_transform(df['no_stopwords'])
feature_names = vectorizer.get_feature_names_out()

print("\n" + "="*50)
print(f"TF-IDF ВЕКТОРИЗАЦИЯ УСПЕШНО ВЫПОЛНЕНА")
print(f"Размер сформированного словаря: {len(feature_names)} уникальных слов")
print(f"Размерность числовой матрицы: {tfidf_matrix.shape}")
print("="*50)

# ==========================================
# ЭТАП 6: Информационный семантический поиск по косинусному сходству
# ==========================================
def search_texts(query, vectorizer, tfidf_matrix, texts, top_n=2):
    query_clean = clean_text(query)
    query_lemma = lemmatize_text(query_clean)
    query_vec = vectorizer.transform([query_lemma])

    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0: # Выводим только если есть семантическое совпадение
            results.append({'Текст': texts.iloc[idx], 'Похожесть': similarities[idx]})
    return results

# Тестируем семантический поиск на реальном запросе
test_query = "кто удивился?"
search_results = search_texts(test_query, vectorizer, tfidf_matrix, df['text'], top_n=2)

print(f"\nПоисковый запрос ИИ: '{test_query}'")
print(f"Найдено совпадений: {len(search_results)}")
for i, res in enumerate(search_results, 1):
    print(f"{i}. [Косинусное сходство: {res['Похожесть']:.4f}] — Текст: {res['Текст']}")
print("="*50)
