import numpy as np
import pandas as pd

# Воссоздаем датасет точно по вашей формуле
np.random.seed(42)
n_samples = 1338
age = np.random.randint(18, 65, n_samples)
sex = np.random.choice(['female', 'male'], n_samples)
bmi = np.random.normal(30, 5, n_samples).round(1)
children = np.random.randint(0, 6, n_samples)
smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
region = np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples)

charges = age * 250 + (smoker == 'yes') * 23000 + (bmi > 30) * (smoker == 'yes') * 15000 + np.random.normal(2000, 500, n_samples)
charges = np.round(np.clip(charges, 1000, 65000), 2)

df = pd.DataFrame({
    'age': age, 'bmi': bmi, 'smoker': smoker, 'charges': charges
})

# Считаем общую среднюю стоимость страховки
mean_all = df['charges'].mean()

# Применяем фильтрацию по заданному условию (Курильщики с ИМТ > 30)
filtered_df = df[(df['smoker'] == 'yes') & (df['bmi'] > 30)]

# Считаем метрики по отфильтрованной группе
count_filtered = len(filtered_df)
mean_filtered = filtered_df['charges'].mean()
min_filtered = filtered_df['charges'].min()
max_filtered = filtered_df['charges'].max()

print(f"Общее среднее по всей выборке: {mean_all:.2f} $")
print(f"Количество записей после фильтрации: {count_filtered} шт.")
print(f"Средняя стоимость в группе риска: {mean_filtered:.2f} $")
print(f"Минимальная стоимость в группе риска: {min_filtered:.2f} $")
print(f"Максимальная стоимость в группе риска: {max_filtered:.2f} $")
