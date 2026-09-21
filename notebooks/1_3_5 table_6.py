import numpy as np
import pandas as pd

# Воссоздаем локальный датасет точно по вашей формуле для сохранения идентичности цифр
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

df = pd.DataFrame({'charges': charges})

# Математический расчет выбросов по методу IQR
q1 = df['charges'].quantile(0.25)
q3 = df['charges'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[df['charges'] > upper_bound]

print(f"Первый квартиль (Q1): {q1:.2f}")
print(f"Третий квартиль (Q3): {q3:.2f}")
print(f"Межквартильный размах (IQR): {iqr:.2f}")
print(f"Верхняя граница нормы: {upper_bound:.2f}")
print(f"Количество выбросов выше границы: {len(outliers)} шт. ({len(outliers)/n_samples*100:.2f}%)")
