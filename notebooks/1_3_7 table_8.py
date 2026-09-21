import numpy as np
import pandas as pd

# Воссоздаем локальный датасет строго по вашей формуле для тождественности цифр
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

df = pd.DataFrame({'bmi': bmi, 'charges': charges})

# 1. Запоминаем исходные статистики признака bmi
bmi_original_mean = df['bmi'].mean()
bmi_original_std = df['bmi'].std()

# 2. Моделируем случайный измерительный шум (Гауссово распределение с std=0.5)
# Фиксируем случайность внутри шага для воспроизводимости
np.random.seed(100)
noise = np.random.normal(0, 0.5, n_samples)

# Создаем зашумленный признак
df['bmi_noisy'] = (df['bmi'] + noise).round(1)

# 3. Рассчитываем новые статистики зашумленного признака
bmi_noisy_mean = df['bmi_noisy'].mean()
bmi_noisy_std = df['bmi_noisy'].std()

# 4. Оцениваем отклонение в процентах
mean_diff_pct = abs(bmi_noisy_mean - bmi_original_mean) / bmi_original_mean * 100
std_diff_pct = abs(bmi_noisy_std - bmi_original_std) / bmi_original_std * 100

print(f"Исходное среднее bmi: {bmi_original_mean:.4f}")
print(f"Зашумленное среднее bmi: {bmi_noisy_mean:.4f} (Отклонение: {mean_diff_pct:.4f}%)")
print(f"Исходное std bmi: {bmi_original_std:.4f}")
print(f"Зашумленное std bmi: {bmi_noisy_std:.4f} (Отклонение: {std_diff_pct:.4f}%)")
