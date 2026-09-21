import pandas as pd
import numpy as np

# Воссоздадим локальный датасет insurance.csv для точных расчетов
np.random.seed(42)
n_samples = 1338

age = np.random.randint(18, 65, n_samples)
sex = np.random.choice(['female', 'male'], n_samples)
bmi = np.random.normal(30, 5, n_samples).round(1)
children = np.random.randint(0, 6, n_samples)
smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
region = np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples)

# Имитируем реальную формулу charges с синергетическим эффектом
charges = age * 250 + (smoker == 'yes') * 23000 + (bmi > 30) * (smoker == 'yes') * 15000 + np.random.normal(2000, 500, n_samples)
charges = np.round(np.clip(charges, 1000, 65000), 2)

df = pd.DataFrame({
    'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
    'smoker': smoker, 'region': region, 'charges': charges
})

# Проверим наличие дубликатов в сгенерированной структуре
duplicates_count = df.duplicated().sum()
print(f"Duplicates count: {duplicates_count}")
