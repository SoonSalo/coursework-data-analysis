import numpy as np
import pandas as pd

# Фиксируем случайные числа для воспроизводимости данных
np.random.seed(42)

# Генерируем 715 последовательных меток времени (шаг - 1 минута)
timestamps = pd.date_range(
    start="2026-03-15 08:00:00", periods=715, freq="min"
)

# GVF (объем газа): плавно растет от 0 до 70% с небольшим шумом
gvf = np.linspace(0.0, 0.70, 715) + np.random.normal(0, 0.01, 715)
gvf = np.clip(gvf, 0.0, 0.70)  # Строго удерживаем в диапазоне кейса

# WC (обводненность): колеблется от 0 до 100% (имитируем смену режимов)
wc = 0.5 + 0.45 * np.sin(np.linspace(0, 12, 715)) + np.random.normal(0, 0.02, 715)
wc = np.clip(wc, 0.0, 1.0)  # Строго удерживаем в диапазоне от 0% до 100%

# Истинный массовый расход (стабильный базовый поток на стенде)
mass_flow_true = 15.0 + 1.5 * np.cos(np.linspace(0, 5, 715))

# Сырой массовый расход с прибора (падает и шумит при росте завоздушенности GVF)
mass_flow_raw = mass_flow_true - (gvf * 5.5) + np.random.normal(0, 0.15, 715)

# Плотность смеси (падает при увеличении газа)
density_true = 980.0 * (1 - gvf) + np.random.normal(0, 2.0, 715)

# Собираем финальную таблицу
df_tuv = pd.DataFrame(
    {
        "timestamp": timestamps,
        "mass_flow_raw": np.round(mass_flow_raw, 2),
        "density": np.round(density_true, 1),
        "GVF": np.round(gvf, 3),
        "WC": np.round(wc, 3),
        "mass_flow_true": np.round(mass_flow_true, 2),
    }
)

# Сохраняем строго под именем, которое мы будем использовать в коде
df_tuv.to_csv("tuv_nel_data.csv", index=False)
print("=" * 50)
print("Ура! Датасет для Кейса №60 успешно сгенерирован!")
print(f"Файл: tuv_nel_data.csv | Строк: {len(df_tuv)} | Колонок: {len(df_tuv.columns)}")
print("=" * 50)
