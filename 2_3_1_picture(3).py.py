import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# ЭТАП 1: Загрузка и первичное знакомство
# ==========================================
# Загрузка локального файла данных, соответствующего кейсу TUV-NEL
df = pd.read_csv("tuv_nel_data.csv")

# Выводим общую информацию о структуре в консоль
print("=" * 60)
print("ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ")
print("=" * 60)
print(df.info())
print("\nПервые 5 строк датасета:")
print(df.head())
print("=" * 60)

# ==========================================
# ЭТАП 2: Визуализация исходных данных
# Используем наш фирменный изумрудный стиль с сеткой
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

# Создаем сетку графиков: 2 строки, 2 столбца
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)

# 1. График массового расхода (Сравнение Истинного и Сырого)
axes[0, 0].plot(
    df["mass_flow_true"],
    label="Истинный расход (true)",
    color="#008080",
    linewidth=2,
)
axes[0, 0].plot(
    df["mass_flow_raw"],
    label="Сырой расход (raw)",
    color="#E07A5F",
    alpha=0.6,
    linestyle="--",
)
axes[0, 0].set_title("Динамика массового расхода")
axes[0, 0].set_ylabel("Расход, кг/с")
axes[0, 0].legend()

# 2. График плотности смеси
axes[0, 1].plot(df["density"], color="#008080", linewidth=2)
axes[0, 1].set_title("Изменение плотности смеси во времени")
axes[0, 1].set_ylabel("Плотность, кг/м³")

# 3. График объемного содержания газа (GVF)
axes[1, 0].plot(df["GVF"], color="#008080", linewidth=2)
axes[1, 0].set_title("Объемное содержание газа (GVF)")
axes[1, 0].set_xlabel("Индекс временной точки")
axes[1, 0].set_ylabel("Доля газа (0.0 - 0.7)")

# 4. График обводненности (WC)
axes[1, 1].plot(df["WC"], color="#008080", linewidth=2)
axes[1, 1].set_title("Обводненность жидкой фазы (WC)")
axes[1, 1].set_xlabel("Индекс временной точки")
axes[1, 1].set_ylabel("Доля воды (0.0 - 1.0)")

plt.tight_layout()
# Сохраняем график для отчета
plt.savefig("figure3.png", dpi=300)
plt.show()
