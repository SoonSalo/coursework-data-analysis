import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Загрузка датасета временного ряда
df = pd.read_csv("tuv_nel_data.csv")

# Настройка уникального и чистого стиля для анализа шумов
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

# 2. Математическое выделение шума (Детрендирование через скользящее среднее)
# Берем окно в 15 точек, чтобы построить плавную линию основного тренда
smooth_trend = df["mass_flow_raw"].rolling(window=15, center=True).mean()
# Вычитаем тренд из сырого сигнала, оставляя только чистую шумовую компоненту
noise_component = df["mass_flow_raw"] - smooth_trend

# 3. Визуализация результатов анализа
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Верхний график: Сырой сигнал и выделенный гладкий тренд
axes[0].plot(
    df["mass_flow_raw"],
    color="#E07A5F",
    alpha=0.5,
    label="Сырой зашумленный сигнал (raw)",
)
axes[0].plot(
    smooth_trend,
    color="#008080",
    linewidth=2.5,
    label="Выделенный полезный тренд (после фильтрации)",
)
axes[0].set_title(
    "Процесс фильтрации: отделение полезного тренда от измерительного шума"
)
axes[0].set_ylabel("Массовый расход, кг/с")
axes[0].legend(loc="upper right")

# Нижний график: Чистая шумовая компонента (высокочастотные пульсации)
axes[1].plot(
    noise_component,
    color="#E07A5F",
    linewidth=1,
    alpha=0.9,
    label="Выделенный измерительный шум",
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
axes[1].set_title("Хронологический график высокочастотной шумовой компоненты")
axes[1].set_xlabel("Индекс временной точки (последовательные замеры)")
axes[1].set_ylabel("Амплитуда шума, кг/с")
axes[1].legend(loc="upper right")

plt.tight_layout()
# Сохраняем итоговый рисунок для отчета
plt.savefig("figure5.png", dpi=300)
plt.show()
