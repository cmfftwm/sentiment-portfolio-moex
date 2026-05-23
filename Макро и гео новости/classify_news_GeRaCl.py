"""
Классификация новостей по категориям: macro / geo / company / skip
Модель: deepvk/GeRaCl-USER2-base (zero-shot classification)
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# geracl лежит рядом
sys.path.insert(0, "geracl")
from transformers import AutoTokenizer
from geracl import GeraclHF, ZeroShotClassificationPipeline

# ─── Настройки ──────────────────────────────────────────────────────────────

DB_PATH    = "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages.db"
CHANNEL    = "rbc_news"
BATCH_SIZE = 32
LIMIT      = 500        # None = все сообщения
OUTPUT     = "classified_news.csv"

MODEL_NAME = "deepvk/GeRaCl-USER2-base"

# Метки для zero-shot классификации.
# GeRaCl выбирает одну из них — формулировки влияют на качество.
LABELS = [
    "макроэкономическая новость о ключевой ставке ЦБ РФ, инфляции, ВВП, бюджете России, курсе рубля или ценах на нефть как экономическом индикаторе для бюджета РФ",
    "геополитическая новость о войне на Украине, западных санкциях против России, военных действиях, переговорах о мире или международных конфликтах влияющих на российский рынок",
    "корпоративная новость о финансовых результатах конкретной компании, её дивидендах, IPO, сделках или смене руководства",
    "нерелевантная новость не влияющая на российский фондовый рынок: политика, спорт, культура, криминал, технологии, погода",
]

LABEL_MAP = {
    0: "macro",
    1: "geo",
    2: "company",
    3: "skip",
}

# ─── Предфильтр ──────────────────────────────────────────────────────────────
# Убираем явно нерелевантное перед вызовом модели (ускорение в ~2x).

RELEVANT_KEYWORDS = [
    # macro
    "ставк", "инфляц", "цб рф", "центробанк", "центральный банк",
    "ввп", "минфин", "бюджет", "дефицит", "профицит",
    "нефть", "брент", "urals", "рубл", "курс доллар", "курс евро",
    "фрс", "федрезерв", "fomc", "денежно-кредитн",
    "экспорт", "импорт", "торговый баланс", "безработиц",
    # geo
    "сво", "украин", "зеленск", "нато", "санкц", "переговор",
    "прекращени огня", "всу", "обстрел", "наступлени",
    "путин", "трамп", "байден", "тайвань",
    "израил", "газа", "ближний восток", "геополитик", "эмбарго",
]

def keyword_prefilter(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in RELEVANT_KEYWORDS)


# ─── Загрузка данных ─────────────────────────────────────────────────────────

def load_messages(db_path: str, channel: str, limit=None) -> pd.DataFrame:
    table = f"messages_{channel}"
    query = f"SELECT message_id, date, message FROM {table} WHERE message IS NOT NULL"
    if limit:
        query += f" ORDER BY date DESC LIMIT {limit}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    print(f"Загружено {len(df)} сообщений из {table}")
    return df


# ─── Главный пайплайн ────────────────────────────────────────────────────────

def main():
    # 1. Загрузка данных
    df = load_messages(DB_PATH, CHANNEL, limit=LIMIT)

    # 2. Предфильтр
    mask = df["message"].apply(keyword_prefilter)
    df_filtered = df[mask].copy().reset_index(drop=True)
    print(f"После предфильтра: {len(df_filtered)} сообщений ({mask.mean():.1%} от загруженных)")

    if df_filtered.empty:
        print("Нет сообщений после фильтра.")
        return

    # 3. Загрузка модели
    # GeRaCl не поддерживает MPS — используем CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загрузка модели {MODEL_NAME} на {device}...")

    model     = GeraclHF.from_pretrained(MODEL_NAME).to(device).eval()
    model._classification_core._device = device  # GeraclCore хранит device отдельно
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipe      = ZeroShotClassificationPipeline(model, tokenizer, device=device)

    # 4. Классификация батчами
    texts = df_filtered["message"].tolist()
    all_labels = []
    all_scores = []

    print("Классификация...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        # pipe возвращает list[int] — индекс лучшего лейбла
        results = pipe(batch, LABELS, batch_size=len(batch))
        for j, idx in enumerate(results):
            # GeRaCl возвращает индекс выбранного класса
            # Для confidence нужно вызвать с return_scores=True если поддерживается
            all_labels.append(LABEL_MAP[idx])
            all_scores.append(idx)  # заменим на score ниже

    df_filtered["category"] = all_labels

    # 5. Результаты
    print("\n=== Распределение по категориям ===")
    counts = df_filtered["category"].value_counts()
    print(counts)

    print("\n=== Примеры по категориям ===")
    for cat in ["macro", "geo", "company"]:
        subset = df_filtered[df_filtered["category"] == cat].head(3)
        if subset.empty:
            continue
        print(f"\n--- {cat.upper()} ---")
        for _, row in subset.iterrows():
            preview = row["message"][:250].replace("\n", " ")
            print(f"  {preview}")

    # 6. Сохранение
    df_filtered[["message_id", "date", "message", "category"]].to_csv(OUTPUT, index=False)
    print(f"\nСохранено в {OUTPUT}")

    # Также сохраним только macro и geo
    df_relevant = df_filtered[df_filtered["category"].isin(["macro", "geo"])]
    df_relevant.to_csv("classified_relevant.csv", index=False)
    print(f"Только macro+geo: {len(df_relevant)} сообщений → classified_relevant.csv")


if __name__ == "__main__":
    main()
