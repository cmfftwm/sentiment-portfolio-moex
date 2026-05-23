"""
macro_geo_signal.py

Для каждой macro/geo новости определяет через YandexGPT:
  - scope:   "market" (весь рынок) или "sector" (конкретные секторы)
  - sectors: список затронутых секторов из tickers.json
  → маппинг секторов на тикеры

Сентимент — отдельно, позже через GeRaCl.

Вход:  classified_relevant_yandex.csv  (macro + geo новости)
Выход: macro_geo_signals.csv
"""

import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Настройки ───────────────────────────────────────────────────────────────

INPUT_CSV    = "classified_relevant_yandex.csv"
TICKERS_JSON = "tickers.json"
OUTPUT_CSV   = "macro_geo_signals.csv"
MODEL_NAME   = "yandex/YandexGPT-5-Lite-8B-instruct"

# ─── Загрузка секторов и тикеров из tickers.json ────────────────────────────

def load_tickers(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sector_to_tickers: dict[str, list[str]] = {}
    for item in data:
        sector = item["sector"]
        ticker = item["tiker"]
        sector_to_tickers.setdefault(sector, []).append(ticker)
    return sector_to_tickers

# ─── Промпт ──────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """Ты анализируешь новости для управления портфелем российских акций (MOEX).

Доступные секторы:
{sectors_list}

Правила:
- scope="market" ТОЛЬКО если новость одновременно влияет на ВСЕ или почти все секторы (например: начало войны, дефолт, резкий обвал рубля, глобальный кризис)
- scope="sector" во всех остальных случаях — даже если затронуто 3-5 секторов
- sectors — перечисли ТОЛЬКО реально затронутые секторы через запятую (точно из списка выше)

Примеры:
- "Нефть Urals упала до $37" → scope: sector / sectors: Нефть и газ
- "Санкции на теневой флот" → scope: sector / sectors: Морские перевозки, Нефть и газ
- "ЦБ поднял ставку до 21%" → scope: sector / sectors: Банки, Девелопмент
- "Россия начала СВО" → scope: market / sectors: все

Отвечай строго в формате (2 строки, без пояснений):
scope: market/sector
sectors: Банки, Нефть и газ

Новость: {text}"""

# ─── Парсинг ответа ──────────────────────────────────────────────────────────

def parse_response(raw: str, valid_sectors: set[str], all_sectors: list[str]) -> dict:
    result = {"scope": "sector", "sectors": []}

    for line in raw.lower().splitlines():
        line = line.strip()

        if line.startswith("scope:"):
            val = line.split(":", 1)[1].strip()
            result["scope"] = "market" if "market" in val or "весь" in val or "все" in val else "sector"

        elif line.startswith("sectors:"):
            val = line.split(":", 1)[1].strip()
            if "все" in val or "all" in val:
                result["sectors"] = all_sectors
            else:
                found = [s for s in valid_sectors if s.lower() in val]
                result["sectors"] = found

    # Если scope=market но sectors пустой — заполняем все
    if result["scope"] == "market" and not result["sectors"]:
        result["sectors"] = all_sectors

    return result

# ─── Инференс одной новости ──────────────────────────────────────────────────

def analyze_one(text: str, sectors_list: str, valid_sectors: set,
                all_sectors: list, model, tokenizer, device: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(sectors_list=sectors_list, text=text[:1200])
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_response(raw, valid_sectors, all_sectors)

# ─── Главный пайплайн ────────────────────────────────────────────────────────

def main():
    # 1. Данные
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено новостей: {len(df)}")
    print(df["category"].value_counts())

    # 2. Тикеры и секторы
    sector_to_tickers = load_tickers(TICKERS_JSON)
    valid_sectors = set(sector_to_tickers.keys())
    all_sectors   = sorted(valid_sectors)
    sectors_list  = "\n".join(f"- {s}" for s in all_sectors)
    print(f"\nСекторов в tickers.json: {len(valid_sectors)}")

    # 3. Модель
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nЗагрузка {MODEL_NAME} на {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    ).eval()
    print("Модель загружена.")

    # 4. Анализ
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Анализ"):
        res = analyze_one(
            row["message"], sectors_list, valid_sectors,
            all_sectors, model, tokenizer, device
        )
        # market — тикеры не заполняем, обрабатывается отдельно в бэктесте
        if res["scope"] == "market":
            affected_sectors = ""
            affected_tickers = ""
        else:
            affected_sectors = ", ".join(res["sectors"])
            affected_tickers = ", ".join(sorted(set(
                t for s in res["sectors"] for t in sector_to_tickers.get(s, [])
            )))

        rows.append({
            "message_id":       row["message_id"],
            "date":             row["date"],
            "category":         row["category"],
            "message":          row["message"],
            "scope":            res["scope"],
            "affected_sectors": affected_sectors,
            "affected_tickers": affected_tickers,
        })

    # 5. Результаты
    out = pd.DataFrame(rows)
    print("\n=== Распределение scope ===")
    print(out["scope"].value_counts())

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nСохранено в {OUTPUT_CSV}")

    print("\n=== Примеры ===")
    for _, r in out.head(5).iterrows():
        print(f"[{r['category']}] scope={r['scope']}")
        print(f"  Секторы: {r['affected_sectors']}")
        print(f"  Тикеры:  {r['affected_tickers']}")
        print(f"  {r['message'][:150].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
