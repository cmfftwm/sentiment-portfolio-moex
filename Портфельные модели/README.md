# Портфельные модели

Итоговые портфельные стратегии из магистерской работы (сентимент-анализ новостного фона → портфель акций MOEX).

- Стартовое обучение: **2020 г.**
- Out-of-sample walk-forward тест: **2021-01-01 — 2025-11-19** (expanding-окно)
- Переобучение: ежемесячное (основной режим); для M1 также есть недельный и квартальный режимы
- Комиссия: 0.03% на сделку
- Бенчмарки: IMOEX, ставка ЦБ

## Портфели

| Модель | Скрипт | Доходность за период | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|
| **v1** — простое сентимент-правило | `step_v1_sentiment_expanding.py` | +40.4% | +6.7% | 0.335 | −30.4% |
| **M1** — базовая ML-модель (ежемесячно) | `step_walkforward_monthly.py` | +84.1% | +12.4% | 0.508 | −36.2% |
| **M2** — ML + softmax-взвешивание | `step_walkforward_monthly_weighted_vol.py` | +131.7% | +17.5% | 0.694 | −34.1% |
| **M3** — расширенная ML-модель (полная) | `step_walkforward_monthly_weighted_v4_brent_divret.py` | +783.0% | +51.8% | 2.221 | −25.3% |

**v1** — top-10 по `ticker_sent_7d`, равновзвешенно, выход в кэш при `market_sent_7d` ниже expanding-перцентиля.
**M1** — LightGBM прогноз 5-дневной доходности, top-5 равновзвешенно. Варианты частоты переобучения: `step_walkforward_weekly.py` (Sharpe 0.409), `step_walkforward.py` (квартально, 0.465), `step_walkforward_monthly.py` (ежемесячно, 0.508).
**M2** — то же + softmax-взвешивание позиций по уверенности модели.
**M3** — полная модель: + Brent, дивиденды, go_cash под ставку ЦБ, CBR pos_scale. Абляции:
- `..._v4_brent_divret_cashonly.py` — кэш под 0% вместо ставки ЦБ (Sharpe 2.138)
- `..._v4_brent_divret_nogocash.py` — без защитного выхода в кэш (Sharpe 1.332)
- `..._nosent_v4_brent_divret_nogocash.py` — без сентимента и без go_cash (Sharpe 1.426)

## Конвейер сборки фич (порядок запуска)

```
step1_build_features.py      # → features_daily.parquet
step2_add_prices.py          # → dataset.parquet
step2b_add_volume.py         # → dataset_vol.parquet        (для v1, M1, M2)
step2d_add_negpos.py         # → dataset_vol_negpos.parquet
step2e_add_dividends.py      # → dataset_vol_divs.parquet
step2g_add_brent.py          # → dataset_vol_brent.parquet  (для M3)
```

После сборки датасетов запускаются скрипты портфелей (выше). `plot_thesis_charts.py` строит графики для работы.

## Исходные данные (не в репозитории — большие/локальные)

В корне проекта:
- `telegram_messages.db` — новости + сентимент по тикерам
- `moex_1m.db` — минутные свечи MOEX

В `Парсер рыночных данных/`:
- `brent_daily.csv`, `dividends_moex.csv`, `cbr_key_rate_2020_2025.csv`, `usdrub_cbr.csv`, `moex_indices.db`

## Отчёты

`backtest_report_*.txt` рядом с каждым скриптом — метрики (доходность, Sharpe, Max DD, разбивка по годам) и сравнение с бенчмарками.
