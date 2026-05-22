"""
fetch_usdrub_cbr.py

Загружает официальный курс USD/RUB с сайта ЦБ РФ (XML API).
Выход: usdrub_cbr.csv (date, usdrub)
"""

import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import ssl
import os
import time

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_CSV   = os.path.join(BASE_DIR, "usdrub_cbr.csv")

DATE_FROM = "01/01/2019"
DATE_TO   = "30/11/2025"
USD_CODE  = "R01235"   # код USD в справочнике ЦБ

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

def fetch_cbr_xml(date_from, date_to, val_code):
    url = (
        f"https://www.cbr.ru/scripts/XML_dynamic.asp"
        f"?date_req1={date_from}&date_req2={date_to}&VAL_NM_RQ={val_code}"
    )
    with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
        return r.read()

def parse_cbr_xml(raw_xml):
    root = ET.fromstring(raw_xml)
    rows = []
    for record in root.findall("Record"):
        date_str = record.attrib["Date"]           # формат DD/MM/YYYY
        value    = record.find("Value").text.replace(",", ".")
        nominal  = record.find("Nominal").text
        rows.append({
            "date":   pd.to_datetime(date_str, dayfirst=True),
            "usdrub": float(value) / float(nominal),
        })
    return pd.DataFrame(rows)

print(f"Загружаю USD/RUB с ЦБ РФ ({DATE_FROM} → {DATE_TO})...")
raw = fetch_cbr_xml(DATE_FROM, DATE_TO, USD_CODE)
df  = parse_cbr_xml(raw)
df  = df.sort_values("date").reset_index(drop=True)

print(f"Загружено: {len(df)} строк")
print(f"Период: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
print(f"Пример:\n{df.tail()}")

df.to_csv(OUT_CSV, index=False)
print(f"\nСохранено → {OUT_CSV}")
