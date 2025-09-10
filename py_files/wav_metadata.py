import re, datetime
from pathlib import Path

EXCEL_EPOCH = datetime.datetime(1899, 12, 30)  # Excel (Windows) 1900 date system

def excel_serial_to_dt(serial: float) -> datetime.datetime:
    return EXCEL_EPOCH + datetime.timedelta(days=float(serial))

def timestamp_from_name(name: str):
    m = re.search(r"(\d{5}\.\d+)", name)  # grabs 45764.68179121
    if not m:
        return None
    return excel_serial_to_dt(float(m.group(1)))

# Example
fn = "R08_45765.933130_4_18_0_15_33.wav"
print(timestamp_from_name(fn))  # 2025-04-17 16:21:46.760544
