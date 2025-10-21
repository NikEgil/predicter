import pandas as pd
import json
import time as time
from urllib.parse import urlencode, urlparse
import requests
import traceback


def read_json_lines(filename):
    data = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # пропускаем пустые строки
                    continue
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Ошибка парсинга JSON на строке {line_num}: {e}")
                    print(f"Содержимое строки: {line}")
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
    return data


def restore_series(observed):
    restored = []
    for i, y in enumerate(observed):
        if i == 0:
            if y + 25.5 <= 50:
                restored.append(y)  # пока по умолчанию
            else:
                restored.append(y)
        else:
            prev = restored[i - 1]
            x1 = y
            x2 = y + 25.5
            candidates = []
            if -10 <= x1 <= 50:
                candidates.append(x1)
            if -10 <= x2 <= 50:
                candidates.append(x2)
            best = min(candidates, key=lambda x: abs(x - prev))
            restored.append(best)
    return restored


def append_data(data: list, meteo_path: str, ugv_path: str):
    while True:
        try:
            meteo = pd.read_feather(meteo_path)
            ugv = pd.read_feather(ugv_path)
            break
        except:
            time.sleep(1)
    m_new = 0
    u_new = 0
    try:
        for i in range(len(data)):
            if data[i]["sens"]["id_s"] == "1" or data[i]["sens"]["id_s"] == "11":
                try:
                    print(data[i])
                    if "port_1" in data[i]:
                        t = []
                        h = []
                        for j in range(5):
                            t.append(data[i]["port_1"][j]["temperature"])
                            h.append(data[i]["port_1"][j]["moisture"])
                        sens = list(data[i]["sens"].values())
                        li = sens[0:3]
                        li.append(pd.to_datetime(sens[3] + ".2025", format="%H:%M:%S %d.%m.%Y"))
                        li.extend(t)
                        li.extend(h)
                        ugv.loc[len(ugv)] = li
                        u_new += 1
                except:
                    print('ops')
                    print(data[i])
                    break
            if (
                data[i]["sens"]["id_s"] == "5"
                or data[i]["sens"]["id_s"] == "7"
                or data[i]["sens"]["id_s"] == "10"
            ):
                if "meteo" in data[i] and "soil" in data[i]:
                    sens = list(data[i]["sens"].values())
                    li = sens[0:3]
                    li.append(pd.to_datetime(sens[3] + ".2025", format="%H:%M:%S %d.%m.%Y"))
                    li.extend(data[i]["meteo"].values())
                    li.extend(data[i]["soil"].values())
                    meteo.loc[len(meteo)] = li
                    m_new += 1
    except Exception as e:
        print("Полный traceback:")
        traceback.print_exc()

    ugv=ugv.sort_values(by='datetime', ascending=True).reset_index(drop=True)
    meteo=meteo.sort_values(by='datetime', ascending=True).reset_index(drop=True)

    for q in ["1", "11"]:
        ind = ugv[ugv["id_s"] == q].index
        for p in ["t1", "t2", "t3", "t4", "t5"]:
            temp = restore_series(ugv[ugv["id_s"] == q][p].to_numpy())
            ser = ugv[ugv["id_s"] != q][p]
            ugv[p] = ser.combine_first(pd.Series(temp, index=ind))

    while True:
        try:
            ugv.to_feather(ugv_path)
            meteo.to_feather(meteo_path)
            break
        except:
            time.sleep(1)
    print("добавлено meteo:", m_new, "  ugv:", u_new)


def get_jsons(date):
    base_url = "http://93.183.71.44/agro/api/v1/device-initial-list"
    params = {"from_datetime": date}

    # Полный URL с параметрами
    url = f"{base_url}?{urlencode(params)}"

    print("Отправка запроса на:", url)
    try:
        response = requests.get(url, timeout=10)
        
        # Проверяем статус ответа
        if response.status_code == 200:
            print("Успешный ответ:")
            print(len(response.json()), " строк")
            if len(response.json()) > 0:
                return response.json()
            else:
                return False
        else:

            print(f"Ошибка: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.RequestException as e:
        print("Ошибка при выполнении запроса:", e)


meteo_path = "meteo.feather"
ugv_path = "ugv.feather"


while True:
    try:
        ugv = pd.read_feather(ugv_path)
        meteo = pd.read_feather(meteo_path)
        if len(meteo)>0 and len(ugv)>0:
            lt = max(ugv["datetime"].max(), meteo["datetime"].max())
            lt = str(max(ugv["datetime"].max(), meteo["datetime"].max()))
            last_time = lt[11:] + " " + lt[8:10] + "." + lt[5:7]
        else:
            last_time='00:00:00 01.01'
        print(last_time)
        data = get_jsons(last_time)
        if data != False:
            append_data(data, meteo_path, ugv_path)
    except FileNotFoundError:
        meteo = pd.DataFrame(
            columns=[
                "id_s",
                "batt",
                "rssi",
                "datetime",
                "temp_air",
                "hum_air",
                "pres_hpa",
                "press_mm",
                "wind_dir",
                "wind_speed",
                "wind_gust",
                "uv_power",
                "uv_index",
                "ilum",
                "rain",
                "hum_soil",
                "temp_soil",
            ]
        )
        ugv = pd.DataFrame(
            columns=[
                "id_s",
                "batt",
                "rssi",
                "datetime",
                "t1",
                "t2",
                "t3",
                "t4",
                "t5",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
            ]
        )
        ugv.to_feather(ugv_path)
        meteo.to_feather(meteo_path)
        print("файлы созданы")


    except Exception as e:
        print("Произошла ошибка:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        print("Полный traceback:")
        traceback.print_exc()
    finally:
        print("сон")
        time.sleep(120)
