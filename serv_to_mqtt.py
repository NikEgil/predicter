import json
import time
import paho.mqtt.client as mqtt
import logging
from datetime import date, timedelta
from urllib.parse import urlencode, urlparse
import requests
import traceback


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


def generate_past_dates(days: int):
    return (date.today() - timedelta(days=days)).isoformat()


def fetch_sensor_data(start_date: str):
    base_url = "http://93.183.71.105:8802/api/v1/module-data"
    params = {
        "sensor_ids": ['00000005', '00000007', '00000010'],
        "start_date": start_date,
        "end_date": "2027-12-23T00:00:00",
    }
    # urlencode с параметром doseq=True корректно обрабатывает списки
    query_string = urlencode(params, doseq=True)
    url = f"{base_url}?{query_string}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def transform_data(data):
    return {
        "sens": data["sens"],
        "port_1": {
            "hum_air": data["meteo"]["humidity"],
            "pressure_hpa": data["meteo"]["pressure_hpa"],
            "pressure_mmhg": data["meteo"]["pressure_mmhg"],
            "wind_direction": data["meteo"]["wind_direction"],
            "wind_speed": data["meteo"]["wind_speed"],
            "wind_gust": data["meteo"]["wind_gust"],
            "uv_intensity": data["meteo"]["uv_intensity"],
            "uv_index": data["meteo"]["uv_index"],
            "illuminance": data["meteo"]["illuminance"],
            "rainfall": data["meteo"]["rainfall"],
            "temp_air": data["meteo"]["temperature"],
            "address": "36"
        },
        "port_2": {
            "hum_soil_10": data["soil"]["moisture"],
            "temp_soil_10": data["soil"]["temperature"],
            "address": "6"
        },
    }

id_pairs={
    '5':'00000005',
    '7':'00000007',
    '10':'00000010'
}

def convert(jsons):
    q = []
    for i in range(len(jsons)):
        id =jsons[i]["sens"]["id_s"]
        if id in ["7", "5", "10"]:
            q.append(transform_data(jsons[i]))
            q[-1]['sens']["id_s"] =id_pairs[id]
            
    print(len(q))
    return q


def get_last_time(shift=0):
    pac = []
    shift = shift
    while True:
        d = generate_past_dates(shift)
        pac = fetch_sensor_data(str(d) + "T00:00:00")
        # print(pac)
        if len(pac) != 0:
            break
        else:
            shift += 1
            time.sleep(0.1)
        print(d, shift)
    if len(pac) != 0:
        return pac[-1]["ts"]
    else :return False


def sender(data_list):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Настройки MQTT
    BROKER = "93.183.71.105"
    PORT = 1883
    DEVICE_SERIAL = "00000001"
    USERNAME = DEVICE_SERIAL
    PASSWORD = "A1B2CA3D4E56"

    MQTT_TOPIC_DATA = f"mqtt/devices/{DEVICE_SERIAL}/data"

    # Callback
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info("Connected to MQTT Broker")
        else:
            logger.error(f"Failed to connect, reason_code={reason_code}")

    def on_publish(client, userdata, mid, reason_code, properties):
        logger.info(f"Message {mid} published with reason code {reason_code}")

    # Создаём клиент и подключаемся
    client = mqtt.Client(client_id=DEVICE_SERIAL,callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    # if USERNAME:
    client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect = on_connect
    client.on_publish = on_publish

    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()


    # Публикуем каждую запись по очереди
    for entry in data_list:
        payload = json.dumps(entry)
        client.publish(MQTT_TOPIC_DATA, payload=payload, qos=1)
        logger.info(f"Published: {payload}")
        time.sleep(0.1)  # пауза между публикациями

    client.loop_stop()
    client.disconnect()
    logger.info("Disconnected from MQTT")

def write_jsonl(data_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

while True:
    try:
        last_time = get_last_time(0)
        # last_time='05:47:22 09.08'
        if last_time!=False:
            print(last_time)
            lt = last_time[-8:] + " " + last_time[8:10] + "." + last_time[5:7]
            # lt='05:47:22 08.08'
            print(lt)
            jsons = get_jsons(lt)
            if jsons !=False:
                # print(jsons[-3:])
                data = convert(jsons)
                print('всего ',len(data))
                unique = []
                for x in data:
                    if x not in unique:
                        unique.append(x)
                print('уникальных ',len(unique))
                write_jsonl(unique,'out.json')
                # sender(unique)
    except Exception as e:
        print("Произошла ошибка:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        print("Полный traceback:")
        traceback.print_exc()
    finally:
        print("сон")
        time.sleep(120)
