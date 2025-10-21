import requests
import time
# insert your real key here!
access_key = "ac541c17-d096-49ad-b17a-35f90e1df72a"

headers = {
    "X-Yandex-Weather-Key": access_key
}

query = """{
  weatherByPoint(request: { lat:44.828683, lon: 38.316646 }) {
    forecast {
      days(limit: 1) {
        hours {
          time
          humidity
          pressure
          temperature
          windAngle
          windDirection
          windSpeed
        }
      }
    }
  }
}"""

response = requests.post('https://api.weather.yandex.ru/graphql/query', headers=headers, json={'query': query})
while True:
  print(response.content)
  with open('yameteo.txt', 'a', encoding='utf-8') as file:
    file.write(time.ctime()+","+str(response.content) + '\n')
  print("сон")
  time.sleep(60*30)