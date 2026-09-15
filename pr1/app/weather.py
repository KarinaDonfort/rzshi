"""Модуль інтеграції із зовнішнім API погоди.

Єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# (connect, read): не чекаємо на мережу нескінченно
TIMEOUT = (3, 10)


class WeatherError(Exception):
    """Помилка отримання погоди, зрозуміла веб-рівню.

    status_code — рекомендований HTTP-статус, який веб-рівень має
    повернути клієнту. Так веб-рівень не думає про причини, лише
    відображає їх.
    """

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _get_json(url: str, params: dict) -> dict:
    """Спільна логіка HTTP-виклику: параметри, timeout, коди, JSON."""
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.Timeout as exc:
        raise WeatherError(
            "Сервіс погоди не відповів вчасно. Спробуйте пізніше.",
            status_code=504,
        ) from exc
    except requests.ConnectionError as exc:
        raise WeatherError(
            "Немає зв'язку із сервісом погоди. Перевірте інтернет.",
            status_code=503,
        ) from exc
    except requests.RequestException as exc:
        raise WeatherError(
            f"Не вдалося звернутися до сервісу погоди ({type(exc).__name__}).",
            status_code=502,
        ) from exc

    # 4xx — помилка на нашому боці, 5xx — проблема сервісу
    if 400 <= response.status_code < 500:
        raise WeatherError(
            f"Сервіс відхилив запит (HTTP {response.status_code}).",
            status_code=response.status_code,
        )
    if response.status_code >= 500:
        raise WeatherError(
            f"Сервіс погоди тимчасово недоступний (HTTP {response.status_code}).",
            status_code=502,
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise WeatherError(
            "Сервіс повернув відповідь у неочікуваному форматі.",
            status_code=502,
        ) from exc

    if not isinstance(data, dict):
        raise WeatherError(
            "Сервіс повернув відповідь у неочікуваному форматі.",
            status_code=502,
        )

    return data


def find_city(name: str) -> dict:
    """Знайти координати міста за назвою."""
    name = (name or "").strip()
    if not name:
        raise WeatherError("Введіть назву міста.", status_code=400)

    data = _get_json(
        GEOCODING_URL,
        params={
            "name": name,
            "count": 1,
            "language": "uk",
            "format": "json",
        },
    )

    # Коли місто не знайдено — ключа "results" немає взагалі
    results = data.get("results")
    if not results:
        raise WeatherError(f"Місто «{name}» не знайдено.", status_code=404)

    top = results[0]
    try:
        return {
            "name": top["name"],
            "latitude": float(top["latitude"]),
            "longitude": float(top["longitude"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(
            "Відповідь геокодера має неочікувану структуру.",
            status_code=502,
        ) from exc


def get_current_weather(city: str) -> dict:
    """Повернути поточну температуру і швидкість вітру."""
    location = find_city(city)

    data = _get_json(
        FORECAST_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        },
    )

    current = data.get("current")
    units = data.get("current_units", {})

    if not isinstance(current, dict):
        raise WeatherError(
            "У відповіді прогнозу немає блоку 'current'.",
            status_code=502,
        )

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    if temperature is None or wind_speed is None:
        raise WeatherError(
            "У відповіді прогнозу немає потрібних полів.",
            status_code=502,
        )

    return {
        "city": location["name"],
        "temperature": temperature,
        "temperature_unit": units.get("temperature_2m", "°C"),
        "wind_speed": wind_speed,
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
    }