import httpx
from ..logger_config import logger

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> human-readable conditions
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def get_weather(city: str):
    logger.info("get_weather called")
    logger.debug(f"Input -> city: {city}")

    try:
        # transport.retries=3 auto-retries on transient TCP/TLS errors
        # (ConnectError, DNS hiccups, handshake timeouts) — does NOT retry on 4xx/5xx.
        transport = httpx.AsyncHTTPTransport(retries=3)
        async with httpx.AsyncClient(timeout=15, transport=transport) as client:
            # Step 1: Geocode the city name -> latitude/longitude
            geo_res = await client.get(
                GEOCODING_URL,
                params={"name": city, "count": 1, "format": "json"},
            )
            geo_res.raise_for_status()
            geo_data = geo_res.json()

            results = geo_data.get("results")
            if not results:
                logger.warning(f"City not found: {city}")
                return {"error": f"City '{city}' not found"}

            location = results[0]
            lat = location["latitude"]
            lon = location["longitude"]
            resolved_name = location.get("name", city)
            country = location.get("country", "")

            # Step 2: Fetch current weather for those coordinates
            weather_res = await client.get(
                WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
            )
            weather_res.raise_for_status()
            weather_data = weather_res.json()

            current = weather_data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            wind = current.get("wind_speed_10m")
            code = current.get("weather_code")
            condition = WEATHER_CODES.get(code, "Unknown")

            result = {
                "city": resolved_name,
                "country": country,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "wind_kmh": wind,
                "condition": condition,
            }

            logger.info(
                f"Weather fetched: {resolved_name}, {country} -> {temp}°C, {condition}"
            )
            return result

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Weather API HTTP error: {e.response.status_code}", exc_info=True
        )
        return {"error": f"Weather API returned status {e.response.status_code}"}
    except httpx.RequestError:
        logger.error("Weather API request failed", exc_info=True)
        return {"error": "Failed to reach weather API"}
    except Exception as e:
        logger.error("Error while fetching weather", exc_info=True)
        return {"error": str(e)}