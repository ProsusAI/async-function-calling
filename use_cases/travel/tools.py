import time
import random
import json

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    print(f"    ⚡ [get_weather] Fetching weather for {city}... (instant)")
    data = {
        "mumbai":    "32°C, humid and partly cloudy",
        "amsterdam": "9°C, overcast with light drizzle",
        "tokyo":     "18°C, rainy",
        "paris":     "14°C, sunny",
    }
    return data.get(city.lower(), f"25°C, clear skies in {city}")


def get_hotels(city: str) -> str:
    delay = random.uniform(15, 30)
    print(f"    ⏳ [get_hotels] Fetching hotels for {city}... ({delay:.1f}s)")
    time.sleep(delay)

    hotels = {
        "mumbai": [
            {"name": "The Taj Mahal Palace", "area": "Colaba",        "cost_per_night": "$280"},
            {"name": "Oberoi Trident",        "area": "Nariman Point", "cost_per_night": "$220"},
            {"name": "ITC Grand Central",     "area": "Parel",         "cost_per_night": "$160"},
            {"name": "Hotel Sea Princess",    "area": "Juhu",          "cost_per_night": "$110"},
            {"name": "Residency Hotel",       "area": "Fort",          "cost_per_night": "$70"},
        ],
        "amsterdam": [
            {"name": "Hotel V Nesplein",      "area": "City Centre",   "cost_per_night": "$210"},
            {"name": "The Dylan",             "area": "Jordaan",       "cost_per_night": "$350"},
            {"name": "citizenM Amsterdam",    "area": "Museum Quarter","cost_per_night": "$160"},
            {"name": "Generator Amsterdam",   "area": "East",          "cost_per_night": "$60"},
            {"name": "INK Hotel",             "area": "De Pijp",       "cost_per_night": "$130"},
        ],
    }

    results = hotels.get(city.lower())
    if not results:
        return f"No hotel data available for {city}."
    lines = [f"Hotels in {city.title()}:"]
    for h in results:
        lines.append(f"  • {h['name']} — {h['area']} — {h['cost_per_night']}/night")
    return "\n".join(lines)


def get_activities(city: str, tag: str = None) -> str:
    """
    city: mumbai | amsterdam
    tag:  couple | family | solo  (optional — returns all if omitted)
    """
    delay = random.uniform(15, 30)
    print(f"    ⏳ [get_activities] Fetching activities for {city} (tag={tag})... ({delay:.1f}s)")
    time.sleep(delay)

    activities = {
        "mumbai": [
            {"name": "Gateway of India sunset walk",  "area": "Colaba",        "tags": ["couple", "solo"]},
            {"name": "Dharavi slum tour",             "area": "Dharavi",       "tags": ["solo", "family"]},
            {"name": "Juhu Beach cricket & snacks",   "area": "Juhu",          "tags": ["family", "solo"]},
            {"name": "Bollywood studio tour",         "area": "Goregaon",      "tags": ["family", "couple"]},
            {"name": "Elephanta Caves ferry trip",    "area": "Harbour",       "tags": ["family", "couple", "solo"]},
            {"name": "Bandra street food walk",       "area": "Bandra",        "tags": ["couple", "solo"]},
            {"name": "Marine Drive evening stroll",   "area": "Nariman Point", "tags": ["couple", "solo"]},
        ],
        "amsterdam": [
            {"name": "Canal boat tour",               "area": "City Centre",   "tags": ["couple", "family"]},
            {"name": "Rijksmuseum visit",             "area": "Museum Quarter","tags": ["family", "solo", "couple"]},
            {"name": "Vondelpark picnic",             "area": "Museum Quarter","tags": ["couple", "family"]},
            {"name": "Anne Frank House",              "area": "Jordaan",       "tags": ["solo", "family"]},
            {"name": "Heineken Experience",           "area": "De Pijp",       "tags": ["couple", "solo"]},
            {"name": "Cycling through Jordaan",       "area": "Jordaan",       "tags": ["couple", "solo"]},
            {"name": "NEMO Science Museum",           "area": "Waterfront",    "tags": ["family"]},
        ],
    }

    city_activities = activities.get(city.lower())
    if not city_activities:
        return f"No activity data for {city}."

    if tag:
        city_activities = [a for a in city_activities if tag.lower() in a["tags"]]
        if not city_activities:
            return f"No activities tagged '{tag}' found in {city}."

    label = f"Activities in {city.title()}" + (f" (tag: {tag})" if tag else "")
    lines = [label + ":"]
    for a in city_activities:
        lines.append(f"  • {a['name']} — {a['area']} [{', '.join(a['tags'])}]")
    return "\n".join(lines)


def get_flights(origin: str, destination: str) -> str:
    delay = random.uniform(8, 25)
    print(f"    ⏳ [get_flights] Searching flights {origin}→{destination}... ({delay:.1f}s)")
    time.sleep(delay)

    flights = {
        ("tokyo", "mumbai"): [
            {"airline": "Air India",   "duration": "10h 05m", "price": "$420", "stops": "Nonstop"},
            {"airline": "IndiGo",      "duration": "11h 20m", "price": "$310", "stops": "1 stop (Delhi)"},
            {"airline": "JAL",         "duration": "10h 15m", "price": "$530", "stops": "Nonstop"},
        ],
        ("tokyo", "amsterdam"): [
            {"airline": "KLM",         "duration": "12h 40m", "price": "$680", "stops": "Nonstop"},
            {"airline": "Lufthansa",   "duration": "15h 10m", "price": "$510", "stops": "1 stop (Frankfurt)"},
            {"airline": "ANA",         "duration": "13h 00m", "price": "$720", "stops": "Nonstop"},
        ],
    }

    key = (origin.lower(), destination.lower())
    options = flights.get(key)
    if not options:
        return f"No flight data for {origin} → {destination}."

    lines = [f"Flights from {origin.title()} to {destination.title()}:"]
    for f in options:
        lines.append(f"  • {f['airline']} — {f['duration']} — {f['price']} — {f['stops']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SLOW_TOOLS: set[str] = {"get_hotels", "get_activities", "get_flights"}

TOOL_FUNCTIONS: dict = {
    "get_weather":    lambda args: get_weather(args["city"]),
    "get_hotels":     lambda args: get_hotels(args["city"]),
    "get_activities": lambda args: get_activities(args["city"], args.get("tag")),
    "get_flights":    lambda args: get_flights(args["origin"], args["destination"]),
}

# ---------------------------------------------------------------------------
# OpenAI schemas (no await_job — the framework appends it automatically)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotels",
            "description": "Get a list of hotels with area and cost per night. Supports: mumbai, amsterdam.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "mumbai or amsterdam"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activities",
            "description": "Get activities for a city, optionally filtered by tag (couple, family, solo). Supports: mumbai, amsterdam.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "mumbai or amsterdam"},
                    "tag":  {"type": "string", "enum": ["couple", "family", "solo"], "description": "Optional filter"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flights",
            "description": "Get flight options. Supports routes: tokyo→mumbai, tokyo→amsterdam.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin":      {"type": "string", "description": "Departure city, e.g. tokyo"},
                    "destination": {"type": "string", "description": "Arrival city, e.g. mumbai or amsterdam"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
]
