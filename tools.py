import json
from urllib.request import urlopen

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from rag import load_index

# --- Web Search ---

_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """旅行先の最新情報、フライト情報、イベント、天気などをWeb検索します。"""
    return _ddg.invoke(query)


# --- RAG Search ---

@tool
def search_travel_guides(query: str) -> str:
    """ローカルの旅行ガイドドキュメントから関連情報を検索します。東京・パリ・バンコクのガイドがあります。"""
    vectorstore = load_index()
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', '不明')}]\n{doc.page_content}" for doc in docs
    )


# --- Weather ---

@tool
def get_weather(city: str) -> str:
    """指定都市の現在の天気と今後3日間の天気予報を取得します。都市名を英語で指定してください（例: Tokyo, Paris, Bangkok）。"""
    # Geocoding
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=ja"
    with urlopen(geo_url) as resp:
        geo = json.loads(resp.read())

    if not geo.get("results"):
        return f"都市 '{city}' が見つかりませんでした。"

    loc = geo["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]
    name = loc.get("name", city)

    # Weather forecast
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weathercode,windspeed_10m"
        f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=auto&forecast_days=3"
    )
    with urlopen(weather_url) as resp:
        weather = json.loads(resp.read())

    wmo_codes = {
        0: "快晴", 1: "おおむね晴れ", 2: "一部曇り", 3: "曇り",
        45: "霧", 48: "着氷性の霧",
        51: "弱い霧雨", 53: "霧雨", 55: "強い霧雨",
        61: "弱い雨", 63: "雨", 65: "強い雨",
        71: "弱い雪", 73: "雪", 75: "強い雪",
        80: "弱いにわか雨", 81: "にわか雨", 82: "激しいにわか雨",
        95: "雷雨", 96: "雹を伴う雷雨", 99: "激しい雹を伴う雷雨",
    }

    current = weather["current"]
    lines = [
        f"## {name} の天気",
        f"**現在**: {wmo_codes.get(current['weathercode'], '不明')} "
        f"{current['temperature_2m']}°C / 風速 {current['windspeed_10m']}km/h",
        "",
        "**今後3日間の予報**:",
    ]

    daily = weather["daily"]
    for i in range(len(daily["time"])):
        code = wmo_codes.get(daily["weathercode"][i], "不明")
        lines.append(
            f"- {daily['time'][i]}: {code} "
            f"{daily['temperature_2m_min'][i]}〜{daily['temperature_2m_max'][i]}°C "
            f"(降水確率 {daily['precipitation_probability_max'][i]}%)"
        )

    return "\n".join(lines)


# --- Budget Estimator ---

@tool
def estimate_budget(destination: str, days: int, people: int, style: str = "中級") -> str:
    """旅行の概算予算を計算します。
    destination: 目的地（Tokyo, Paris, Bangkok）
    days: 旅行日数
    people: 旅行人数
    style: 旅行スタイル（バックパッカー, 中級, 高級）
    """
    budgets = {
        "Tokyo": {
            "currency": "円",
            "バックパッカー": {"宿泊": 4000, "食事": 2500, "交通": 1200, "観光": 1500},
            "中級": {"宿泊": 12000, "食事": 6500, "交通": 1800, "観光": 4000},
            "高級": {"宿泊": 35000, "食事": 20000, "交通": 5000, "観光": 10000},
        },
        "Paris": {
            "currency": "ユーロ",
            "バックパッカー": {"宿泊": 40, "食事": 28, "交通": 8, "観光": 15},
            "中級": {"宿泊": 150, "食事": 65, "交通": 12, "観光": 40},
            "高級": {"宿泊": 400, "食事": 180, "交通": 40, "観光": 100},
        },
        "Bangkok": {
            "currency": "バーツ",
            "バックパッカー": {"宿泊": 550, "食事": 400, "交通": 150, "観光": 350},
            "中級": {"宿泊": 2200, "食事": 1200, "交通": 400, "観光": 750},
            "高級": {"宿泊": 7000, "食事": 4000, "交通": 1500, "観光": 2500},
        },
    }

    city_data = budgets.get(destination)
    if not city_data:
        return f"'{destination}' の予算データがありません。対応都市: {', '.join(budgets.keys())}"

    if style not in city_data:
        return f"スタイル '{style}' は未対応です。選択肢: バックパッカー, 中級, 高級"

    currency = city_data["currency"]
    daily = city_data[style]
    daily_total = sum(daily.values())
    grand_total = daily_total * days * people

    lines = [
        f"## {destination} 旅行 概算予算",
        f"- スタイル: {style}",
        f"- 期間: {days}日間 / {people}人",
        "",
        "**1人1日あたりの内訳:**",
    ]
    for category, amount in daily.items():
        lines.append(f"- {category}: {amount:,} {currency}")
    lines.append(f"- **小計: {daily_total:,} {currency}/人/日**")
    lines.append("")
    lines.append(f"**総額: {grand_total:,} {currency}**（{days}日 x {people}人）")

    return "\n".join(lines)


def get_tools() -> list:
    """全ツールのリストを返す。"""
    return [web_search, search_travel_guides, get_weather, estimate_budget]
