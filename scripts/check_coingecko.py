import requests


def check_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": "false",
    }

    try:
        print(f"Requesting {url}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        print(f"Success! Retrieved {len(data)} items.")
        for i, item in enumerate(data, 1):
            print(f"{i}. {item['symbol'].upper()} - Cap: ${item['market_cap']:,.0f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_coingecko()
