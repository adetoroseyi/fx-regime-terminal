"""
Vercel serverless function — OANDA live prices proxy.

Fetches current bid/ask for FX pairs from OANDA v20 REST API.
Keeps the API key server-side (Vercel env vars).

Environment variables required:
    OANDA_API_KEY: Your OANDA API token
    OANDA_ACCOUNT_ID: Your OANDA account ID
    OANDA_ENV: 'practice' (default) or 'live'
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError


OANDA_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

# OANDA instrument names for our pairs
PAIR_INSTRUMENTS = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF",
}


def get_prices():
    api_key = os.environ.get("OANDA_API_KEY", "")
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    env = os.environ.get("OANDA_ENV", "practice")

    if not api_key or not account_id:
        return {"error": "OANDA credentials not configured", "prices": {}}

    host = OANDA_HOSTS.get(env, OANDA_HOSTS["practice"])
    instruments = ",".join(PAIR_INSTRUMENTS.values())
    url = f"{host}/v3/accounts/{account_id}/pricing?instruments={instruments}"

    req = Request(url)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        return {"error": f"OANDA API error: {str(e)}", "prices": {}}

    # Parse response into our format
    prices = {}
    for price_data in data.get("prices", []):
        instrument = price_data.get("instrument", "")
        # Reverse lookup pair name
        pair = None
        for p, inst in PAIR_INSTRUMENTS.items():
            if inst == instrument:
                pair = p
                break
        if not pair:
            continue

        bids = price_data.get("bids", [])
        asks = price_data.get("asks", [])
        bid = float(bids[0]["price"]) if bids else 0
        ask = float(asks[0]["price"]) if asks else 0
        spread = round((ask - bid) * (100000 if "JPY" not in pair else 1000), 1)

        prices[pair] = {
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "mid": round((bid + ask) / 2, 5 if "JPY" not in pair else 3),
            "tradeable": price_data.get("tradeable", False),
            "time": price_data.get("time", ""),
        }

    return {"prices": prices, "timestamp": data.get("time", "")}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        result = get_prices()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
