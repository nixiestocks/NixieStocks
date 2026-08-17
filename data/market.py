import yfinance as yf


MARKETS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NASDAQ": "^IXIC",
}


def get_market_overview():

    result = {}

    for name, ticker in MARKETS.items():

        try:

            df = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                raise Exception("No data")

            latest = float(df["Close"].iloc[-1])
            previous = float(df["Close"].iloc[-2])

            change = ((latest - previous) / previous) * 100

            if ticker in ["^NSEI", "^BSESN", "^NSEBANK"]:
                price = f"₹ {latest:,.2f}"
            else:
                price = f"${latest:,.2f}"

            result[name] = {
                "price": price,
                "change": round(change, 2),
            }

        except Exception:

            result[name] = {
                "price": "N/A",
                "change": 0.00,
            }

    return result