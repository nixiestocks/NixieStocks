import json
import urllib.parse
import urllib.request
from functools import lru_cache
import yfinance as yf


def clean_text(value):
    return "" if value is None else str(value).strip()


def normalize_text(value):
    return clean_text(value).lower().replace("&", "and")


def is_equity(quote):
    quote_type = clean_text(quote.get("quoteType") or quote.get("typeDisp")).upper()
    return quote_type in {"EQUITY", "STOCK"}


def get_company_name(quote):
    return clean_text(quote.get("longname") or quote.get("shortname") or quote.get("displayName") or quote.get("name"))


def get_exchange(quote):
    return clean_text(quote.get("exchDisp") or quote.get("fullExchangeName") or quote.get("exchange"))


def search_with_yfinance(query):
    try:
        if not hasattr(yf, "Search"):
            return []
        try:
            search = yf.Search(query, max_results=12, news_count=0, lists_count=0, include_cb=True, include_nav_links=False, include_research=False, enable_fuzzy_query=True, recommended=12, timeout=8, raise_errors=False)
        except TypeError:
            search = yf.Search(query, max_results=12, news_count=0)
        return getattr(search, "quotes", []) or []
    except Exception as error:
        print("YFinance search error:", error)
        return []


def search_with_yahoo(query):
    try:
        encoded_query = urllib.parse.quote(query)
        url = "https://query1.finance.yahoo.com/v1/finance/search" f"?q={encoded_query}" "&quotesCount=12&newsCount=0&listsCount=0&enableFuzzyQuery=true"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("quotes", [])
    except Exception as error:
        print("Yahoo fallback search error:", error)
        return []


def rank_company(company_name, query, position):
    company = normalize_text(company_name)
    search = normalize_text(query)
    if company == search:
        return (0, position)
    if company.startswith(search):
        return (1, position)
    if any(word.startswith(search) for word in company.split()):
        return (2, position)
    if search in company:
        return (3, position)
    return (4, position)


@lru_cache(maxsize=300)
def cached_company_search(query):
    query = clean_text(query)
    if len(query) < 2:
        return tuple()
    quotes = search_with_yfinance(query) or search_with_yahoo(query)
    results = []
    used_symbols = set()
    for position, quote in enumerate(quotes):
        if not isinstance(quote, dict) or not is_equity(quote):
            continue
        symbol = clean_text(quote.get("symbol"))
        company_name = get_company_name(quote)
        exchange = get_exchange(quote)
        if not symbol or not company_name or symbol in used_symbols:
            continue
        used_symbols.add(symbol)
        results.append({"name": company_name, "symbol": symbol, "exchange": exchange, "rank": rank_company(company_name, query, position)})
    results.sort(key=lambda item: item["rank"])
    return tuple((item["name"], item["symbol"], item["exchange"]) for item in results[:8])


def search_companies(searchterm):
    searchterm = clean_text(searchterm)
    if len(searchterm) < 2:
        return []
    return [{"name": name, "symbol": symbol, "exchange": exchange} for name, symbol, exchange in cached_company_search(searchterm)]


def search_company_options(searchterm):
    options = []
    for company in search_companies(searchterm):
        name = company["name"]
        symbol = company["symbol"]
        exchange = company["exchange"]
        display_name = f"{name} — {exchange}" if exchange else name
        options.append((display_name, symbol))
    return options
