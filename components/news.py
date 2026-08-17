import streamlit as st
import yfinance as yf
from datetime import datetime


def _get_value(item, key, default=""):
    """
    Handles both normal Yahoo news format and newer nested format.
    """

    if not isinstance(item, dict):
        return default

    # Normal format
    if key in item:
        return item.get(key, default)

    # New nested format
    content = item.get("content", {})

    if isinstance(content, dict):
        return content.get(key, default)

    return default


def _get_link(item):
    """
    Extract article link from different Yahoo Finance formats.
    """

    if not isinstance(item, dict):
        return ""

    # Old/simple format
    link = item.get("link")

    if link:
        return link

    # New format
    content = item.get("content", {})

    if isinstance(content, dict):

        click_url = content.get("clickThroughUrl")

        if isinstance(click_url, dict):
            url = click_url.get("url")

            if url:
                return url

        canonical_url = content.get("canonicalUrl")

        if isinstance(canonical_url, dict):
            url = canonical_url.get("url")

            if url:
                return url

    return ""


def _get_publisher(item):

    publisher = _get_value(item, "publisher", "")

    if publisher:
        return publisher

    content = item.get("content", {})

    if isinstance(content, dict):

        provider = content.get("provider", {})

        if isinstance(provider, dict):
            return provider.get("displayName", "")

    return "Yahoo Finance"


def _get_date(item):

    timestamp = _get_value(item, "providerPublishTime", None)

    if timestamp:

        try:
            return datetime.fromtimestamp(timestamp).strftime(
                "%d %b %Y • %I:%M %p"
            )
        except Exception:
            pass

    content = item.get("content", {})

    if isinstance(content, dict):

        date_value = content.get("pubDate")

        if date_value:
            try:
                dt = datetime.fromisoformat(
                    date_value.replace("Z", "+00:00")
                )

                return dt.strftime(
                    "%d %b %Y • %I:%M %p"
                )

            except Exception:
                return str(date_value)

    return ""


def _load_news(symbol):

    news = []

    # -----------------------------------------
    # HOME PAGE / GENERAL MARKET NEWS
    # -----------------------------------------

    if not symbol or symbol in ["^GSPC", "^DJI", "^IXIC"]:

        try:

            search = yf.Search(
                "stock market",
                news_count=10,
                max_results=1
            )

            news = search.news or []

        except Exception:
            news = []

    # -----------------------------------------
    # STOCK-SPECIFIC NEWS
    # -----------------------------------------

    else:

        try:

            search = yf.Search(
                symbol,
                news_count=10,
                max_results=1
            )

            news = search.news or []

        except Exception:
            news = []

    # -----------------------------------------
    # FALLBACK
    # -----------------------------------------

    if not news:

        try:

            ticker = yf.Ticker(symbol)

            if hasattr(ticker, "get_news"):
                news = ticker.get_news(
                    count=10,
                    tab="news"
                )

            else:
                news = ticker.news

        except Exception:
            news = []

    return news


def market_news(symbol=None):

    st.subheader("Latest Market News")

    with st.spinner("Loading latest news..."):

        news = _load_news(symbol)

    if not news:

        st.info("No news available right now.")
        return

    shown = 0

    for article in news:

        if shown >= 8:
            break

        if not isinstance(article, dict):
            continue

        title = _get_value(
            article,
            "title",
            "Market News"
        )

        link = _get_link(article)

        publisher = _get_publisher(article)

        date = _get_date(article)

        if not title:
            continue

        # -------------------------------------
        # NEWS ARTICLE
        # -------------------------------------

        if link:

            st.markdown(
                f"### [{title}]({link})"
            )

        else:

            st.markdown(
                f"### {title}"
            )

        details = []

        if publisher:
            details.append(publisher)

        if date:
            details.append(date)

        if details:

            st.caption(
                " • ".join(details)
            )

        summary = _get_value(
            article,
            "summary",
            ""
        )

        if summary:

            if len(summary) > 300:
                summary = summary[:300] + "..."

            st.write(summary)

        st.divider()

        shown += 1

    if shown == 0:

        st.info(
            "No readable news articles were returned."
        )