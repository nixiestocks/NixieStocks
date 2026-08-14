import streamlit as st


def _value(value):

    if value is None:
        return "N/A"

    if isinstance(value, str):
        if value.strip() == "":
            return "N/A"
        return value

    return value


def dashboard_company(info):

    st.subheader("Company Information")

    left, right = st.columns(2)

    with left:
        st.write("**Company**", _value(info.get("name")))
        st.write("**Symbol**", _value(info.get("symbol")))
        st.write("**Sector**", _value(info.get("sector")))
        st.write("**Industry**", _value(info.get("industry")))

    with right:
        st.write("**Country**", _value(info.get("country")))
        st.write("**Website**", _value(info.get("website")))

        dividend = info.get("dividend_yield")

        if dividend is None:
            dividend = "N/A"
        else:
            try:
                dividend = f"{float(dividend) * 100:.2f}%"
            except Exception:
                dividend = "N/A"

        st.write("**Dividend Yield**", dividend)

    st.divider()
    st.subheader("Business Summary")

    summary = info.get("summary")

    if summary is None or str(summary).strip() == "":
        st.info("Business summary is not available for this stock.")
    else:
        st.write(summary)
