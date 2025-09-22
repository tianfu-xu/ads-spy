import sys
from pathlib import Path
from typing import Optional
from io import BytesIO

import csv
import pandas as pd
import requests
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT / 'ads_cleaner'))

from data_cleaner import (
    EMPTY_YOUTUBE_DATA,
    clean_dataframe,
    create_fetcher,
    create_youtube_fetcher,
)

APP_TITLE = "Ads Data Cleaner"
DEFAULT_FILE_NAME = "cleaned_ads_data.csv"
DEFAULT_YOUTUBE_KEY = "AIzaSyADKppatNo0fInKzRzuqgfC4fCAYDw6Lyg"

EMPTY_CREATIVE_DATA = {
    "headline": None,
    "long_headline": None,
    "description": None,
    "landing_page": None,
}


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV/XLS/XLSX into a DataFrame."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")

    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def build_fetchers(skip_creative: bool, skip_youtube: bool, youtube_key: Optional[str]):
    """Return creative and YouTube fetcher callables."""
    session = requests.Session()

    if skip_creative:
        def _creative_fetcher(_url: str):
            return {**EMPTY_CREATIVE_DATA}
    else:
        _creative_fetcher = create_fetcher(session)

    if skip_youtube:
        def _youtube_fetcher(_video_id: Optional[str]):
            return {**EMPTY_YOUTUBE_DATA}
    else:
        _youtube_fetcher = create_youtube_fetcher(session, youtube_key)

    return _creative_fetcher, _youtube_fetcher


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Upload a Google Ads Transparency export (CSV/XLSX). We will enrich it with creative"
        " details and, if enabled, YouTube metrics before returning a cleaned CSV."
    )

    with st.sidebar:
        st.header("Options")
        limit_rows = st.number_input("Limit rows (0 = all)", min_value=0, value=0, step=1)
        skip_creative = st.checkbox("Skip creative metadata fetch", value=False)
        use_youtube = st.checkbox("Fetch YouTube view/like counts", value=True)
        if use_youtube:
            st.caption("Using the built-in YouTube Data API key.")
        else:
            st.caption("YouTube metrics will be skipped.")
        youtube_key = DEFAULT_YOUTUBE_KEY if use_youtube else ""

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        st.write(f"Selected file: {uploaded_file.name}")

        run_col, csv_col, xlsx_col = st.columns([1, 1, 1])
        run_clicked = run_col.button("Run cleaner", type="primary", use_container_width=True)

        if run_clicked:
            try:
                df = read_uploaded_file(uploaded_file)
                if limit_rows:
                    df = df.head(int(limit_rows))

                skip_youtube = not use_youtube
                fetcher, youtube_fetcher = build_fetchers(skip_creative, skip_youtube, youtube_key)

                with st.spinner("Processing data..."):
                    cleaned_df = clean_dataframe(df, fetcher, youtube_fetcher)

                st.success("Cleaning finished. Preview of the first 50 rows:")
                st.dataframe(cleaned_df.head(50))

                export_df = cleaned_df.fillna("")
                csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
                excel_buffer = BytesIO()
                export_df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)

                download_name = f"cleaned_{uploaded_file.name}" if uploaded_file.name else DEFAULT_FILE_NAME

                csv_col.download_button(
                    "Download cleaned CSV",
                    data=csv_bytes,
                    file_name=download_name,
                    mime="text/csv",
                    use_container_width=True,
                )
                xlsx_col.download_button(
                    "Download cleaned Excel",
                    data=excel_buffer.getvalue(),
                    file_name=download_name.replace('.csv', '.xlsx'),
                    mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error("Processing failed. Please double-check the file format.")
                st.exception(exc)
    else:
        st.info("Upload a CSV or Excel file to get started.")


if __name__ == "__main__":
    main()
