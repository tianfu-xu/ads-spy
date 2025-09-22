from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import requests

warnings.filterwarnings("ignore", "invalid escape sequence", DeprecationWarning)

LOGGER = logging.getLogger("ads_cleaner")
GOOGLE_TEMPLATE_ANCHOR = "google_template_data"
AD_CLICK_PATTERN = re.compile(r"https://www\.googleadservices\.com/pagead/aclk\?[^'\" ]+")
GENERIC_URL_PATTERN = re.compile(r"https?://[^\s'\"\\]+")
TRACKING_SNIPPETS = ("google", "gstatic", "ytimg", "doubleclick", "ampproject")
UNQUOTED_KEY_PATTERN = re.compile(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:')
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

EMPTY_YOUTUBE_DATA = {
    "youtube_title": None,
    "youtube_published_at": None,
    "youtube_view_count": None,
    "youtube_like_count": None,
}

DEFAULT_YOUTUBE_KEY = "AIzaSyADKppatNo0fInKzRzuqgfC4fCAYDw6Lyg"









def quote_unquoted_keys(text: str) -> str:
    def repl(match):
        key = match.group(1)
        return f"'{key}':"
    return UNQUOTED_KEY_PATTERN.sub(repl, text)
def is_tracking_link(url: str) -> bool:
    lowered = url.lower()
    parsed = urlparse(lowered)
    host = parsed.netloc or lowered
    return any(snippet in host for snippet in TRACKING_SNIPPETS)

def normalize_text(value):



    """Try to repair strings that were decoded with latin-1 instead of UTF-8."""



    if not isinstance(value, str):



        return value



    current = value



    for _ in range(4):



        try:



            converted = current.encode('latin-1').decode('utf-8')



        except UnicodeEncodeError:



            break



        except UnicodeDecodeError:



            break



        if converted == current:



            break



        current = converted



    return current







def unescape_js_fragment(text: str, rounds: int = 2) -> str:
    """Decode JavaScript escape sequences like \\xNN or \\uNNNN."""
    result = text
    for _ in range(rounds):
        try:
            result = bytes(result, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            break
    return result


def extract_template_block(js_text: str) -> Optional[dict]:
    idx = js_text.find(GOOGLE_TEMPLATE_ANCHOR)
    if idx == -1:
        return None
    brace_start = js_text.find("{", idx)
    if brace_start == -1:
        return None

    depth = 0
    end = None
    for pos in range(brace_start, len(js_text)):
        ch = js_text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        return None

    snippet = js_text[brace_start:end]
    decoded = unescape_js_fragment(snippet, rounds=2)
    normalized = quote_unquoted_keys(decoded)
    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        LOGGER.debug("Failed to literal_eval creative block", exc_info=True)
        return None


def clean_url(value: str) -> Optional[str]:
    if not value:
        return None
    decoded = unescape_js_fragment(value, rounds=2)
    decoded = unquote(decoded)
    decoded = decoded.strip("'")
    return decoded if decoded else None



def extract_landing_page(js_text: str) -> Optional[str]:
    decoded = unescape_js_fragment(js_text, rounds=2)

    match = re.search(r"final_url:\s*'([^']+)'", decoded)
    if match:
        candidate = clean_url(match.group(1))
        if candidate and not is_tracking_link(candidate):
            return candidate

    direct_match = re.search(r"'(https?://[^']+)',\s*final_url", decoded)
    if direct_match:
        candidate = clean_url(direct_match.group(1))
        if candidate and not is_tracking_link(candidate):
            return candidate

    adurl_index = decoded.find("adurl=")
    if adurl_index != -1:
        start_idx = adurl_index + len("adurl=")
        if start_idx < len(decoded) and decoded[start_idx] in ("'", '"'):
            start_idx += 1
        end_idx = decoded.find("'", start_idx)
        if end_idx == -1:
            end_idx = decoded.find('"', start_idx)
        if end_idx == -1:
            end_idx = len(decoded)
        landing_raw = decoded[start_idx:end_idx]
        landing = clean_url(landing_raw.strip(" '"))
        if landing and not is_tracking_link(landing):
            return landing

    dest_match = re.search(r"destination_url:\s*'([^']+)'", decoded)
    if dest_match:
        candidate = clean_url(dest_match.group(1))
        if candidate and not is_tracking_link(candidate):
            return candidate

    landing_match = re.search(r"landing_page_url:\s*'([^']+)'", decoded)
    if landing_match:
        candidate = clean_url(landing_match.group(1))
        if candidate and not is_tracking_link(candidate):
            return candidate

    for raw_url in GENERIC_URL_PATTERN.findall(decoded):
        if is_tracking_link(raw_url):
            continue
        cleaned = clean_url(raw_url)
        if cleaned and not is_tracking_link(cleaned):
            return cleaned

    return None

def extract_video_id(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        hostname = (parsed.hostname or '').lower()
        if hostname.endswith('youtu.be'):
            candidate = parsed.path.lstrip('/')
            return candidate or None
        query_id = parse_qs(parsed.query).get('v', [''])[0]
        if query_id:
            return query_id
        path_parts = [segment for segment in parsed.path.split('/') if segment]
        if path_parts:
            return path_parts[-1]
        return None
    return raw


def fetch_youtube_metadata(session: requests.Session, api_key: str, video_id: str, timeout: float = 10.0) -> Dict[str, Optional[str]]:
    params = {
        'part': 'snippet,statistics',
        'id': video_id,
        'key': api_key,
    }
    try:
        response = session.get(YOUTUBE_API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        LOGGER.warning('Failed to fetch YouTube metadata for %s', video_id, exc_info=True)
        return {**EMPTY_YOUTUBE_DATA}
    except ValueError:
        LOGGER.warning('Invalid JSON returned for YouTube video %s', video_id)
        return {**EMPTY_YOUTUBE_DATA}

    items = payload.get('items') or []
    if not items:
        return {**EMPTY_YOUTUBE_DATA}

    item = items[0]
    snippet = item.get('snippet') or {}
    statistics = item.get('statistics') or {}

    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        'youtube_title': snippet.get('title'),
        'youtube_published_at': snippet.get('publishedAt'),
        'youtube_view_count': _to_int(statistics.get('viewCount')),
        'youtube_like_count': _to_int(statistics.get('likeCount')),
    }


def create_youtube_fetcher(session: requests.Session, api_key: Optional[str]):
    if not api_key:
        return lambda _video_id: {**EMPTY_YOUTUBE_DATA}

    @lru_cache(maxsize=256)
    def _fetch(video_id: Optional[str]) -> Dict[str, Optional[str]]:
        if not video_id:
            return {**EMPTY_YOUTUBE_DATA}
        return fetch_youtube_metadata(session, api_key, video_id)

    return _fetch
def extract_from_image_url(session: requests.Session, url: str, timeout: float = 12.0) -> Dict[str, Optional[str]]:
    empty = {
        "headline": None,
        "long_headline": None,
        "description": None,
        "landing_page": None,
    }
    if not isinstance(url, str) or not url:
        return empty

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.warning("Failed to fetch %s", url, exc_info=True)
        return empty

    text = response.text
    block = extract_template_block(text)
    headline = long_headline = description = None
    if block and isinstance(block, dict):
        ad_list = block.get("adData")
        if isinstance(ad_list, Iterable):
            first = next(iter(ad_list), None)
            if isinstance(first, dict):
                headline = first.get("headline")
                long_headline = first.get("longHeadline") or first.get("long_headline")
                description = first.get("description")

    landing_page = extract_landing_page(text)

    headline = normalize_text(headline)
    long_headline = normalize_text(long_headline)
    description = normalize_text(description)
    landing_page = normalize_text(landing_page)

    return {
        "headline": headline,
        "long_headline": long_headline,
        "description": description,
        "landing_page": landing_page,
    }


def create_fetcher(session: requests.Session):
    @lru_cache(maxsize=256)
    def _fetch(url: str) -> Dict[str, Optional[str]]:
        return extract_from_image_url(session, url)

    return _fetch


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def clean_dataframe(df: pd.DataFrame, fetcher, youtube_fetcher) -> pd.DataFrame:
    required_columns = [
        "first_shown_date",
        "last_shown_date",
        "regions_shown",
        "youtube_video_id",
        "transparency_url",
        "image_url",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    base = df[["first_shown_date", "last_shown_date", "regions_shown", "youtube_video_id", "transparency_url"]].copy()

    details = df["image_url"].apply(lambda x: fetcher(x) if isinstance(x, str) and x else {})
    details_df = pd.DataFrame(details.tolist(), index=df.index)

    youtube_meta = df["youtube_video_id"].apply(lambda x: youtube_fetcher(extract_video_id(x)))
    youtube_df = pd.DataFrame(youtube_meta.tolist(), index=df.index)
    if not youtube_df.empty:
        if 'youtube_title' in youtube_df.columns:
            youtube_df["youtube_title"] = youtube_df["youtube_title"].apply(normalize_text)

    result = pd.concat([base, details_df, youtube_df], axis=1)

    desired_order = [
        "youtube_video_id",
        "youtube_view_count",
        "youtube_like_count",
        "first_shown_date",
        "last_shown_date",
        "youtube_published_at",
        "youtube_title",
        "regions_shown",
        "headline",
        "long_headline",
        "description",
        "landing_page",
        "transparency_url",
    ]
    ordered_cols = [col for col in desired_order if col in result.columns]
    remaining_cols = [col for col in result.columns if col not in ordered_cols]
    result = result[ordered_cols + remaining_cols]
    return result


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ads export and extract creative metadata.")
    parser.add_argument("input_path", type=Path, help="Path to the source CSV or Excel file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination CSV path (defaults to <input-name>_cleaned.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of rows to process (useful for quick tests).",
    )

    parser.add_argument(
        "--youtube-key",
        dest="youtube_key",
        help="YouTube Data API v3 key (or set env YOUTUBE_API_KEY).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip remote lookups (headline data will be empty).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    input_path: Path = args.input_path
    if not input_path.exists():
        LOGGER.error("Input file does not exist: %s", input_path)
        return 1

    df = load_dataframe(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.no_fetch:
        fetcher = lambda _: {
            "headline": None,
            "long_headline": None,
            "description": None,
            "landing_page": None,
        }
        youtube_fetcher = lambda _: {**EMPTY_YOUTUBE_DATA}
    else:
        session = requests.Session()
        fetcher = create_fetcher(session)
        youtube_api_key = args.youtube_key or os.getenv("YOUTUBE_API_KEY") or DEFAULT_YOUTUBE_KEY
        youtube_fetcher = create_youtube_fetcher(session, youtube_api_key)

    cleaned = clean_dataframe(df, fetcher, youtube_fetcher)

    output_path = args.output or input_path.with_name(f"{input_path.stem}_cleaned.csv")
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote cleaned dataset to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






