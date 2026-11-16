"""Download 10-K filings from SEC EDGAR and convert them to plain text + a metadata sidecar.

    python -m kgrag.corpus.sec --out data/corpus --tickers NVDA AMD INTC QCOM AVGO --years 2

SEC fair-access rules: a descriptive User-Agent with a contact address is mandatory
(set SEC_USER_AGENT in .env) and requests are throttled to < 10/s.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from kgrag.config import get_settings
from kgrag.log import get_logger

log = get_logger(__name__)

# Default corpus: five semiconductor issuers (same industry => dense supplier/competitor graph).
CIKS: dict[str, tuple[int, str]] = {
    "NVDA": (1045810, "NVIDIA Corporation"),
    "AMD": (2488, "Advanced Micro Devices, Inc."),
    "INTC": (50863, "Intel Corporation"),
    "QCOM": (804328, "QUALCOMM Incorporated"),
    "AVGO": (1730168, "Broadcom Inc."),
}


def _client(user_agent: str) -> httpx.Client:
    if not user_agent or "@" not in user_agent:
        raise SystemExit(
            "SEC_USER_AGENT must be set in .env to '<name or org> <contact email>' "
            "(SEC fair-access policy). Nothing was downloaded."
        )
    return httpx.Client(headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
                        timeout=60, follow_redirects=True)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    # Hidden XBRL blocks
    for tag in soup.select('[style*="display:none"]'):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        if re.fullmatch(r"\d{1,3}|Table of Contents", ln):  # page numbers / running headers
            continue
        out.append(ln)
    return "\n".join(out).strip()


def fetch(out_dir: Path, tickers: list[str], years: int, user_agent: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with _client(user_agent) as c:
        for ticker in tickers:
            cik, name = CIKS[ticker]
            subs = c.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
            recent = subs["filings"]["recent"]
            rows = [i for i, f in enumerate(recent["form"]) if f == "10-K"][:years]
            for i in rows:
                acc = recent["accessionNumber"][i]
                doc = recent["primaryDocument"][i]
                filed = recent["filingDate"][i]
                period = recent["reportDate"][i]
                fy = period[:4]
                stem = f"{ticker}_10K_{fy}"
                txt_path = out_dir / f"{stem}.txt"
                if txt_path.exists():
                    log.info("sec.skip_existing", file=str(txt_path))
                    written.append(txt_path)
                    continue
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
                log.info("sec.download", ticker=ticker, url=url)
                html = c.get(url).text
                time.sleep(0.25)
                txt_path.write_text(html_to_text(html), encoding="utf-8")
                meta = {"company": name, "ticker": ticker, "cik": cik, "form": "10-K",
                        "filing_date": filed, "period_of_report": period, "fiscal_year": fy,
                        "accession": acc, "source_url": url}
                (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                written.append(txt_path)
            time.sleep(0.25)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/corpus"))
    p.add_argument("--tickers", nargs="+", default=list(CIKS))
    p.add_argument("--years", type=int, default=1, help="most recent N 10-Ks per company")
    a = p.parse_args()
    files = fetch(a.out, a.tickers, a.years, get_settings().sec_user_agent)
    print(f"{len(files)} filings in {a.out}")


if __name__ == "__main__":
    main()
