"""
download_documents.py

Downloads the reference document corpus for the agentic document
intelligence project: BMW / Mercedes-Benz annual reports, supplier
compliance PDFs, EU AI Act, VDA guidelines, and Mercedes owner's manuals.

Two strategies are combined:
  1. KNOWN_PDFS  - direct links already confirmed to point at a PDF.
  2. HUB_PAGES    - listing/index pages that are crawled for any <a href="...pdf">
                    links, so you pick up additional documents (e.g. more
                    annual report years) beyond what's hardcoded.

"""

import os
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 30
OUTPUT_ROOT = Path("./corpus")

# ---------------------------------------------------------------------------
# 1. Direct, already-confirmed PDF links
# ---------------------------------------------------------------------------
KNOWN_PDFS = {
    "bmw_annual_reports": [
        "https://www.bmwgroup.com/content/dam/grpw/websites/bmwgroup_com/ir/downloads/en/2025/investor-presentation/BMW_Investor_Presentation_2025.pdf",
    ],
    "mercedes_annual_reports": [
        "https://group.mercedes-benz.com/documents/investors/reports/annual-report/mercedes-benz/mercedes-benz-group-ag-annual-financial-statements-entity-ag-2025.pdf",
    ],
    "eu_ai_act": [
        # EUR-Lex serves the PDF via a content-negotiated URL; the script
        # will attempt the direct PDF endpoint and fall back to scraping
        # the OJ page for the real PDF link if this changes.
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
    ],
    "vda_guidelines": [
        "https://vda-qmc.de/wp-content/uploads/2023/01/VDA_Volume_Assessment_of_Quality_Management_Methods__Guideline__1st_Edition__November_2017__Online-Document.pdf",
        "https://vda-qmc.de/wp-content/uploads/2023/01/VDA_Band_7_QDX_2021_English.pdf",
    ],
    "supplier_compliance": [
        "https://www.bmwgroup.com/content/dam/grpw/websites/bmwgroup_com/responsibility/downloads/en/2022/BMW-Group-Supplier-Code-of-Conduct-V.3.0_englisch_20221206.pdf",
        "https://gyar.mercedes-benz.hu/application/files/8817/6061/3338/MB_Responsible-Sourcing-Standards-2025_EN_4_1.pdf",
    ],
}

# ---------------------------------------------------------------------------
# 2. Hub / index pages to crawl for additional PDF links
# ---------------------------------------------------------------------------
HUB_PAGES = {
    "bmw_annual_reports": [
        "https://www.bmwgroup.com/en/investor-relations/company-reports.html",
        "https://www.bmwgroup.com/en/download-centre.html",
    ],
    "mercedes_annual_reports": [
        "https://group.mercedes-benz.com/investors/reports-news/annual-reports/download/",
        "https://group.mercedes-benz.com/investors/services/media-center/",
    ],
    "mercedes_manuals": [
        "https://www.mbusa.com/en/owners/manuals",
        "https://www.mbvans.com/en/owner-manuals",
    ],
    "vda_guidelines": [
        "https://vda-qmc.de/en/",
        "https://vda-qmc.de/en/vda-zertifizierungen/vda-6-x/",
    ],
    "supplier_compliance": [
        "https://group.mercedes-benz.com/sustainability/human-rights/supply-chains/management.html",
    ],
}

# Some hub pages render links via JavaScript and won't yield PDFs to a
# plain requests/bs4 crawl (mbusa.com and mbvans.com in particular). These
# are left in HUB_PAGES for completeness, but don't be surprised if the
# crawler finds nothing there — in that case, open the page in a browser
# and copy the PDF links manually into KNOWN_PDFS.


def safe_filename(url: str) -> str:
    """Derive a readable, collision-safe filename from a URL."""
    name = os.path.basename(urlparse(url).path) or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Prefix with a short hash of the full URL to avoid collisions between
    # different documents that happen to share a filename.
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{url_hash}_{name}"


def download_pdf(url: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / safe_filename(url)
    if filepath.exists() and filepath.stat().st_size > 0:
        print(f"  [skip] already downloaded: {filepath.name}")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            print(f"  [warn] not a PDF ({content_type}): {url}")
            return False
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [ok] {filepath.name}  ({filepath.stat().st_size // 1024} KB)")
        return True
    except requests.RequestException as e:
        print(f"  [fail] {url} -> {e}")
        return False


def find_pdf_links(page_url: str) -> list[str]:
    """Scrape a hub page for any links pointing at a PDF."""
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [fail] could not fetch hub page {page_url} -> {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            links.add(urljoin(page_url, href))
    return sorted(links)


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    print("=== Downloading known direct PDF links ===")
    for category, urls in KNOWN_PDFS.items():
        dest = OUTPUT_ROOT / category
        print(f"\n[{category}]")
        for url in urls:
            download_pdf(url, dest)
            time.sleep(0.5)  # be polite

    print("\n=== Crawling hub pages for additional PDFs ===")
    for category, pages in HUB_PAGES.items():
        dest = OUTPUT_ROOT / category
        for page in pages:
            print(f"\n[{category}] scanning {page}")
            pdf_links = find_pdf_links(page)
            if not pdf_links:
                print("  (no PDF links found on this page — may require a browser/JS)")
                continue
            print(f"  found {len(pdf_links)} PDF link(s)")
            for url in pdf_links:
                download_pdf(url, dest)
                time.sleep(0.5)

    print("\nDone. Check ./corpus/<category>/ for downloaded files.")
    print("Categories with few or no files may need manual download —")
    print("see the note in HUB_PAGES about JS-rendered pages.")


if __name__ == "__main__":
    main()