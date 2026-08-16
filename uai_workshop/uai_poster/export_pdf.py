"""Export poster.html to a print-ready A0 PDF via headless Chromium."""
import os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "poster.html")
OUT_DIR = os.path.join(HERE, "build")
OUT_PATH = os.path.join(OUT_DIR, "poster_A0.pdf")

A0_WIDTH_MM = 841
A0_HEIGHT_MM = 1189


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        page.goto(f"file://{HTML_PATH}")
        page.wait_for_timeout(300)  # let KaTeX auto-render finish
        page.pdf(
            path=OUT_PATH,
            width=f"{A0_WIDTH_MM}mm",
            height=f"{A0_HEIGHT_MM}mm",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
        )
        browser.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
