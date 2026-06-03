from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "templates" / "dashboard_template.html"
OUTPUT_DIR = ROOT / "output"
FRAGMENT_DIR = OUTPUT_DIR / "fragments"


PLACEHOLDER_MAP: Dict[str, str] = {
    "<!-- INJECT:HEADLINE_PILLS -->": "headline_pills.html",
    "<!-- INJECT:US_MACRO -->": "us_macro.html",
    "<!-- INJECT:FLOW -->": "flow.html",
    "<!-- INJECT:GLOBAL_REGIME -->": "global_regime.html",
    "<!-- INJECT:SYNTHESIS -->": "synthesis.html",
    "<!-- INJECT:COUNTRIES -->": "countries.html",
    "<!-- INJECT:PAGE_SCRIPTS -->": "page_scripts.html",
}


def resolve_report_date(cli_date: str | None) -> str:
    if cli_date:
        # Validate explicit date for filename consistency.
        datetime.strptime(cli_date, "%Y-%m-%d")
        return cli_date
    return date.today().isoformat()


def read_template(template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Dashboard template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def read_fragment(fragment_name: str, fragment_dir: Path) -> str:
    path = fragment_dir / fragment_name
    if not path.exists():
        return f"<!-- Missing fragment: {fragment_name} -->"
    return path.read_text(encoding="utf-8")


def assemble_dashboard(
    report_date: str,
    template_path: Path = TEMPLATE_PATH,
    fragment_dir: Path = FRAGMENT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    html = read_template(template_path).replace("{{ report_date }}", report_date)
    for placeholder, fragment_name in PLACEHOLDER_MAP.items():
        html = html.replace(placeholder, read_fragment(fragment_name, fragment_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dashboard_{report_date}.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble tabbed macro dashboard from section fragments."
    )
    parser.add_argument(
        "--date",
        dest="report_date",
        default=None,
        help="Report date in YYYY-MM-DD format. Defaults to today's date.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE_PATH,
        help="Path to dashboard template HTML.",
    )
    parser.add_argument(
        "--fragments",
        type=Path,
        default=FRAGMENT_DIR,
        help="Directory containing fragment HTML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for final dashboard output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_date = resolve_report_date(args.report_date)
    output_path = assemble_dashboard(
        report_date=report_date,
        template_path=args.template,
        fragment_dir=args.fragments,
        output_dir=args.output_dir,
    )
    print(output_path)


if __name__ == "__main__":
    main()
