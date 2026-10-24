"""Extract U+4E00–U+9FFF characters without Unihan kJapanese* readings.

This selects characters without recorded Japanese readings, not specifically
simplified Chinese characters. Input order is preserved in the output.
"""

import argparse
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LINE_RE = re.compile(r"^(U\+[0-9A-F]{4,6})\t(\w+)\t(.+)$")


def extract_characters(input_path: Path, output_path: Path) -> int:
    """Write character/codepoint pairs and return the number written."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    # Track all characters to retain the original script's counts and order.
    has_japanese: dict[int, bool] = {}
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            match = LINE_RE.match(line.strip())
            if not match:
                continue
            codepoint, field, _ = match.groups()
            cp = int(codepoint[2:], 16)
            has_japanese.setdefault(cp, False)
            if field.startswith("kJapanese"):
                has_japanese[cp] = True

    bmp_chars = [cp for cp in has_japanese if 0x4E00 <= cp <= 0x9FFF]
    selected = [cp for cp in bmp_chars if not has_japanese[cp]]
    with output_path.open("w", encoding="utf-8") as output:
        for cp in selected:
            output.write(f"{chr(cp)}\t{cp:04X}\n")

    print(f"Total chars: {len(has_japanese)}")
    print(f"Chars in BMP CJK Unified Ideographs: {len(bmp_chars)}")
    print(f"Chars in BMP CJK Unified Ideographs and without Japanese readings: {len(selected)}")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=BASE_DIR / "Unihan" / "Unihan_Readings.txt",
        help="Unihan readings file (default: bundled Unicode 17.0.0 data)",
    )
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "unihan_bmp_no_japanese.txt",
        help="Output TSV file (default: beside this script; overwritten)",
    )
    args = parser.parse_args()
    extract_characters(args.input, args.output)


if __name__ == "__main__":
    main()
