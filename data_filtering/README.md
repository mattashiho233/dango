# Japanese corpus filtering

`filtering.py` filters the corpus using character and language contamination rules.
`run_filtering_ja_cc_level0_demo.sh` is an example launcher; update its input and
output paths before use.

## Reproduce the Chinese-only character list

We built a Chinese-only character list for the Chinese filter, derived from the Unihan data. 
From the repository root, run:

```bash
python data_filtering/unihan_without_japanese_readings.py
```

The script uses the bundled `Unihan/Unihan_Readings.txt` and generates
`data_filtering/unihan_bmp_no_japanese.txt`. Its default paths are relative to the script. To use different paths:

```bash
python data_filtering/unihan_without_japanese_readings.py \
  --input /path/to/Unihan_Readings.txt \
  --output /path/to/characters.txt
```

## Unicode data

- Source: Unicode Character Database, **Unicode 17.0.0**,
  [Unihan.zip](https://www.unicode.org/Public/17.0.0/ucd/Unihan.zip).
- Bundled source file: `Unihan/Unihan_Readings.txt`, dated 2025-07-24 in its header,
  moved here without modification.
- Derived file: `unihan_bmp_no_japanese.txt`.
- License: [Unicode License v3](Unihan/LICENSE-UNICODE.txt). This applies to the
  bundled Unicode data and derived character list, separately from the project's
  code license.
