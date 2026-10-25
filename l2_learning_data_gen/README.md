# L2 learning data generation

Install dependencies with `uv sync` (or `pip install .`) and set `OPENAI_API_KEY` in your environment.

Enter your venv and execute:

```bash
bash run_ver2_full_en-ja.sh
python convert_llm_learning_ver2_to_axolotl_format.py
python convert_llm_learning_ver2_eng-jpn_to_jpn-eng_to_axolotl_format.py
```

The launcher generates English examples and Japanese translations for all 9,934 vocabulary prompts in one run. 

Both converters use that same generated file. The first exports English-to-Japanese examples; the second swaps the example and translation to export Japanese-to-English examples while retaining the English grammar explanation. Outputs are grouped by CEFR level under `axolotl_datasets/`. They are used for subsequent L2 acquisition SFT using the axolotl training framework. 
