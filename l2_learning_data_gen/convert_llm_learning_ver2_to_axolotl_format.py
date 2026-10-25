import os
import json
from pathlib import Path


base_dir = Path(__file__).resolve().parent
source_path = base_dir / "runs/ver2_en-ja/output_system_ver2_user_full_vocab.jsonl"

source_lang = "eng"
target_lang = "jpn"

lang_name_map = {
    "jpn": "日本語",
    "eng": "英語",
}

output_path_prefix = str(base_dir / "axolotl_datasets" / f"llm_gen_learning_ver2_{source_lang}-{target_lang}")
# files will be created for each cefr level
output_file_name_prefix = f"/llm_gen_learning_ver2_{source_lang}-{target_lang}"

# mkdir of output directory if not exists
os.makedirs(output_path_prefix, exist_ok=True)

with open(source_path, "r", encoding="utf-8", newline="") as f:
    for line in f:
        row = json.loads(line)
        # ipdb.set_trace()
        word_info = row["request"]["input"]
        assert len(word_info.split(",")) == 3, f"word_info: {word_info}"
        word_lemma = word_info.split(",")[0]
        word_pos = word_info.split(",")[1].strip()
        _word_level = word_info.split(",")[2].split(" ")[-1].strip()
        assistant_message = row["response"]["output"][0]["content"][0]["text"]

        # user_message = f"以下の{lang_name_map[source_lang]}の文を{lang_name_map[target_lang]}に訳しなさい。" + "\n" + row[1].strip()
        if source_lang == "eng" and target_lang == "jpn":
            user_message = f"以下の英単語の意味を日本語で解釈し、単語の翻訳例を挙げなさい。また、その単語を使った例文と日本語の翻訳を作成の上、例文中に現れる文法の要点を説明しなさい。" + "\n" + word_lemma + " (" + word_pos + ")"
        elif source_lang == "jpn" and target_lang == "eng":
            user_message = f"以下の英単語の意味を日本語で解釈し、単語の翻訳例を挙げなさい。また、その単語の意味の一つを使った日本語の例文と英語の翻訳を作成の上、英文中に現れる文法の要点を説明しなさい。" + "\n" + word_lemma + " (" + word_pos + ")"
        else:
            raise ValueError(f"Unsupported language pair: {source_lang}-{target_lang}")
        assistant_message = assistant_message.strip()

        if not user_message or not assistant_message:
            continue


        axolotl_data = {
            "conversations": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            "word_info": word_info
        }
        
        output_file_path = output_path_prefix + output_file_name_prefix + f"_{_word_level}.jsonl"
        with open(output_file_path, "a", encoding="utf-8", newline="") as out_f:
            out_f.write(json.dumps(axolotl_data, ensure_ascii=False) + "\n")
