# Dango🍡

**Dango** is a large language model (LLM) trained on an extensively filtered Japanese corpus.  
It is intended primarily for research on second language acquisition, but it can also be used as a Japanese speaker simulator.

## Model Release

You can access Dango checkpoints at the following links:

- **Pre-trained model:** https://huggingface.co/mattashiho/dango-1.8b-100Btok
- **Model after L2 (English) acquisition:** Coming soon

The Dango architecture is highly comparable to the multilingual llm-jp-3 models.

- **llm-jp-3-1.8b:** https://huggingface.co/llm-jp/llm-jp-3-1.8b

## Code and Resources for Reproducing Dango

1. We used the Japanese portion of the llm-jp-corpus-v3 dataset.  
   - **Dataset:** https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/tree/main

2. We applied strict filtering to reduce L2 contamination.  
   - Filtering scripts: Coming soon

3. We trained Dango with an architecture comparable to llm-jp-3 models.  
   We adopted the training scripts/environment from:  
   - https://github.com/llm-jp/scripts/tree/main/pretrain/scripts/v4-high-quality-abci/pretrain  
   - Scripts used in our experiments: Coming soon

## License

- **Codes and Scripts:** CC BY 4.0
- **Pre-trained models:** CC BY 4.0
- **Instruction-tuned models (after L2 acquisition training):** Trained with Ichikara instruction data, which prohibits commercial use of derivative models  
  - **Ichikara data source:** https://liat-aip.sakura.ne.jp/wp/llm%e3%81%ae%e3%81%9f%e3%82%81%e3%81%ae%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%a9%e3%82%af%e3%82%b7%e3%83%a7%e3%83%b3%e3%83%87%e3%83%bc%e3%82%bf%e4%bd%9c%e6%88%90/llm%e3%81%ae%e3%81%9f%e3%82%81%e3%81%ae%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%a9%e3%82%af%e3%82%b7%e3%83%a7%e3%83%b3%e3%83%87%e3%83%bc%e3%82%bf-%e5%85%ac%e9%96%8b/

## Citation

If you use Dango in your research, please cite:

```bibtex
@inproceedings{matta2026anlp,
  author    = {Shiho Matta and Yin Jou Huang and Fei Cheng and Takashi Kodama and Hirokazu Kiyomaru and Yugo Murawaki},
  title     = {Pretraining a Japanese-Only Large Language Model for Studying Second Language Acquisition},
  booktitle = {Proceedings of the Thirty-second Annual Meeting of the Association for Natural Language Processing},
  year      = {2026},
  pages     = {3225--3230},
  publisher = {Association for Natural Language Processing},
  address   = {Utsunomiya, Japan}
}
```