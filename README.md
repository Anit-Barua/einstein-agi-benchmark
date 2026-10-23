# Einstein AGI Benchmark (EAB)

Open-ended benchmark and rule-based scorer for **scientific-discovery reasoning** in language models: given the anomalous observation behind a landmark discovery, derive the insight, with no answer options and no named discovery.


## Read this first: the released results are a simulation

The run in `results/` (benchmark v9.0.0, seed 20250106, Kaggle 2×T4) **did not use live model inference**.

* The reference oracle "EBR" answers come from a bank of hand-written derivations, not from GPT-4o.
* The eight open-weight models (SmolLM2, TinyLlama, Qwen2.5) were represented by four capability-tier templates scaled to parameter count.
* Scorer parameters were tuned over several development sessions and are not frozen.
* Calibration, consistency, adversarial and Krippendorff results use scripted or synthetic inputs.
* For the eight simulated models, three of the five CAGI terms are heuristics, so **CAGI cannot be compared across models**.

So the numbers show how the scoring pipeline behaves. They are **not** measurements of model capability. The paper explains each caveat.

## Layout

| Path | Contents |
|---|---|
| `data/questions_core41.{csv,json}` | The 41 core questions with canonical key insights (LaTeX source text) |
| `results/` | All result CSV/JSON files from the v9.0 run, `results_table_I.tex`, run metadata |
| `figures/` | The ten original TIFF figures from the run |
| `code/einstein_benchmark_v9_demo.ipynb` | Sanitised notebook (scorer, simulator, statistics). Stored outputs removed; API key removed |
| `analysis/reproduce_paper_numbers.py` | Recomputes every number quoted in the paper from `results/` (output in `reproduced_numbers.txt`) |
| `analysis/make_paper_figure.py` | Regenerates `paper/fig_results.pdf` |
| `paper/` | `eab_ieee.tex`, `references.bib`, figure, compiled PDF |

The fictional probes (X01-X05) and adversarial items (ADV01-ADV04) are defined in the notebook; the adversarial premises are also listed in `results/results_adversarial.csv`.

## Reproduce

```bash
pip install -r requirements.txt
python analysis/reproduce_paper_numbers.py     # prints every number used in the paper
python analysis/make_paper_figure.py           # rebuilds paper/fig_results.pdf
```

Compile the paper on Overleaf (upload the contents of `paper/`; document class IEEEtran, BibTeX) or locally: `pdflatex eab_ieee && bibtex eab_ieee && pdflatex eab_ieee && pdflatex eab_ieee`.

## API keys

No credentials are stored here. The notebook reads `OPENAI_API_KEY` from the environment (use Kaggle Secrets or a local `.env`). A key that was hard-coded in an earlier development version must be treated as compromised and revoked.

## Next steps for a valid evaluation

Live inference with fixed decoding settings; freeze scorer parameters before scoring held-out items; human-validated alignment measure; human expert baseline; independent raters; sensitivity analysis of CAGI weights.

