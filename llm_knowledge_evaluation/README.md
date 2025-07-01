# llm_variable_estimation

This module evaluates whether Large Language Models (LLMs) can estimate demographic variable distributions — such as age distributions or household sizes — using prompting alone. It supports both local models (via [Ollama](https://ollama.com)) and OpenAI batch submission workflows.

---

## 📦 Folder Structure

```
llm_variable_estimation/
├── core/                   # Estimation logic and metrics
├── data/                   # Ground truth data for Newcastle upon Tyne, UK
├── openai_batch/           # Submit and retrieve OpenAI batch jobs
├── ollama_batch/           # Run batch jobs locally using Ollama
├── preprocessing/          # Scripts to convert raw data to LLM-ready format
├── prompts/                # Prompt templates
├── schemas/                # Response schemas
├── dashboard.py            # Dash dashboard to explore results
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You’ll also need:

- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Ollama](https://ollama.com) installed and models pulled locally, if using local models

---

### 2. Set Python Path
```bash
source .env
```

### 3. Prepare Input Data

Raw demographic data must be preprocessed into LLM-friendly format.

For UK locations, raw data can be downloaded from the [Office for National Statistics (ONS) datasets portal](https://www.ons.gov.uk/datasets/create). After selecting the relevant dataset (e.g. age or household size distribution by upper tier local authority), save it as:

```
data/{variable_name}/raw_data.csv
```

Then run the appropriate preprocessing script(s):

```bash
python preprocessing/resample_age_bins.py
python preprocessing/add_percentage_col.py
```

This will produce processed input files at:

```
data/evaluation/{variable}/sampled_data.csv
```

---

### 4. Run Estimations

#### ✅ A) Using Local Models via Ollama

```bash
python ollama_batch/run_batch_estimations.py
```

Edit `VARIABLES` and `MODELS` in that file to change the run configuration.

#### ☁️ B) Using OpenAI Batch Submission

1. Prepare the batch input file:
   ```bash
   python openai_batch/submit_batch.py
   ```

2. After the batch completes (check the OpenAI dashboard), download and insert results:
   ```bash
   python batch/download_batch_results.py --metadata outputs/batch_jobs/<...>_metadata.json
   ```

---

## 📊 Explore Results

Launch the interactive dashboard to explore predictions vs. ground truth:

```bash
streamlit run dashboard.py
```

Features:
- Compare performance across variables, models, locations, and groups
- Visualise predicted distributions vs census data
- View error metrics by category

---

## 📁 Outputs

OpenAI predictions and run metadata are saved under `outputs/batch_jobs/`.  Both OpenAI and Ollama data are inserted into the local database under `data/outputs.sqlite`.

---

## 🧪 Supported Variables

The following variables are currently supported:

- `age_distribution`
- `household_size`

To add support for new variables, update `VARIABLE_CONFIG` in `core/estimator.py`.

---