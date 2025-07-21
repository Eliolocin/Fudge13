# LLM-as-Judge Translation Interface

A simple web interface for judging English-to-Filipino translations using LLM models (Gemini & GPT).

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Environment Setup

### 1. Create a Python Virtual Environment

Create and activate a virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create `.env` file with your API keys:
```bash
cp .env.example .env
```

Add your API keys to `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** API keys are optional - the application will run in demo mode without them.

### 4. Create Required Directories

```bash
# Create prompts directory for system prompts
mkdir prompts

# Create results directory for saving judgments (optional - auto-created)
mkdir results
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at:
- http://127.0.0.1:5000 (localhost)
- http://192.168.1.7:5000 (network interface)

## Usage

1. Select or create a system prompt with tokens
2. Input source text and Filipino translation
3. Choose your LLM judge
4. Click "Judge" to get structured evaluation results

## Tokens Available

- `{{source_text}}` - The English source text
- `{{fil_translation}}` - The Filipino translation to be judged
- `{{ref_translation}}` - Optional reference translation

## Automated Evaluation System

The project includes an automated evaluation script (`auto_evaluate.py`) for systematically testing LLM judge performance against human-labeled translation datasets. This is particularly useful for research and benchmarking.

### Prerequisites for Evaluation

1. **CSV Dataset**: Prepare a CSV file with human-labeled translation pairs
2. **Required Columns**:
   - `English`: Source text in English
   - `Filipino-Correct`: Human-verified good translation
   - `Filipino-Flawed`: Human-verified bad translation
3. **Optional Columns** (for Spearman correlation analysis):
   - `Human-Score-Correct`: Human score 1-5 for correct translations
   - `Human-Score-Flawed`: Human score 1-5 for flawed translations
4. **API Keys**: Valid Google API key in `.env` file (required for actual LLM evaluation)

### CSV Format Examples

#### Basic Format (Binary Analysis Only)
```csv
English,Filipino-Correct,Filipino-Flawed
"Hello, how are you?","Kumusta ka?","Kamusta kayo sa lahat?"
"I love reading books.","Mahilig akong magbasa ng mga libro.","Gusto ko ang pagbabasa mga libro mo."
"The weather is beautiful today.","Ang panahon ay maganda ngayon.","Ang temperatura ay beautiful ngayong araw."
```

#### Extended Format (With Human Scores for Correlation Analysis)
```csv
English,Filipino-Correct,Filipino-Flawed,Human-Score-Correct,Human-Score-Flawed
"Hello, how are you?","Kumusta ka?","Kamusta kayo sa lahat?",5,2
"I love reading books.","Mahilig akong magbasa ng mga libro.","Gusto ko ang pagbabasa mga libro mo.",5,1
"The weather is beautiful today.","Ang panahon ay maganda ngayon.","Ang temperatura ay beautiful ngayong araw.",4,1
```

**Human Score Scale**: 1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent

See `example_evaluation_data.csv` for a complete example.

### Running Evaluations

#### Basic Usage

```bash
# Evaluate all translations in the dataset (both correct and flawed)
python auto_evaluate.py --csv_file your_dataset.csv

# Use the provided example dataset
python auto_evaluate.py --csv_file example_evaluation_data.csv
```

#### Advanced Options

```bash
# Evaluate specific row range (useful for large datasets)
python auto_evaluate.py --csv_file data.csv --row_start 1 --row_end 100

# Evaluate only correct translations
python auto_evaluate.py --csv_file data.csv --translation_type correct

# Evaluate only flawed translations  
python auto_evaluate.py --csv_file data.csv --translation_type flawed

# Use agentic judge mode (with Google Search grounding)
python auto_evaluate.py --csv_file data.csv --use_agentic

# Specify different LLM model
python auto_evaluate.py --csv_file data.csv --llm_model gemini-2.5-pro

# Custom output directory
python auto_evaluate.py --csv_file data.csv --output_dir my_results/

# Evaluate with reruns for consistency analysis (each condition run 3 times total)
python auto_evaluate.py --csv_file data.csv --reruns 2
```

#### Combined Options

```bash
# Comprehensive evaluation with agentic judge on specific range
python auto_evaluate.py --csv_file dataset.csv \
                       --row_start 1 --row_end 50 \
                       --translation_type default \
                       --use_agentic \
                       --llm_model gemini-2.5-flash \
                       --output_dir evaluation_results/

# Consistency analysis with reruns and agentic mode
python auto_evaluate.py --csv_file dataset.csv \
                       --reruns 2 --use_agentic \
                       --row_end 10
```

### Evaluation Logic

The script automatically determines pass/fail for each evaluation:

- **Correct Translations**: 
  - ✅ **Pass** if LLM score ≥ 3 (Good/Excellent)
  - ❌ **Fail** if LLM score < 3 (Poor)

- **Flawed Translations**:
  - ✅ **Pass** if LLM score < 3 (Correctly identified as poor)
  - ❌ **Fail** if LLM score ≥ 3 (Incorrectly rated as good)

### Output Files

Results are organized in session-specific directories to prevent conflicts between different evaluation runs:

```
evaluation_results/
├── session_data_gemini-2-5-flash_prompt_20250721_143022/
│   ├── eval_20250721_143022_params.json    # Complete evaluation results
│   ├── eval_config_20250721_143022.json    # Parameters used for evaluation
│   ├── eval_summary_20250721_143022.json   # Summary statistics and accuracy
│   └── eval_progress_20250721_143022.log   # Detailed execution logs
├── session_data_gemini-2-5-pro_agentic_20250721_144000/
│   └── ...
└── session_data_gemini-2-5-flash_prompt_reruns2_20250721_145000/
    └── ...
```

**Session Directory Naming**: `session_{csv_name}_{model}_{mode}_{additional_params}_{timestamp}`

### Understanding Results

#### Individual Result Structure
```json
{
  "timestamp": "2025-07-21T14:30:22.123456",
  "request_data": { ... },
  "llm_response": { ... },
  "final_score": {
    "score": 4,
    "label": "Good", 
    "true_count": 4
  },
  "evaluation_metadata": {
    "csv_row": 15,
    "translation_type": "correct",
    "expected_good_translation": true,
    "llm_pass": true,
    "run_id": 0,
    "run_total": 3,
    "base_eval_id": "data_row15_correct_gemini-2-5-flash_prompt",
    "unique_eval_id": "data_row15_correct_gemini-2-5-flash_prompt_run0",
    "human_score": 5,
    "has_human_scores": true,
    "evaluation_parameters": { ... }
  }
}
```

#### Summary Statistics
```json
{
  "evaluation_summary": {
    "total_evaluations": 100,
    "unique_conditions": 50,
    "runs_per_condition": 2,
    "correct_translations": {
      "count": 50,
      "passes": 45,
      "pass_rate": 0.90
    },
    "flawed_translations": {
      "count": 50, 
      "passes": 42,
      "pass_rate": 0.84
    },
    "overall_accuracy": 0.87
  },
  "rerun_statistics": {
    "unique_conditions_evaluated": 25,
    "runs_per_condition": 2,
    "consistency_analysis": {
      "data_row1_correct_gemini-2-5-flash_prompt": {
        "runs_completed": 2,
        "scores": [5, 4],
        "score_mean": 4.5,
        "score_std": 0.5,
        "pass_consistency": 1.0
      }
    }
  }
}
```

### Error Handling & Rate Limiting

The script includes robust error handling:

- **Exponential Backoff**: Automatically retries failed API calls with 15s → 30s → 60s delays
- **Rate Limit Detection**: Gracefully handles API quota limits
- **Incremental Saving**: Results saved every 10 evaluations to prevent data loss
- **Resumable Execution**: Use `--row_start` to continue from specific points
- **Comprehensive Logging**: Detailed logs for debugging and progress tracking

### Tips for Large Datasets

1. **Test with small ranges first**: Use `--row_start 1 --row_end 10` to verify setup
2. **Monitor API usage**: Check your Google API quota in the console
3. **Use incremental processing**: Process datasets in chunks (e.g., 50-100 rows at a time)
4. **Save intermediate results**: The script automatically saves progress every 10 evaluations

### Advanced Features

#### Consistency Analysis with Reruns
- Use `--reruns N` to run each evaluation multiple times  
- Analyze LLM consistency across identical inputs
- Statistical measures: mean score, standard deviation, pass consistency
- Useful for reliability studies and variance analysis

#### Human Score Integration
- Add `Human-Score-Correct` and `Human-Score-Flawed` columns (1-5 scale)
- Enables Spearman rank correlation analysis between human and LLM scores
- Automatic detection and validation of human scores
- Results include both binary pass/fail and continuous score data

#### Session Management
- Parameter-aware result organization prevents conflicts
- Each unique parameter combination gets its own session directory
- Enables safe parallel evaluation with different settings
- Facilitates systematic comparison across models/modes

### Troubleshooting

- **"No valid Google API key"**: Ensure `GOOGLE_API_KEY` is set in your `.env` file
- **"CSV missing required columns"**: Verify your CSV has `English`, `Filipino-Correct`, `Filipino-Flawed` columns
- **"Invalid human scores"**: Human scores must be integers 1-5 if present
- **Rate limiting errors**: Wait for quota reset or reduce evaluation frequency
- **Memory issues with large CSVs**: Process in smaller chunks using `--row_start` and `--row_end`
