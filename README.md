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

## LLM Judge Analysis System

The project includes a comprehensive analysis script (`analysis.py`) for evaluating the performance and reliability of the LLM judge against human evaluations. This tool processes all evaluation results from the `evaluation_results/` directory and computes three critical metrics for LLM judge validation.

### Prerequisites for Analysis

1. **Evaluation Data**: Complete evaluation results generated by `auto_evaluate.py`
2. **Human Scores** (Optional): For correlation analysis, include `Human-Score-Correct` and `Human-Score-Flawed` columns in your CSV
3. **Rerun Data** (Optional): For consistency analysis, generate reruns using `--reruns` parameter in evaluation
4. **Python Dependencies**: `scipy` (optional but recommended for more accurate statistical calculations)

### Running Analysis

#### Basic Usage

```bash
# Analyze all evaluation results
python analysis.py

# Analyze with summary output to console
python analysis.py --format summary

# Save to custom directory
python analysis.py --output-dir my_analysis_results
```

#### Filtered Analysis

```bash
# Analyze specific LLM model only
python analysis.py --filter-model gemini-2.5-flash

# Analyze specific evaluation session
python analysis.py --filter-session session_example_evaluation_data_gemini-2-5-flash_prompt_rows1-5_20250721_143022

# Combine filters and custom output
python analysis.py --filter-model gemini-2.5-pro --output-dir flash_vs_pro_analysis
```

### Analysis Metrics

#### 1. Basic Classification Metrics

Evaluates how well the LLM judge distinguishes between good and bad translations:

- **Precision**: Of translations marked as good, what percentage were actually good?
- **Recall**: Of actually good translations, what percentage did the LLM identify correctly?
- **Accuracy**: Overall percentage of correct classifications (both good and bad)
- **F1-Score**: Harmonic mean of precision and recall (balanced performance measure)

**Interpretation Guide:**
- `Accuracy > 90%`: Excellent classification performance
- `Accuracy 70-90%`: Good performance, some refinement possible
- `Accuracy < 70%`: Significant improvement needed

#### 2. Spearman's Rank Correlation Coefficient

Measures how well LLM scores (1-5) correlate with human scores (1-5):

- **Correlation Strength**: -1.0 to 1.0 scale
  - `|r| > 0.9`: Very Strong correlation
  - `|r| > 0.7`: Strong correlation  
  - `|r| > 0.5`: Moderate correlation
  - `|r| > 0.3`: Weak correlation
  - `|r| ≤ 0.3`: Very Weak correlation
- **Statistical Significance**: p-value < 0.05 indicates significant correlation
- **Sample Size**: Minimum 3 paired scores required for analysis

**Requirements**: CSV must include `Human-Score-Correct` and `Human-Score-Flawed` columns with values 1-5.

#### 3. Score Variation Analysis

Analyzes consistency of LLM judge across multiple runs of identical conditions:

- **Pass Consistency**: Percentage of reruns with same pass/fail decision
- **Score Standard Deviation**: Numerical variation in scores (lower = more consistent)
- **Coefficient of Variation**: Relative variability measure
- **Most/Least Consistent Conditions**: Identifies which translations show highest/lowest variability

**Requirements**: Generate rerun data using `python auto_evaluate.py --reruns N` where N > 0.

### Output Files

Analysis results are saved in timestamped JSON files with comprehensive data:

```
analysis_results/
└── llm_judge_analysis_20250721_143022.json
```

#### JSON Structure

```json
{
  "analysis_metadata": {
    "generated_at": "2025-07-21T14:30:22.123456",
    "total_evaluations_analyzed": 150,
    "unique_sessions": 5,
    "evaluation_parameters_summary": {
      "models_used": ["gemini-2.5-flash", "gemini-2.5-pro"],
      "modes_used": ["agentic", "prompt"],
      "csv_files_used": ["dataset1", "dataset2"]
    }
  },
  "classification_metrics": {
    "confusion_matrix": { "true_positive": 45, "false_positive": 5, ... },
    "metrics": { "precision": 0.90, "recall": 0.95, "accuracy": 0.92, "f1_score": 0.925 }
  },
  "correlation_analysis": {
    "spearman_correlation": 0.85,
    "p_value": 0.001,
    "sample_size": 100,
    "is_significant": true
  },
  "variation_analysis": {
    "overall_pass_consistency": 0.95,
    "overall_score_std": 0.25,
    "total_conditions_analyzed": 50
  },
  "summary_insights": {
    "key_findings": ["Excellent classification accuracy: 92.0%", ...],
    "recommendations": ["Consider prompt engineering...", ...]
  }
}
```

### Understanding Results

#### Performance Benchmarks

**Classification Performance:**
- `Accuracy ≥ 90%`: Production-ready judge
- `Accuracy 80-89%`: Good performance, monitor edge cases  
- `Accuracy 70-79%`: Acceptable for research, needs improvement for production
- `Accuracy < 70%`: Requires significant refinement

**Human Correlation:**
- `r ≥ 0.8`: Excellent alignment with human judgments
- `r 0.6-0.79`: Good correlation, minor disagreements expected
- `r 0.4-0.59`: Moderate correlation, investigate systematic differences
- `r < 0.4`: Poor correlation, major evaluation criteria misalignment

**Consistency Standards:**
- `Pass Consistency ≥ 95%`: Highly reliable judge
- `Pass Consistency 85-94%`: Generally reliable, some variability
- `Pass Consistency < 85%`: Inconsistent, consider temperature/prompt tuning

### Advanced Usage Examples

#### Research Quality Assessment

```bash
# Generate comprehensive research report
python analysis.py --output-dir research_analysis_2025

# Compare models side-by-side
python analysis.py --filter-model gemini-2.5-flash --output-dir flash_analysis
python analysis.py --filter-model gemini-2.5-pro --output-dir pro_analysis
```

#### Production Validation

```bash
# Validate production readiness with full dataset
python auto_evaluate.py --csv_file production_validation_dataset.csv --reruns 3
python analysis.py --format summary

# Monitor consistency across deployment
python analysis.py --filter-model production-model --output-dir consistency_monitoring
```

### Analysis Troubleshooting

**Common Issues:**

- **"Insufficient data for correlation analysis"**: Add `Human-Score-Correct` and `Human-Score-Flawed` columns to CSV data
- **"No rerun data found for variation analysis"**: Generate reruns using `--reruns` parameter in evaluation
- **"No evaluation data found"**: Run `auto_evaluate.py` first to generate evaluation results
- **Low accuracy unexpected**: Check if evaluation logic matches human labeling criteria
- **High variation in scores**: Consider lowering model temperature or refining prompts

**Optimization Recommendations:**

1. **Low Classification Accuracy**: 
   - Review prompt engineering and evaluation criteria
   - Increase training examples or adjust scoring thresholds
   - Consider different models or model parameters

2. **Poor Human Correlation**:
   - Investigate systematic differences in evaluation approach
   - Ensure human scores follow same 1-5 scale as LLM
   - Validate human annotation quality and consistency

3. **High Score Variation**:
   - Lower model temperature for more deterministic responses  
   - Refine prompts to reduce ambiguity
   - Increase sample size to identify true performance ranges
