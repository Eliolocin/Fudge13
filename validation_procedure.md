# Validation Test Procedure for Academic Paper

This document outlines the standardized testing procedure for evaluating LLM-as-Judge performance using the validation dataset for academic research.

## Dataset Overview

- **File**: `validation_set.csv`
- **Format**: 3-column CSV (Source Text (English), Target Text (Filipino), Final Score)
- **Total Rows**: 40 translation pairs for Western Animated Series
- **Score Range**: 1-5 (human evaluation scores)
- **Content**: Mix of conversational text, technical terms, cultural references, literary quotes, and idiomatic expressions

## Test Configuration

- **Model**: Gemini 2.5 Flash only
- **Reruns**: 2 additional runs (3 total runs per condition)
- **Modes**: Both Agentic and Non-Agentic
- **Output**: Separate session directories for each mode

## Step-by-Step Procedure

### Prerequisites

1. **Create and set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment (Windows)
   venv\Scripts\activate
   
   # On Linux/macOS use:
   # source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file and add your Google API key:
   # GOOGLE_API_KEY=your_actual_api_key_here
   # OPENAI_API_KEY=your_openai_key_here (optional for this test)
   ```

3. **Verify setup**:
   ```bash
   # Check that validation_set.csv exists
   ls validation_set.csv
   
   # Verify environment variables are loaded
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GOOGLE_API_KEY loaded:', bool(os.getenv('GOOGLE_API_KEY')))"
   ```

4. **Clean previous results** (optional):
   ```bash
   # Remove old evaluation results if starting fresh
   # rm -rf evaluation_results/
   # rm -rf analysis_results/
   ```

## Section A: Non-Agentic Evaluation

This section covers the standard prompt-engineered judge evaluation without agentic features.

### Step A1: Run Non-Agentic Evaluation

Execute the standard prompt-engineered judge evaluation:

```bash
python auto_evaluate.py --csv_file validation_set.csv --llm_model gemini-2.5-flash --reruns 2 --prompt_file prompts/alternative_judge.txt
```

**Expected output location**: `evaluation_results/session_validation_set_gemini-2-5-flash_prompt_reruns2_[TIMESTAMP]/`
(Note: If using custom prompt, session name will include prompt name)

**Expected runtime**: ~15-20 minutes (58 rows × 3 runs = 174 API calls)

### Step A2: Generate Non-Agentic Analysis

Generate comprehensive analysis for the non-agentic results:

```bash
python analysis.py --filter-model gemini-2.5-flash --filter-session session_validation_set_gemini-2-5-flash_prompt --output-dir analysis_results/non_agentic_analysis
```

**Output location**: `analysis_results/non_agentic_analysis/llm_judge_analysis_[TIMESTAMP].json`

## Section B: Agentic Evaluation

This section covers the agentic judge evaluation with Google Search grounding and enhanced reasoning capabilities.

### Step B1: Run Agentic Evaluation

Execute the agentic judge evaluation with Google Search grounding:

```bash
python auto_evaluate.py --csv_file validation_set.csv --llm_model gemini-2.5-flash --use_agentic --reruns 2 --prompt_file prompts/alternative_judge.txt
```

**Expected output location**: `evaluation_results/session_validation_set_gemini-2-5-flash_agentic_reruns2_[TIMESTAMP]/`
(Note: If using custom prompt, session name will include prompt name)

**Expected runtime**: ~25-30 minutes (additional processing time for search grounding)

### Step B2: Generate Agentic Analysis

Generate comprehensive analysis for the agentic results:

```bash
python analysis.py --filter-model gemini-2.5-flash --filter-session session_validation_set_gemini-2-5-flash_agentic --output-dir analysis_results/agentic_analysis
```

**Output location**: `analysis_results/agentic_analysis/llm_judge_analysis_[TIMESTAMP].json`

## Section C: Combined Analysis

### Step C1: Generate Combined Analysis

For comprehensive comparison across both modes:

```bash
python analysis.py --filter-model gemini-2.5-flash --output-dir analysis_results/combined_analysis
```

**Output location**: `analysis_results/combined_analysis/llm_judge_analysis_[TIMESTAMP].json`


## File Organization for Paper

After completion, your results will be organized as:

```
evaluation_results/
├── session_validation_set_gemini-2-5-flash_prompt_reruns2_[TIMESTAMP]/
│   ├── eval_[TIMESTAMP]_params.json        # Raw evaluation data
│   ├── eval_config_[TIMESTAMP].json        # Configuration parameters
│   ├── eval_summary_[TIMESTAMP].json       # Statistical summary
│   └── eval_progress_[TIMESTAMP].log       # Execution logs
└── session_validation_set_gemini-2-5-flash_agentic_reruns2_[TIMESTAMP]/
    └── [same structure as above]

analysis_results/
├── non_agentic_analysis/
│   └── llm_judge_analysis_[TIMESTAMP].json
├── agentic_analysis/
│   └── llm_judge_analysis_[TIMESTAMP].json
└── combined_analysis/
    └── llm_judge_analysis_[TIMESTAMP].json
```

## Troubleshooting

- **Rate limiting**: If you encounter API rate limits, the script will automatically retry with exponential backoff
- **Incomplete runs**: Use `--row_start` and `--row_end` parameters to resume from specific points
- **Missing analysis data**: Ensure evaluation completed successfully before running analysis

## Prompt Selection Options

### Default Prompt
By default, the system uses `prompts/basic_translation_judge.txt` which evaluates translations on 6 boolean criteria:
- Accuracy, Fluency, Coherence, Cultural Appropriateness, Guideline Adherence, Completeness

### Custom Prompts
To use different evaluation criteria or approaches:
1. Create new prompt files in the `prompts/` directory
2. Use the `--prompt_file` parameter to specify the custom prompt
3. Ensure custom prompts include the required tokens: `{{source_text}}` and `{{fil_translation}}`

### Session Organization
When using custom prompts, the system automatically:
- Includes the prompt name in session directory naming
- Prevents conflicts between different prompt evaluations
- Maintains separate analysis results for each prompt variant

## Key Metrics for Paper

The analysis will generate the following metrics suitable for academic reporting:

### 1. Classification Performance
- **Accuracy**: Overall percentage of correct pass/fail classifications
- **Precision**: Of translations marked as good, what percentage were actually good
- **Recall**: Of actually good translations, what percentage were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### 2. Human-LLM Correlation
- **Spearman's Rank Correlation**: Correlation between human scores (1-5) and LLM scores (1-5)
- **Statistical Significance**: p-value for correlation significance testing

### 3. Consistency Analysis
- **Pass Consistency**: Percentage of identical pass/fail decisions across reruns
- **Score Standard Deviation**: Numerical variation in scores across reruns
- **Inter-run Reliability**: Measure of judge consistency

## Total Expected Runtime

- **Section A (Non-Agentic)**: ~15-20 minutes for evaluation + <1 minute for analysis
- **Section B (Agentic)**: ~25-30 minutes for evaluation + <1 minute for analysis  
- **Section C (Combined)**: <1 minute for combined analysis
- **Total procedure time**: ~45-55 minutes

## Paper Reporting Guidelines

For consistent academic reporting, extract the following from analysis JSON files:

1. **Sample size**: 58 translation pairs × 3 runs = 174 evaluations per mode
2. **Statistical significance**: Report p-values for correlation analysis
3. **Effect sizes**: Include both correlation coefficients and classification accuracy
4. **Consistency measures**: Report both pass consistency and score standard deviation
5. **Comparative analysis**: Direct comparison between agentic vs non-agentic performance across all metrics