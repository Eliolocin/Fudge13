#!/usr/bin/env python3
"""
Automated LLM Judge Evaluation Script

This script systematically evaluates LLM judge performance against human-labeled 
translations from a 3-column CSV dataset (English, Translation, Final Score). 
It supports both agentic and non-agentic judge modes with comprehensive error 
handling and result analysis.

Usage:
    python auto_evaluate.py --csv_file data.csv --row_start 1 --row_end 100 
                           --use_agentic --llm_model gemini-2.5-flash
"""

import argparse
import pandas as pd
import json
import os
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Suppress harmless Pydantic warnings from google-genai library
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent", category=UserWarning)

# Import existing components
from llms.prompt_engineered_judge_main import create_llm_client, TranslationJudgment
from llms.agentic_judge_main import create_agentic_llm_client
from utils.prompt_utils import replace_tokens, validate_tokens
from utils.eval_utils import calculate_final_score


class AutoEvaluator:
    """
    Automated evaluation system for LLM-as-Judge translation assessment.
    
    Processes CSV datasets with human-labeled translation pairs and evaluates
    LLM judge performance with configurable parameters and robust error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoEvaluator with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.results = []
        self.progress_log = []
        self.start_time = datetime.now()
        
        # Generate session identifier for unique evaluation tracking
        self.session_id = self._generate_session_id()
        
        # Set up logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize LLM client
        self.llm_client = self._create_llm_client()
        
        # Rate limiting parameters
        self.retry_delays = [15, 30, 60]  # Exponential backoff: 15s, 30s, 60s
        self.max_retries = len(self.retry_delays)
        
        self.logger.info(f"AutoEvaluator initialized with config: {self.config}")
        self.logger.info(f"Session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session identifier that includes key evaluation parameters.
        
        This helps prevent conflicts when analyzing results from different evaluation runs
        with different parameters, models, or datasets.
        
        Returns:
            str: Unique session identifier (e.g., "data_gemini-2.5-flash_agentic_20250721_144940")
        """
        # Extract CSV filename (without path and extension)
        csv_path = Path(self.config['csv_file'])
        csv_name = csv_path.stem
        
        # Build model identifier
        model_name = self.config.get('llm_model', 'gemini-2.5-flash')
        # Clean model name for filename safety
        model_clean = model_name.replace('.', '-').replace('_', '-')
        
        # Mode identifier
        mode = 'agentic' if self.config.get('use_agentic', False) else 'prompt'
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build session ID components (translation_type removed in 3-column format)
        components = [csv_name, model_clean, mode]
        
        # Add row range if specified
        row_start = self.config.get('row_start')
        row_end = self.config.get('row_end')
        if row_start != 1 or row_end is not None:
            range_str = f"rows{row_start}"
            if row_end is not None:
                range_str += f"-{row_end}"
            else:
                range_str += "-end"
            components.append(range_str)
        
        # Add reruns if specified
        reruns = self.config.get('reruns', 0)
        if reruns > 0:
            components.append(f"reruns{reruns}")
        
        # Combine with timestamp
        session_parts = '_'.join(components) + f"_{timestamp}"
        
        return session_parts
    
    def _generate_base_eval_id(self, row_num: int) -> str:
        """
        Generate parameter-aware base evaluation ID for grouping runs.
        
        This ensures that evaluations with different parameters (models, modes, datasets)
        are properly separated in analysis, preventing incorrect grouping.
        
        Args:
            row_num (int): CSV row number
            
        Returns:
            str: Enhanced base evaluation ID that includes key parameters
        """
        # Extract key identifying components
        csv_name = Path(self.config['csv_file']).stem
        model_clean = self.config.get('llm_model', 'gemini-2.5-flash').replace('.', '-').replace('_', '-')
        mode = 'agentic' if self.config.get('use_agentic', False) else 'prompt'
        
        # Build base evaluation ID (translation_type removed in 3-column format)
        base_id = f"{csv_name}_row{row_num}_{model_clean}_{mode}"
        
        return base_id
    
    def _setup_logging(self):
        """Set up comprehensive logging with session-specific directory organization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base evaluation results directory
        base_output_dir = Path(self.config.get('output_dir', 'evaluation_results'))
        base_output_dir.mkdir(exist_ok=True)
        
        # Create session-specific subdirectory
        session_dir_name = f"session_{self.session_id}"
        self.output_dir = base_output_dir / session_dir_name
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logger with session-specific log file
        log_file = self.output_dir / f"eval_progress_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.timestamp = timestamp
        self.log_file = log_file
        
        self.logger.info(f"Session directory created: {self.output_dir}")
        self.logger.info(f"All evaluation files will be saved to: {self.output_dir}")
    
    def _validate_config(self):
        """Validate configuration parameters and raise errors for invalid values."""
        required_fields = ['csv_file']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Required configuration field missing: {field}")
        
        # Validate CSV file exists
        csv_path = Path(self.config['csv_file'])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Translation type validation removed - 3-column format processes single translations only
        
        # Validate model name format
        llm_model = self.config.get('llm_model', 'gemini-2.5-flash')
        if not isinstance(llm_model, str) or len(llm_model) == 0:
            raise ValueError(f"Invalid llm_model: {llm_model}")
        
        self.logger.info("Configuration validation passed")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from prompts directory."""
        prompt_file = Path('prompts/basic_translation_judge.txt')
        if not prompt_file.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        # Validate prompt contains required tokens
        validation = validate_tokens(prompt)
        if not validation['valid']:
            raise ValueError(f"System prompt missing required tokens: {validation['missing_tokens']}")
        
        self.logger.info(f"System prompt loaded from {prompt_file}")
        return prompt
    
    def _create_llm_client(self):
        """Create appropriate LLM client based on configuration."""
        use_agentic = self.config.get('use_agentic', False)
        llm_model = self.config.get('llm_model', 'gemini-2.5-flash')
        
        try:
            if use_agentic:
                self.logger.info(f"Creating agentic LLM client with model: {llm_model}")
                # For agentic mode, always use Google provider
                client = create_agentic_llm_client('google', llm_model)
                if client is None:
                    raise ValueError("Failed to create agentic LLM client")
                return client
            else:
                self.logger.info(f"Creating prompt-engineered LLM client with model: {llm_model}")
                # For non-agentic mode, use Google provider with the specified model
                client = create_llm_client('google', llm_model)
                if client is None:
                    raise ValueError("Failed to create prompt-engineered LLM client")
                return client
                
        except Exception as e:
            self.logger.error(f"Error creating LLM client: {e}")
            raise
    
    def load_csv_data(self) -> pd.DataFrame:
        """
        Load and validate CSV data with human-labeled translation pairs.
        
        Returns:
            pd.DataFrame: Loaded CSV data with required columns
        """
        try:
            df = pd.read_csv(self.config['csv_file'])
            self.logger.info(f"CSV loaded with {len(df)} rows")
            
            # Validate required columns for 3-column format
            required_columns = ['English', 'Translation', 'Final Score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")
            
            # Validate Final Score column values (1-5 integers)
            final_score_col = 'Final Score'
            invalid_scores = df[~df[final_score_col].between(1, 5) | ~df[final_score_col].apply(lambda x: isinstance(x, (int, float)) and x == int(x))]
            if not invalid_scores.empty:
                raise ValueError(f"Invalid scores in 'Final Score' column: must be integers 1-5")
            
            # Human scores are always available in the new format
            has_human_scores = True
            self.logger.info("Final Score column detected - Spearman correlation analysis will be available")
            
            self.has_human_scores = has_human_scores
            
            # Apply row range filters
            row_start = self.config.get('row_start', 1) - 1  # Convert to 0-based index
            row_end = self.config.get('row_end')
            
            # Handle None value for row_end (when not specified in command line)
            if row_end is None:
                row_end = len(df)
            
            if row_start < 0 or row_start >= len(df):
                raise ValueError(f"row_start ({row_start + 1}) is out of range")
            
            if row_end <= row_start or row_end > len(df):
                row_end = len(df)
            
            df_filtered = df.iloc[row_start:row_end].copy()
            df_filtered['original_row_index'] = df_filtered.index + 1  # Store 1-based original row numbers
            
            self.logger.info(f"Applied row range filter: rows {row_start + 1} to {row_end}, {len(df_filtered)} rows to process")
            
            return df_filtered
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e}")
            raise
    
    def evaluate_translation_pair(self, source_text: str, fil_translation: str, 
                                expected_good: bool, row_num: int, 
                                run_id: int = 0, 
                                run_total: int = 1, base_eval_id: str = None, 
                                unique_eval_id: str = None, human_score: int = None) -> Dict[str, Any]:
        """
        Evaluate a single English-to-Filipino translation.
        
        Args:
            source_text (str): English source text
            fil_translation (str): Filipino translation to evaluate
            expected_good (bool): Whether translation is expected to be good
            row_num (int): Original CSV row number  
            run_id (int): Run number (0=original, 1+=reruns)
            run_total (int): Total number of runs planned for this condition
            base_eval_id (str): Base identifier grouping all runs of same condition
            unique_eval_id (str): Unique identifier for this specific run
            human_score (int, optional): Human-labeled score (1-5) if available
            
        Returns:
            Dict[str, Any]: Complete evaluation result with pass/fail analysis and run tracking
        """
        run_info = f"run {run_id + 1}/{run_total}" if run_total > 1 else ""
        self.logger.info(f"Evaluating row {row_num} translation {run_info}".strip())
        
        # Prepare prompt with token replacement
        filled_prompt = replace_tokens(
            self.system_prompt,
            source_text=source_text.strip(),
            fil_translation=fil_translation.strip(),
            ref_translation=""  # No reference translation for this evaluation
        )
        
        # Attempt LLM evaluation with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"LLM call attempt {attempt + 1} for row {row_num}")
                
                # Call LLM judge
                llm_response = self.llm_client.generate_judgment(filled_prompt)
                
                if not llm_response.get('success', False):
                    raise Exception(f"LLM call failed: {llm_response.get('error', 'Unknown error')}")
                
                # Calculate final score using existing utility
                final_score = calculate_final_score(llm_response['data'])
                
                # Determine pass/fail based on expected outcome
                llm_score = final_score['score']
                if expected_good:
                    # For correct translations, pass if score >= 3
                    llm_pass = llm_score >= 3
                else:
                    # For flawed translations, pass if score < 3  
                    llm_pass = llm_score < 3
                
                # Construct complete result object
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "request_data": {
                        "source_text": source_text.strip(),
                        "fil_translation": fil_translation.strip(),
                        "ref_translation": "",
                        "system_prompt": self.system_prompt,
                        "raw_prompt": filled_prompt,
                        "llm_provider": "google",
                        "llm_model": self.config.get('llm_model', 'gemini-2.5-flash')
                    },
                    "llm_response": llm_response,
                    "final_score": final_score,
                    "evaluation_metadata": {
                        "csv_row": row_num,
                        "expected_good_translation": expected_good,
                        "llm_pass": llm_pass,
                        "use_agentic": self.config.get('use_agentic', False),
                        "run_id": run_id,
                        "run_total": run_total,
                        "base_eval_id": base_eval_id,
                        "unique_eval_id": unique_eval_id,
                        "human_score": human_score,
                        "has_human_scores": self.has_human_scores,
                        "evaluation_parameters": {
                            "row_start": self.config.get('row_start', 1),
                            "row_end": self.config.get('row_end'),
                            "llm_model": self.config.get('llm_model', 'gemini-2.5-flash'),
                            "use_agentic": self.config.get('use_agentic', False),
                            "reruns": self.config.get('reruns', 0)
                        }
                    },
                    "metadata": {
                        "app_version": "1.0",
                        "filename": f"auto_eval_{self.timestamp}_{row_num}_run{run_id}.json",
                        "saved_at": datetime.now().isoformat()
                    }
                }
                
                run_info = f" run {run_id + 1}/{run_total}" if run_total > 1 else ""
                self.logger.info(f"Row {row_num}{run_info}: Score={llm_score}, Pass={llm_pass}")
                return result
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for row {row_num}: {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delays[attempt]
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts failed for row {row_num}")
                    raise
    
    def save_incremental_results(self):
        """Save current results to prevent data loss."""
        if not self.results:
            return
        
        results_file = self.output_dir / f"eval_{self.timestamp}_params.json"
        config_file = self.output_dir / f"eval_config_{self.timestamp}.json"
        
        # Save main results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Incremental results saved: {len(self.results)} evaluations")
    
    def run_evaluation(self):
        """
        Execute the complete evaluation process.
        
        Main entry point that orchestrates CSV loading, evaluation loop,
        and result compilation with comprehensive error handling.
        """
        try:
            self.logger.info("Starting automated LLM judge evaluation")
            
            # Load CSV data
            df = self.load_csv_data()
            
            # Create evaluation plan for 3-column format (one translation per row)
            evaluation_plan = []
            for idx, row in df.iterrows():
                row_num = row['original_row_index']
                source_text = str(row['English'])
                fil_translation = str(row['Translation'])
                final_score = int(row['Final Score'])
                
                # Determine if translation is expected to be good based on threshold
                # Score >= 3 means "good translation" (3, 4, 5 are considered good)
                expected_good = final_score >= 3
                
                eval_item = {
                    'row_num': row_num,
                    'source_text': source_text,
                    'fil_translation': fil_translation,
                    'expected_good': expected_good,
                    'human_score': final_score
                }
                
                evaluation_plan.append(eval_item)
            
            # Calculate total evaluations including reruns
            reruns = self.config.get('reruns', 0)
            total_runs_per_eval = reruns + 1
            total_evaluations = len(evaluation_plan) * total_runs_per_eval
            
            self.logger.info(f"Evaluation plan: {len(evaluation_plan)} conditions × {total_runs_per_eval} runs = {total_evaluations} total evaluations")
            
            # Execute evaluations with immediate reruns per condition
            eval_counter = 0
            for condition_idx, eval_item in enumerate(evaluation_plan, 1):
                # Generate parameter-aware base evaluation ID for grouping runs
                base_eval_id = self._generate_base_eval_id(eval_item['row_num'])
                
                self.logger.info(f"Processing condition {condition_idx}/{len(evaluation_plan)}: Row {eval_item['row_num']} - {total_runs_per_eval} runs")
                
                # Execute original + reruns for this condition
                for run_id in range(total_runs_per_eval):
                    eval_counter += 1
                    unique_eval_id = f"{base_eval_id}_run{run_id}"
                    
                    try:
                        run_info = f"run {run_id + 1}/{total_runs_per_eval}" if total_runs_per_eval > 1 else ""
                        self.logger.info(f"Progress: {eval_counter}/{total_evaluations} - Row {eval_item['row_num']} {run_info}".strip())
                        
                        # Execute evaluation with run tracking parameters
                        result = self.evaluate_translation_pair(
                            **eval_item,
                            run_id=run_id,
                            run_total=total_runs_per_eval,
                            base_eval_id=base_eval_id,
                            unique_eval_id=unique_eval_id
                        )
                        self.results.append(result)
                        
                        # Save incremental results every 10 evaluations
                        if eval_counter % 10 == 0:
                            self.save_incremental_results()
                        
                    except Exception as e:
                        run_info = f" run {run_id + 1}/{total_runs_per_eval}" if total_runs_per_eval > 1 else ""
                        self.logger.error(f"Failed to evaluate row {eval_item['row_num']}{run_info}: {e}")
                        # Save current progress before potentially exiting
                        self.save_incremental_results()
                        
                        # Check if this is a rate limiting issue
                        if "rate" in str(e).lower() or "quota" in str(e).lower():
                            self.logger.error("Rate limiting detected. Saving results and exiting.")
                            return  # Exit the entire evaluation
                        
                        # For other errors, continue to next run
                        continue
            
            # Final save
            self.save_incremental_results()
            
            # Generate summary statistics
            self._generate_summary()
            
            self.logger.info("Evaluation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            self.save_incremental_results()  # Save whatever we have
            raise
    
    def _generate_summary(self):
        """Generate summary statistics of the evaluation results, accounting for reruns."""
        if not self.results:
            return
        
        total_evaluations = len(self.results)
        good_evaluations = [r for r in self.results if r['evaluation_metadata']['expected_good_translation']]
        bad_evaluations = [r for r in self.results if not r['evaluation_metadata']['expected_good_translation']]
        
        good_passes = sum(1 for r in good_evaluations if r['evaluation_metadata']['llm_pass'])
        bad_passes = sum(1 for r in bad_evaluations if r['evaluation_metadata']['llm_pass'])
        
        # Calculate rerun statistics if applicable
        reruns = self.config.get('reruns', 0)
        unique_conditions = set()
        rerun_stats = {}
        
        if reruns > 0:
            # Group results by base_eval_id to analyze consistency across runs
            from collections import defaultdict
            by_condition = defaultdict(list)
            
            for result in self.results:
                base_id = result['evaluation_metadata']['base_eval_id']
                unique_conditions.add(base_id)
                by_condition[base_id].append(result)
            
            # Calculate consistency statistics
            rerun_stats = {
                "unique_conditions_evaluated": len(unique_conditions),
                "runs_per_condition": reruns + 1,
                "consistency_analysis": {}
            }
            
            # Analyze score consistency for each condition
            for base_id, runs in by_condition.items():
                scores = [r['final_score']['score'] for r in runs]
                passes = [r['evaluation_metadata']['llm_pass'] for r in runs]
                
                rerun_stats["consistency_analysis"][base_id] = {
                    "runs_completed": len(runs),
                    "scores": scores,
                    "score_mean": sum(scores) / len(scores) if scores else 0,
                    "score_std": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5 if len(scores) > 1 else 0,
                    "pass_consistency": sum(passes) / len(passes) if passes else 0,  # % of runs that passed
                    "score_range": [min(scores), max(scores)] if scores else [0, 0]
                }
        
        summary = {
            "evaluation_summary": {
                "total_evaluations": total_evaluations,
                "unique_conditions": len(unique_conditions) if unique_conditions else total_evaluations,
                "runs_per_condition": reruns + 1,
                "good_translations": {
                    "count": len(good_evaluations),
                    "passes": good_passes,
                    "pass_rate": good_passes / len(good_evaluations) if good_evaluations else 0
                },
                "bad_translations": {
                    "count": len(bad_evaluations),
                    "passes": bad_passes,  
                    "pass_rate": bad_passes / len(bad_evaluations) if bad_evaluations else 0
                },
                "overall_accuracy": (good_passes + bad_passes) / total_evaluations if total_evaluations > 0 else 0,
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "config_used": self.config
            }
        }
        
        # Add rerun statistics if applicable
        if rerun_stats:
            summary["rerun_statistics"] = rerun_stats
        
        summary_file = self.output_dir / f"eval_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation summary saved to {summary_file}")
        self.logger.info(f"Overall accuracy: {summary['evaluation_summary']['overall_accuracy']:.2%}")
        
        if reruns > 0:
            self.logger.info(f"Evaluated {len(unique_conditions)} unique conditions with {reruns + 1} runs each")


def parse_arguments():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Automated LLM Judge Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate rows 1-100 from 3-column CSV (English, Translation, Final Score)
  python auto_evaluate.py --csv_file data.csv --row_start 1 --row_end 100
  
  # Evaluate using agentic mode
  python auto_evaluate.py --csv_file data.csv --use_agentic
  
  # Use specific model with custom output directory
  python auto_evaluate.py --csv_file data.csv --llm_model gemini-2.5-pro --output_dir results/
  
  # Evaluate with reruns for consistency analysis (each translation run 3 times total)
  python auto_evaluate.py --csv_file data.csv --reruns 2
  
  # Combine reruns with other options
  python auto_evaluate.py --csv_file data.csv --reruns 2 --use_agentic --row_end 10
        """
    )
    
    parser.add_argument('--csv_file', required=True, type=str,
                       help='Path to CSV file with English, Translation, Final Score columns')
    parser.add_argument('--row_start', type=int, default=1,
                       help='Starting row number (1-based, default: 1)')
    parser.add_argument('--row_end', type=int, default=None,
                       help='Ending row number (1-based, default: last row)')
    parser.add_argument('--use_agentic', action='store_true',
                       help='Use agentic judge instead of prompt-engineered judge')
    parser.add_argument('--llm_model', type=str, default='gemini-2.5-flash',
                       help='LLM model name (default: gemini-2.5-flash)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--reruns', type=int, default=0,
                       help='Number of additional runs per evaluation (default: 0, meaning no reruns)')
    
    return parser.parse_args()


def main():
    """Main entry point for the automated evaluation script."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Convert arguments to configuration dictionary
        config = {
            'csv_file': args.csv_file,
            'row_start': args.row_start,
            'row_end': args.row_end,
            'use_agentic': args.use_agentic,
            'llm_model': args.llm_model,
            'output_dir': args.output_dir,
            'reruns': args.reruns
        }
        
        # Initialize and run evaluation
        evaluator = AutoEvaluator(config)
        evaluator.run_evaluation()
        
        print(f"\nEvaluation completed! Results saved to: {evaluator.output_dir}")
        print(f"Check {evaluator.log_file} for detailed logs")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())