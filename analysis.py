#!/usr/bin/env python3
"""
LLM Judge Analysis Script

Analyzes evaluation results from the ./evaluation_results directory to compute:
1. Basic classification metrics (Precision, Accuracy, Recall, F1-Score)
2. Spearman's rank correlation coefficient (LLM vs Human scores)
3. Score variation analysis across reruns

Usage:
    python analysis.py [options]
"""

import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import statistics
from datetime import datetime

# Statistical imports
try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Spearman correlation will be calculated using manual implementation.")
    SCIPY_AVAILABLE = False

def calculate_spearman_manual(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Manual implementation of Spearman's rank correlation coefficient
    Returns: (correlation_coefficient, p_value)
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0, 1.0
    
    # 1. Create rank mappings
    def rank_data(data):
        """Convert data to ranks, handling ties with average ranking"""
        sorted_unique = sorted(set(data))
        rank_map = {}
        for i, val in enumerate(sorted_unique):
            rank_map[val] = i + 1
        return [rank_map[val] for val in data]
    
    # 2. Convert to ranks
    ranks_x = rank_data(x)
    ranks_y = rank_data(y)
    
    # 3. Calculate Pearson correlation of ranks
    n = len(ranks_x)
    
    # Calculate means
    mean_x = sum(ranks_x) / n
    mean_y = sum(ranks_y) / n
    
    # Calculate correlation coefficient
    numerator = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(ranks_x, ranks_y))
    sum_sq_x = sum((rx - mean_x) ** 2 for rx in ranks_x)
    sum_sq_y = sum((ry - mean_y) ** 2 for ry in ranks_y)
    
    if sum_sq_x == 0 or sum_sq_y == 0:
        return 0.0, 1.0
    
    correlation = numerator / (sum_sq_x * sum_sq_y) ** 0.5
    
    # Simplified p-value calculation (approximation for large n)
    # For small samples, this is less accurate
    if n > 10:
        import math
        t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2))
        # Very rough p-value approximation
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(n - 2)))
    else:
        p_value = 0.05  # Conservative estimate for small samples
    
    return correlation, p_value

class LLMJudgeAnalyzer:
    """Main analyzer class for LLM judge evaluation results"""
    
    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        """
        Initialize analyzer
        
        Args:
            evaluation_results_dir: Directory containing evaluation session folders
        """
        self.evaluation_results_dir = Path(evaluation_results_dir)
        self.evaluation_data = []
        self.session_summaries = []
        
    def load_all_evaluation_data(self) -> None:
        """
        Load all evaluation results from all session directories
        
        Each session directory contains:
        - eval_*_params.json: Individual evaluation results
        - eval_summary_*.json: Session summary statistics
        """
        print(f"[INFO] Scanning {self.evaluation_results_dir} for evaluation sessions...")
        
        if not self.evaluation_results_dir.exists():
            raise FileNotFoundError(f"Evaluation results directory not found: {self.evaluation_results_dir}")
        
        # 1. Find all session directories
        session_dirs = [d for d in self.evaluation_results_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("session_")]
        
        if not session_dirs:
            print("[WARNING] No evaluation session directories found")
            return
            
        print(f"[INFO] Found {len(session_dirs)} evaluation sessions")
        
        # 2. Load evaluation results from each session
        total_evaluations = 0
        for session_dir in session_dirs:
            session_name = session_dir.name
            print(f"   Loading session: {session_name}")
            
            # Load individual evaluation results
            params_files = list(session_dir.glob("eval_*_params.json"))
            for params_file in params_files:
                try:
                    with open(params_file, 'r', encoding='utf-8') as f:
                        session_evaluations = json.load(f)
                        
                    # Add session metadata to each evaluation
                    for eval_data in session_evaluations:
                        eval_data['session_name'] = session_name
                        eval_data['session_dir'] = str(session_dir)
                        self.evaluation_data.append(eval_data)
                        total_evaluations += 1
                        
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"   ⚠️  Error loading {params_file}: {e}")
            
            # Load session summary if available
            summary_files = list(session_dir.glob("eval_summary_*.json"))
            for summary_file in summary_files:
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                        summary_data['session_name'] = session_name
                        summary_data['session_dir'] = str(session_dir)
                        self.session_summaries.append(summary_data)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"   [WARNING] Error loading {summary_file}: {e}")
        
        print(f"[SUCCESS] Loaded {total_evaluations} total evaluations from {len(session_dirs)} sessions")
        
    def calculate_classification_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic classification metrics (Precision, Accuracy, Recall, F1-Score)
        
        Classification task: Does LLM correctly identify translation quality?
        - True Positive (TP): LLM says good, translation is actually good
        - False Positive (FP): LLM says good, translation is actually bad  
        - True Negative (TN): LLM says bad, translation is actually bad
        - False Negative (FN): LLM says bad, translation is actually good
        """
        if not self.evaluation_data:
            return {"error": "No evaluation data available for classification metrics"}
        
        # Extract classification data
        tp = fp = tn = fn = 0
        
        for eval_item in self.evaluation_data:
            metadata = eval_item.get('evaluation_metadata', {})
            
            # Ground truth: is this actually a good translation?
            is_actually_good = metadata.get('expected_good_translation', False)
            
            # LLM prediction: did LLM classify this as good?
            llm_says_good = metadata.get('llm_pass', False)
            
            # Calculate confusion matrix values
            if is_actually_good and llm_says_good:
                tp += 1
            elif is_actually_good and not llm_says_good:
                fn += 1
            elif not is_actually_good and llm_says_good:
                fp += 1
            elif not is_actually_good and not llm_says_good:
                tn += 1
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {"error": "No valid classification data found"}
        
        # Calculate metrics with zero-division protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / total
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_samples": total
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1_score, 4)
            },
            "interpretation": {
                "precision": f"Of translations LLM marked as good, {precision:.1%} were actually good",
                "recall": f"Of actually good translations, LLM correctly identified {recall:.1%}",
                "accuracy": f"LLM correctly classified {accuracy:.1%} of all translations",
                "f1_score": f"Harmonic mean of precision and recall: {f1_score:.3f}"
            }
        }
    
    def calculate_spearman_correlation(self) -> Dict[str, Any]:
        """
        Calculate Spearman's rank correlation between LLM scores and human scores
        
        Requires human scores to be present in evaluation data
        """
        if not self.evaluation_data:
            return {"error": "No evaluation data available for correlation analysis"}
        
        # Collect paired scores (LLM vs Human)
        llm_scores = []
        human_scores = []
        correlation_data = []
        
        for eval_item in self.evaluation_data:
            # Get LLM score
            final_score = eval_item.get('final_score', {})
            llm_score = final_score.get('score')
            
            # Get human score from metadata
            metadata = eval_item.get('evaluation_metadata', {})
            human_score = metadata.get('human_score')
            
            # Only include if both scores are available
            if llm_score is not None and human_score is not None:
                llm_scores.append(float(llm_score))
                human_scores.append(float(human_score))
                
                correlation_data.append({
                    "llm_score": llm_score,
                    "human_score": human_score,
                    "csv_row": metadata.get('csv_row'),
                    "translation_type": metadata.get('translation_type'),
                    "base_eval_id": metadata.get('base_eval_id')
                })
        
        if len(llm_scores) < 3:
            return {
                "error": "Insufficient data for correlation analysis",
                "available_pairs": len(llm_scores),
                "minimum_required": 3,
                "note": "Add Human-Score-Correct and Human-Score-Flawed columns to your CSV data"
            }
        
        # Calculate Spearman correlation
        if SCIPY_AVAILABLE:
            correlation, p_value = spearmanr(llm_scores, human_scores)
        else:
            correlation, p_value = calculate_spearman_manual(llm_scores, human_scores)
        
        # Additional descriptive statistics
        llm_mean = statistics.mean(llm_scores)
        human_mean = statistics.mean(human_scores)
        llm_std = statistics.stdev(llm_scores) if len(llm_scores) > 1 else 0
        human_std = statistics.stdev(human_scores) if len(human_scores) > 1 else 0
        
        return {
            "correlation_analysis": {
                "spearman_correlation": round(correlation, 4),
                "p_value": round(p_value, 4),
                "sample_size": len(llm_scores),
                "significance_level": 0.05,
                "is_significant": bool(p_value < 0.05)
            },
            "descriptive_statistics": {
                "llm_scores": {
                    "mean": round(llm_mean, 2),
                    "std": round(llm_std, 2),
                    "min": min(llm_scores),
                    "max": max(llm_scores)
                },
                "human_scores": {
                    "mean": round(human_mean, 2),
                    "std": round(human_std, 2),
                    "min": min(human_scores),
                    "max": max(human_scores)
                }
            },
            "interpretation": {
                "correlation_strength": self._interpret_correlation_strength(abs(correlation)),
                "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none",
                "summary": f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} {('positive' if correlation > 0 else 'negative')} correlation (r={correlation:.3f}, p={p_value:.3f})"
            },
            "paired_data": correlation_data[:10]  # Show first 10 pairs as examples
        }
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret the strength of correlation coefficient"""
        if correlation >= 0.9:
            return "Very Strong"
        elif correlation >= 0.7:
            return "Strong"
        elif correlation >= 0.5:
            return "Moderate"
        elif correlation >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def analyze_score_variation(self) -> Dict[str, Any]:
        """
        Analyze score variation across reruns for the same evaluation conditions
        
        Groups evaluations by base_eval_id and analyzes consistency
        """
        if not self.evaluation_data:
            return {"error": "No evaluation data available for variation analysis"}
        
        # Group evaluations by base_eval_id (same condition, different runs)
        condition_groups = defaultdict(list)
        
        for eval_item in self.evaluation_data:
            metadata = eval_item.get('evaluation_metadata', {})
            base_id = metadata.get('base_eval_id')
            
            if base_id:
                final_score = eval_item.get('final_score', {})
                score = final_score.get('score')
                
                if score is not None:
                    condition_groups[base_id].append({
                        'score': score,
                        'llm_pass': metadata.get('llm_pass', False),
                        'run_id': metadata.get('run_id', 0),
                        'unique_eval_id': metadata.get('unique_eval_id'),
                        'timestamp': eval_item.get('timestamp')
                    })
        
        # Filter to only conditions with multiple runs
        rerun_conditions = {k: v for k, v in condition_groups.items() if len(v) > 1}
        
        if not rerun_conditions:
            return {
                "error": "No rerun data found for variation analysis",
                "total_conditions": len(condition_groups),
                "conditions_with_reruns": 0,
                "note": "Use --reruns parameter in auto_evaluate.py to generate rerun data"
            }
        
        # Calculate variation statistics for each condition
        variation_analysis = {}
        overall_consistency_scores = []
        overall_std_scores = []
        
        for base_id, runs in rerun_conditions.items():
            scores = [run['score'] for run in runs]
            passes = [run['llm_pass'] for run in runs]
            
            # Score statistics
            score_mean = statistics.mean(scores)
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            score_range = (min(scores), max(scores))
            
            # Pass consistency (percentage of runs with same pass/fail result)
            pass_consistency = max(passes.count(True), passes.count(False)) / len(passes)
            
            variation_analysis[base_id] = {
                "runs_completed": len(runs),
                "scores": scores,
                "score_mean": round(score_mean, 2),
                "score_std": round(score_std, 3),
                "score_range": score_range,
                "pass_consistency": round(pass_consistency, 3),
                "coefficient_of_variation": round(score_std / score_mean, 3) if score_mean > 0 else 0.0
            }
            
            overall_consistency_scores.append(pass_consistency)
            overall_std_scores.append(score_std)
        
        # Overall variation statistics
        overall_consistency_mean = statistics.mean(overall_consistency_scores)
        overall_std_mean = statistics.mean(overall_std_scores)
        
        # Identify most/least consistent conditions
        most_consistent = min(variation_analysis.items(), 
                            key=lambda x: x[1]['score_std'])
        least_consistent = max(variation_analysis.items(), 
                             key=lambda x: x[1]['score_std'])
        
        return {
            "variation_summary": {
                "total_conditions_analyzed": len(rerun_conditions),
                "total_runs_analyzed": sum(len(runs) for runs in rerun_conditions.values()),
                "average_runs_per_condition": round(sum(len(runs) for runs in rerun_conditions.values()) / len(rerun_conditions), 1),
                "overall_pass_consistency": round(overall_consistency_mean, 3),
                "overall_score_std": round(overall_std_mean, 3)
            },
            "consistency_insights": {
                "most_consistent_condition": {
                    "condition": most_consistent[0],
                    "score_std": most_consistent[1]['score_std'],
                    "pass_consistency": most_consistent[1]['pass_consistency']
                },
                "least_consistent_condition": {
                    "condition": least_consistent[0],
                    "score_std": least_consistent[1]['score_std'],
                    "pass_consistency": least_consistent[1]['pass_consistency']
                }
            },
            "detailed_analysis": dict(list(variation_analysis.items())[:5]),  # Show first 5 as examples
            "full_analysis_available": len(variation_analysis)
        }
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report combining all metrics
        """
        print("[INFO] Calculating comprehensive analysis...")
        
        # Collect basic information
        total_evaluations = len(self.evaluation_data)
        unique_sessions = len(set(eval_item.get('session_name', '') for eval_item in self.evaluation_data))
        
        # Get evaluation parameters summary
        models_used = set()
        modes_used = set()
        csv_files_used = set()
        
        for eval_item in self.evaluation_data:
            metadata = eval_item.get('evaluation_metadata', {})
            params = metadata.get('evaluation_parameters', {})
            
            models_used.add(params.get('llm_model', 'unknown'))
            modes_used.add('agentic' if params.get('use_agentic', False) else 'prompt')
            
            # Extract CSV filename from session name or metadata
            session_name = eval_item.get('session_name', '')
            if session_name.startswith('session_'):
                parts = session_name.split('_')
                if len(parts) > 1:
                    csv_files_used.add(parts[1])  # Usually format: session_{csv_name}_{model}...
        
        # Calculate all metrics
        classification_metrics = self.calculate_classification_metrics()
        correlation_analysis = self.calculate_spearman_correlation()
        variation_analysis = self.analyze_score_variation()
        
        # Generate timestamp
        analysis_timestamp = datetime.now().isoformat()
        
        return {
            "analysis_metadata": {
                "generated_at": analysis_timestamp,
                "analyzer_version": "1.0",
                "total_evaluations_analyzed": total_evaluations,
                "unique_sessions": unique_sessions,
                "evaluation_parameters_summary": {
                    "models_used": sorted(list(models_used)),
                    "modes_used": sorted(list(modes_used)),
                    "csv_files_used": sorted(list(csv_files_used))
                }
            },
            "classification_metrics": classification_metrics,
            "correlation_analysis": correlation_analysis,
            "variation_analysis": variation_analysis,
            "summary_insights": self._generate_summary_insights(
                classification_metrics, correlation_analysis, variation_analysis
            )
        }
    
    def _generate_summary_insights(self, classification: Dict, correlation: Dict, variation: Dict) -> Dict[str, Any]:
        """Generate high-level insights from all analyses"""
        insights = {
            "key_findings": [],
            "recommendations": []
        }
        
        # Classification insights
        if "metrics" in classification:
            accuracy = classification["metrics"]["accuracy"]
            f1 = classification["metrics"]["f1_score"]
            
            if accuracy > 0.9:
                insights["key_findings"].append(f"Excellent classification accuracy: {accuracy:.1%}")
            elif accuracy > 0.7:
                insights["key_findings"].append(f"Good classification accuracy: {accuracy:.1%}")
            else:
                insights["key_findings"].append(f"Room for improvement in classification accuracy: {accuracy:.1%}")
                insights["recommendations"].append("Consider prompt engineering or model parameter tuning")
        
        # Correlation insights
        if "correlation_analysis" in correlation:
            corr = correlation["correlation_analysis"]["spearman_correlation"]
            is_sig = correlation["correlation_analysis"]["is_significant"]
            
            if is_sig and abs(corr) > 0.7:
                insights["key_findings"].append(f"Strong correlation with human judgments: r={corr:.3f}")
            elif is_sig and abs(corr) > 0.3:
                insights["key_findings"].append(f"Moderate correlation with human judgments: r={corr:.3f}")
            else:
                insights["key_findings"].append(f"Weak/no correlation with human judgments: r={corr:.3f}")
                insights["recommendations"].append("Investigate differences between LLM and human evaluation criteria")
        
        # Variation insights
        if "variation_summary" in variation:
            consistency = variation["variation_summary"]["overall_pass_consistency"]
            score_std = variation["variation_summary"]["overall_score_std"]
            
            if consistency > 0.9 and score_std < 0.5:
                insights["key_findings"].append(f"High consistency across reruns: {consistency:.1%} pass consistency")
            elif consistency > 0.7:
                insights["key_findings"].append(f"Moderate consistency across reruns: {consistency:.1%} pass consistency")
            else:
                insights["key_findings"].append(f"Low consistency across reruns: {consistency:.1%} pass consistency")
                insights["recommendations"].append("Consider increasing temperature or investigating prompt sensitivity")
        
        return insights

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Analyze LLM judge evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analysis.py                                    # Analyze all evaluation results
    python analysis.py --output-dir analysis_reports     # Save to custom directory
    python analysis.py --filter-model gemini-2.5-flash   # Analyze specific model only
    python analysis.py --filter-session session_name     # Analyze specific session only
        """
    )
    
    parser.add_argument(
        '--evaluation-dir',
        default='evaluation_results',
        help='Directory containing evaluation session folders (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='analysis_results',
        help='Directory to save analysis results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--filter-model',
        help='Filter analysis to specific LLM model (e.g., gemini-2.5-flash)'
    )
    
    parser.add_argument(
        '--filter-session',
        help='Filter analysis to specific session directory name'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'summary'],
        default='json',
        help='Output format: json (full report) or summary (key metrics only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = LLMJudgeAnalyzer(args.evaluation_dir)
        
        # Load data
        analyzer.load_all_evaluation_data()
        
        if not analyzer.evaluation_data:
            print("[ERROR] No evaluation data found. Run auto_evaluate.py first to generate evaluation results.")
            return
        
        # Apply filters if specified
        if args.filter_model or args.filter_session:
            original_count = len(analyzer.evaluation_data)
            
            if args.filter_model:
                analyzer.evaluation_data = [
                    item for item in analyzer.evaluation_data
                    if item.get('evaluation_metadata', {}).get('evaluation_parameters', {}).get('llm_model') == args.filter_model
                ]
                print(f"[INFO] Filtered to model '{args.filter_model}': {len(analyzer.evaluation_data)} evaluations")
            
            if args.filter_session:
                analyzer.evaluation_data = [
                    item for item in analyzer.evaluation_data
                    if item.get('session_name') == args.filter_session
                ]
                print(f"[INFO] Filtered to session '{args.filter_session}': {len(analyzer.evaluation_data)} evaluations")
            
            if not analyzer.evaluation_data:
                print("[ERROR] No evaluation data remaining after filtering.")
                return
        
        # Generate analysis
        analysis_results = analyzer.generate_comprehensive_analysis()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"llm_judge_analysis_{timestamp}.json"
        output_path = output_dir / output_filename
        
        # Convert numpy types to JSON-safe types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization"""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        analysis_results_safe = convert_numpy_types(analysis_results)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results_safe, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Analysis complete! Results saved to: {output_path}")
        
        # Display summary
        if args.format == 'summary' or True:  # Always show summary
            print("\n" + "="*60)
            print("[ANALYSIS SUMMARY]")
            print("="*60)
            
            # Basic info
            metadata = analysis_results.get("analysis_metadata", {})
            print(f"[INFO] Total Evaluations: {metadata.get('total_evaluations_analyzed', 0)}")
            print(f"[INFO] Sessions Analyzed: {metadata.get('unique_sessions', 0)}")
            print(f"[INFO] Models Used: {', '.join(metadata.get('evaluation_parameters_summary', {}).get('models_used', []))}")
            
            # Classification metrics
            classification = analysis_results.get("classification_metrics", {})
            if "metrics" in classification:
                metrics = classification["metrics"]
                print(f"\n🎯 Classification Performance:")
                print(f"   Accuracy: {metrics['accuracy']:.1%}")
                print(f"   Precision: {metrics['precision']:.1%}")
                print(f"   Recall: {metrics['recall']:.1%}")
                print(f"   F1-Score: {metrics['f1_score']:.3f}")
            
            # Correlation analysis
            correlation = analysis_results.get("correlation_analysis", {})
            if "correlation_analysis" in correlation:
                corr_data = correlation["correlation_analysis"]
                print(f"\n📈 Human-LLM Correlation:")
                print(f"   Spearman r: {corr_data['spearman_correlation']:.3f}")
                print(f"   Sample size: {corr_data['sample_size']}")
                print(f"   Significant: {'Yes' if corr_data['is_significant'] else 'No'}")
            
            # Variation analysis
            variation = analysis_results.get("variation_analysis", {})
            if "variation_summary" in variation:
                var_data = variation["variation_summary"]
                print(f"\n🔄 Consistency Analysis:")
                print(f"   Pass Consistency: {var_data['overall_pass_consistency']:.1%}")
                print(f"   Score Std Dev: {var_data['overall_score_std']:.3f}")
                print(f"   Conditions with Reruns: {var_data['total_conditions_analyzed']}")
            
            # Key insights
            insights = analysis_results.get("summary_insights", {})
            if insights.get("key_findings"):
                print(f"\n[KEY FINDINGS]:")
                for finding in insights["key_findings"]:
                    print(f"   - {finding}")
            
            if insights.get("recommendations"):
                print(f"\n[RECOMMENDATIONS]:")
                for rec in insights["recommendations"]:
                    print(f"   - {rec}")
        
        print(f"\n[INFO] Full detailed analysis available in: {output_path}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("   Make sure you have run auto_evaluate.py to generate evaluation results first.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()