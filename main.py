#!/usr/bin/env python3

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import pandas as pd 
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.resource_manager import cleanup_resources
import gc
from config import INTEGRATED_CONFIG
from knowledge_base import ContextRetriever
from IndicKARMA import (
    AgentState, 
    LLMInference, 
    IndicKARMAProcessor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

for directory in ["logs", "results", "results/knowledge_graphs", "visualizations", "cache"]:
    os.makedirs(directory, exist_ok=True)

class AgenticFramework:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = None
        self.initialized = False
        
        self.start_time = time.time()
        self.metrics = {
            "initialization_time": 0,
            "total_processing_time": 0,
            "tweets_analyzed": 0,
            "avg_latency": 0,
            "p50_latency": 0,
            "p95_latency": 0,
            "p99_latency": 0,
            "latencies": [],
            "ta_only_usage": 0,
            "professor_usage": 0,
            "professor_usage_rate": 0.0
        }

    async def initialize(self):
        if self.initialized:
            return
        
        init_start = time.time()
        
        self.processor = IndicKARMAProcessor(self.config)
        
        await self.processor.initialize()
        
        if self.config["PERFORMANCE_CONFIG"]["cache"]["relation_cache_warmup"]:
            warmup_terms = self.config["PERFORMANCE_CONFIG"]["cache"]["warmup_terms"]
            logger.info(f"Warming up caches with terms: {warmup_terms}")
        
        self.initialized = True
        self.metrics["initialization_time"] = time.time() - init_start
        logger.info(f"System initialized with Context-Aware TA-Teacher architecture in {self.metrics['initialization_time']:.3f} seconds")

    async def analyze_tweet(self, tweet_text: str, tweet_id: str = None) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        
        tweet_data = {
            "tid": tweet_id or f"tweet_{int(time.time())}",
            "tweet": tweet_text
        }
        
        start_time = time.time()
        
        result = await self.processor.process_tweet(tweet_data)
        
        latency = time.time() - start_time
        self.metrics["tweets_analyzed"] += 1
        self.metrics["total_processing_time"] += latency
        self.metrics["avg_latency"] = self.metrics["total_processing_time"] / self.metrics["tweets_analyzed"]
        self.metrics["latencies"].append(latency)
        
        if result.get("professor_used", False):
            self.metrics["professor_usage"] += 1
        else:
            self.metrics["ta_only_usage"] += 1
        
        total_tweets = self.metrics["professor_usage"] + self.metrics["ta_only_usage"]
        self.metrics["professor_usage_rate"] = self.metrics["professor_usage"] / total_tweets if total_tweets > 0 else 0.0
        
        if len(self.metrics["latencies"]) % 10 == 0:
            self._update_percentiles()
        
        result["latency"] = latency
        
        formatted_result = {
            "tid": result.get("tid"),
            "tweet": result.get("tweet"),
            "aggression": {
                "level": result.get("aggression_level_description", result.get("aggression_level")),
                "type": result.get("aggression_type_description", result.get("aggression_type")),
                "justification": result.get("aggression_level_justification", "")
            },
            "bias": {
                "gender": result.get("gender_bias_description", result.get("gender_bias")),
                "religious": result.get("religious_bias_description", result.get("religious_bias")),
                "caste": result.get("caste_bias_description", result.get("caste_bias")),
                "ethnicity": result.get("ethnicity_bias_description", result.get("ethnicity_bias")),
                "justification": result.get("gender_bias_justification", "")
            },
            "explanation": result.get("explanation", ""),
            "latency": latency,
            "model_type": "context_aware_ta_teacher_architecture",
            "professor_used": result.get("professor_used", False),
            "low_confidence_tasks": result.get("low_confidence_tasks", [])
        }
        
        if "ta_predictions" in result:
            formatted_result["ta_predictions"] = result["ta_predictions"]
        if "professor_predictions" in result:
            formatted_result["professor_predictions"] = result["professor_predictions"]
        if "ta_confidences" in result:
            formatted_result["ta_confidences"] = result["ta_confidences"]
        
        if "include_kg" in tweet_data and tweet_data["include_kg"]:
            formatted_result["knowledge_graph"] = result.get("knowledge_graph")
            formatted_result["entities"] = result.get("entities")
        
        return formatted_result
    
    def _update_percentiles(self):
        if self.metrics["latencies"]:
            self.metrics["p50_latency"] = np.percentile(self.metrics["latencies"], 50)
            self.metrics["p95_latency"] = np.percentile(self.metrics["latencies"], 95)
            self.metrics["p99_latency"] = np.percentile(self.metrics["latencies"], 99)
    
    async def process_csv(
        self, 
        filepath: str, 
        output_path: str = None,
        batch_size: int = None,
        limit: int = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        if not self.initialized:
            await self.initialize()
        
        batch_size = batch_size or self.config["PERFORMANCE_CONFIG"]["processing"]["max_batch_size"]
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            
            if limit and limit < len(df):
                df = df.head(limit)
                logger.info(f"Limited to {limit} records")
            
            rename_map = {
                'text': 'tweet',
                'aggression_level': 'AggLevel',
                'aggression_type': 'typeTag',
                'gender_bias': 'GenderLabel',
                'caste_bias': 'CasteLabel',
                'religious_bias': 'ReligionTag',
                'ethnicity_bias': 'EthnicTag'
            }

            if 'text' in df.columns and 'tweet' not in df.columns:
                df = df.rename(columns=rename_map)
                logger.info("Renamed 'text' column to 'tweet'")
            
            tweets = df.to_dict(orient="records")
            
            async for results in self.processor.process_dataset(tweets):
                if output_path:
                    intermediate_path = output_path.replace(".json", f"_intermediate_{len(results)}.json")
                    self._save_results(results, intermediate_path)
                
                self.metrics["tweets_analyzed"] = len(results)
                
                yield results
            
            if output_path:
                self._save_results(results, output_path)
                logger.info(f"Final results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
    
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = {}
        
        key_mapping = {
            "aggression_level": "ground_truth_AggLevel",
            "aggression_type": "ground_truth_typeTag",
            "gender_bias": "ground_truth_GenderLabel",
            "religious_bias": "ground_truth_ReligionTag",
            "caste_bias": "ground_truth_CasteLabel",
            "ethnicity_bias": "ground_truth_EthnicTag",
        }

        y_pred = {k: [] for k in key_mapping.keys()}
        y_test = {k: [] for k in key_mapping.keys()}

        for result in results:
            for result_key, gt_key in key_mapping.items():
                if result_key in result and gt_key in result:
                    pred_value = result[result_key]
                    gt_value = result[gt_key]
                    
                    if result_key == "aggression_type":
                        aggression_level = result.get("aggression_level", "NAG")
                        if aggression_level == "NAG" and pred_value == "N/A":
                            continue
                    
                    y_pred[result_key].append(pred_value)
                    y_test[result_key].append(gt_value)

        for sub_task in y_test:
            true_labels = y_test.get(sub_task, [])
            pred_labels = y_pred.get(sub_task, [])

            if len(true_labels) != len(pred_labels):
                print(f"Warning: Length mismatch in sub-task {sub_task}. Skipping...")
                continue

            filtered_true = []
            filtered_pred = []

            for yt, yp in zip(true_labels, pred_labels):
                if yt in [None, "UNC", ""] or yp in [None, "UNC", ""]:
                    continue
                    
                if sub_task == "aggression_type":
                    if yt == "N/A" and yp == "N/A":
                        continue
                
                filtered_true.append(yt)
                filtered_pred.append(yp)

            if not filtered_true:
                print(f"Warning: No valid samples for sub-task {sub_task}. Skipping...")
                continue

            labels = sorted(set(filtered_true + filtered_pred))
            
            precision, recall, f1, support = precision_recall_fscore_support(
                filtered_true, filtered_pred, labels=labels, zero_division=0
            )

            task_result = {
                "classification_report": classification_report(
                    filtered_true, filtered_pred, labels=labels, output_dict=True, zero_division=0
                ),
                "macro_f1": f1_score(filtered_true, filtered_pred, average="macro", zero_division=0),
                "accuracy": accuracy_score(filtered_true, filtered_pred),
                "labels": labels,
                "sample_count": len(filtered_true)
            }

            metrics[sub_task] = task_result

        all_predictions = []
        all_ground_truth = []
        
        for task in key_mapping.keys():
            task_pred = [str(label) if label is not None else "None" for label in y_pred[task]]
            task_truth = [str(label) if label is not None else "None" for label in y_test[task]]
            all_predictions.extend(task_pred)
            all_ground_truth.extend(task_truth)
        
        overall_accuracy = accuracy_score(all_ground_truth, all_predictions) if all_ground_truth else None
        
        cultural_bias_tasks = ["gender_bias", "religious_bias", "caste_bias", "ethnicity_bias"]
        cultural_bias_predictions = []
        cultural_bias_ground_truth = []
        
        for task in cultural_bias_tasks:
            task_pred = [str(label) if label is not None else "None" for label in y_pred[task]]
            task_truth = [str(label) if label is not None else "None" for label in y_test[task]]
            cultural_bias_predictions.extend(task_pred)
            cultural_bias_ground_truth.extend(task_truth)
        
        cultural_bias_index = accuracy_score(
            cultural_bias_ground_truth, 
            cultural_bias_predictions
        ) if cultural_bias_ground_truth else None
        
        latency_metrics = {
            "avg_latency": self.metrics["avg_latency"],
            "p50_latency": self.metrics["p50_latency"],
            "p95_latency": self.metrics["p95_latency"],
            "p99_latency": self.metrics["p99_latency"],
            "subsecond_percentage": sum(1 for l in self.metrics["latencies"] if l < 1.0) / len(self.metrics["latencies"]) if self.metrics["latencies"] else 0
        }
        
        task_accuracies = {}
        for task_name, task_metrics in metrics.items():
            task_accuracies[task_name] = task_metrics.get("accuracy", 0.0)
        
        ta_professor_metrics = {
            "professor_usage_rate": self.metrics["professor_usage_rate"],
            "ta_only_usage": self.metrics["ta_only_usage"],
            "professor_usage": self.metrics["professor_usage"],
            "confidence_threshold": self.config.get("TA_CONFIDENCE_THRESHOLD", 0.7)
        }
        
        return {
            "task_metrics": metrics,
            "task_accuracies": task_accuracies,
            "overall_accuracy": overall_accuracy,
            "cultural_bias_index": cultural_bias_index,
            "latency_metrics": latency_metrics,
            "ta_professor_metrics": ta_professor_metrics,
            "total_samples": len(all_ground_truth),
            "model_type": "context_aware_ta_teacher_architecture"
        }

    def _create_confusion_matrix(self, y_true: List[str], y_pred: List[str], classes: List[str]) -> List[List[int]]:
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        try:
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            return cm.tolist()
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
            return [[0 for _ in classes] for _ in classes]

    async def cleanup(self):
        if self.processor and self.processor.context_retriever:
            await self.processor.context_retriever.close_session()
            logger.info("Context retriever session closed")
        
        if self.processor:
            if hasattr(self.processor, 'cleanup'):
                await self.processor.cleanup()

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "processing_metrics": self.metrics,
            "processor_metrics": self.processor.get_metrics() if self.processor else {},
            "model_type": "context_aware_ta_teacher_architecture",
            "agent_architecture": "kg_agent -> context_aware_ta_agent -> (conditional) teacher_agent -> explainer_agent",
            "ta_professor_stats": {
                "professor_usage_rate": self.metrics["professor_usage_rate"],
                "confidence_threshold": self.config.get("TA_CONFIDENCE_THRESHOLD", 0.7),
                "ta_only_tweets": self.metrics["ta_only_usage"],
                "teacher_refined_tweets": self.metrics["professor_usage"]
            }
        }


async def main_async():
    parser = argparse.ArgumentParser(description="IndicKARMA Verbal Aggression Detection with Context-Aware TA-Teacher Architecture")
    parser.add_argument("--data", type=str, help="Path to the CSV file with tweets")
    parser.add_argument("--output", type=str, default="./results/analysis_results.json", help="Path to save results")
    parser.add_argument("--model", type=str, help="Primary LLM model to use (for non-classification tasks)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tweets to analyze")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing")
    parser.add_argument("--fast", action="store_true", help="Use faster processing mode with optimizations")
    parser.add_argument("--tweet", type=str, help="Single tweet to analyze")
    parser.add_argument("--metrics", action="store_true", help="Output performance metrics")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate results against ground truth")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold for Teacher Agent (default: 0.7)")
    parser.add_argument("--ta-model-path", type=str, help="Path to Context-Aware TA model weights (overrides config)")
    parser.add_argument("--teacher-model-path", type=str, help="Path to Teacher model weights (overrides config)")
    args = parser.parse_args()
    
    framework = None
    try:
        config = INTEGRATED_CONFIG.copy()
        
        if args.model:
            config["LLM_CONFIG"]["primary_model"] = args.model
        
        if args.confidence_threshold:
            config["TA_CONFIDENCE_THRESHOLD"] = args.confidence_threshold
            config["AGENT_CONFIG"]["ta_confidence_threshold"] = args.confidence_threshold
            logger.info(f"Using confidence threshold: {args.confidence_threshold}")
        
        if args.ta_model_path:
            config["TA_MODEL_CONFIG"]["model_path"] = args.ta_model_path
            logger.info(f"Using Context-Aware TA model from: {args.ta_model_path}")
        
        if args.teacher_model_path:
            config["FINETUNED_MODEL_CONFIG"]["lora_weights"] = args.teacher_model_path
            logger.info(f"Using Teacher model from: {args.teacher_model_path}")
        
        if args.fast:
            logger.info("Using fast mode with optimizations")
            config["AGENT_CONFIG"]["timeout"] = 0.2
            config["CONTEXT_CONFIG"]["settings"]["context_timeout"] = 0.3
            config["PERFORMANCE_CONFIG"]["processing"]["timeout"] = 0.3
            
            config["CONTEXT_CONFIG"]["sources"]["wikidata"]["enabled"] = False
            config["KG_SOURCES"]["wikidata"]["enabled"] = False
        
        framework = AgenticFramework(config)
        
        if args.tweet:
            logger.info("Analyzing single tweet with Context-Aware TA-Teacher architecture")
            result = await framework.analyze_tweet(args.tweet)
            print(json.dumps(result, indent=2))
            
            if args.metrics:
                print("\nPerformance Metrics:")
                metrics = framework.get_performance_metrics()
                print(json.dumps(metrics, indent=2))
            
            return
        
        if args.data:
            logger.info(f"Processing CSV with Context-Aware TA-Teacher architecture: {args.data}")
            results = []
            
            async for batch_results in framework.process_csv(
                args.data, 
                args.output, 
                args.batch_size, 
                args.limit
            ):
                results = batch_results
                print(f"Processed {len(results)} tweets so far...")
            
            logger.info(f"Completed processing {len(results)} tweets with Context-Aware TA-Teacher architecture")
            
            if args.evaluate and results:
                logger.info("Evaluating results against ground truth")
                evaluation = framework.evaluate_results(results)
                
                print("\n" + "="*60)
                print("EVALUATION RESULTS (Context-Aware TA-Teacher Architecture)")
                print("="*60)
                print(f"Overall Accuracy: {evaluation['overall_accuracy']:.4f}")
                print(f"Cultural Index: {evaluation['cultural_bias_index']:.4f}")
                print(f"P99 Latency: {evaluation['latency_metrics']['p99_latency']:.3f}s")
                print(f"Subsecond Percentage: {evaluation['latency_metrics']['subsecond_percentage']:.2%}")
                
                print("\nContext-Aware TA-Teacher Performance:")
                ta_prof_metrics = evaluation['ta_professor_metrics']
                print(f"  Teacher Usage Rate: {ta_prof_metrics['professor_usage_rate']:.2%}")
                print(f"  Context-Aware TA-Only Tweets: {ta_prof_metrics['ta_only_usage']}")
                print(f"  Teacher-Refined Tweets: {ta_prof_metrics['professor_usage']}")
                print(f"  Confidence Threshold: {ta_prof_metrics['confidence_threshold']}")
                
                print("\nTask-wise Accuracies:")
                for task, accuracy in evaluation.get('task_accuracies', {}).items():
                    print(f"  {task}: {accuracy:.4f}")
                
                print(f"\nTotal Samples Evaluated: {evaluation['total_samples']}")
                print(f"Model Type: {evaluation['model_type']}")
                
                eval_output_path = args.output.replace(".json", "_evaluation.json")
                with open(eval_output_path, 'w') as f:
                    json.dump(evaluation, f, indent=2)
                logger.info(f"Evaluation results saved to {eval_output_path}")
            
            if args.metrics:
                print("\nPerformance Metrics:")
                metrics = framework.get_performance_metrics()
                print(json.dumps(metrics, indent=2))
            
            return
        
        if not args.data and not args.tweet:
            parser.print_help()
            print("\nExample usage:")
            print("  # Analyze single tweet:")
            print("  python main.py --tweet 'Your tweet text here'")
            print("  # Process CSV file with custom confidence threshold:")
            print("  python main.py --data data.csv --output results.json --evaluate --confidence-threshold 0.7")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if framework:
            try:
                await framework.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            
        cleanup_resources()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        gc.collect()
        logger.info("Application resources cleaned up")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_resources()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        gc.collect()


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()