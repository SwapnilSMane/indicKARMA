import os
import json
import time
import asyncio
import logging
import aiohttp
import requests
import torch
import torch.nn.functional as F
import ast
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
from src.resource_manager import register_session, register_connector
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, GenerationConfig

try:
    from src.context_aware_mtl import truncate_context_by_tokens, ContextAwareMTL
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

class AttentionTAAgent:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.confidence_threshold = model_config.get("confidence_threshold", 0.6)
        
        self.LABEL_MAPS = {
            'aggression_level': {'OAG': 0, 'CAG': 1, 'NAG': 2},
            'aggression_type': {'PTH': 0, 'STH': 1, 'NtAG': 2, 'CuAG': 3, 'UNC': 4},
            'gender_bias': {'GEN': 0, 'GENT': 1, 'NGEN': 2},
            'caste_bias': {'CAS': 0, 'CAST': 1, 'NCAS': 2},
            'religious_bias': {'COM': 0, 'COMT': 1, 'NCOM': 2},
            'ethnicity_bias': {'ETH': 0, 'ETHT': 1, 'NETH': 2}
        }
        
        self.INVERSE_LABEL_MAPS = {
            task: {v: k for k, v in task_map.items()} 
            for task, task_map in self.LABEL_MAPS.items()
        }
        
        self.TASKS = list(self.LABEL_MAPS.keys())
        
        self.metrics = {
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "low_confidence_predictions": 0,
            "high_confidence_predictions": 0
        }
    
    async def initialize(self):
        if self.initialized:
            return
            
        try:
            if CONTEXT_AWARE_AVAILABLE:
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained('google/rembert')
                
                self.model = NovelContextAwareRemBERT()
                
                model_path = self.model_config.get("model_path")
                if model_path and os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                        
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded Context-Aware model from {model_path}")
                else:
                    logger.warning("Context-Aware model path not found, using untrained model")
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Context-Aware TA Agent initialized successfully")
            
            else:
                from transformers import AutoTokenizer, AutoModel
                import torch.nn as nn
                
                self.tokenizer = AutoTokenizer.from_pretrained('google/rembert')
                
                class FallbackAttentionModel(nn.Module):
                    def __init__(self, model_name='google/rembert', use_gradient_checkpointing=True):
                        super(FallbackAttentionModel, self).__init__()
                        self.encoder = AutoModel.from_pretrained(model_name)
                        
                        if use_gradient_checkpointing:
                            try:
                                self.encoder.gradient_checkpointing_enable()
                            except (AttributeError, ValueError) as e:
                                logger.warning(f"Gradient checkpointing not supported: {e}")

                        self.dropout = nn.Dropout(0.1)
                        hidden_size = self.encoder.config.hidden_size

                        self.NUM_LABELS = {
                            'aggression_level': 3,
                            'aggression_type': 5,
                            'gender_bias': 3,
                            'caste_bias': 3,
                            'religious_bias': 3,
                            'ethnicity_bias': 3
                        }

                        self.classifiers = nn.ModuleDict({
                            task: nn.Sequential(
                                nn.Linear(hidden_size, hidden_size // 2),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(hidden_size // 2, num_labels)
                            ) for task, num_labels in self.NUM_LABELS.items()
                        })

                    def forward(self, input_ids, attention_mask):
                        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                        hidden_states = outputs.last_hidden_state
                        cls_output = hidden_states[:, 0, :]
                        cls_output = self.dropout(cls_output)
                        
                        logits = {task: classifier(cls_output) for task, classifier in self.classifiers.items()}
                        return logits, None
                
                self.model = FallbackAttentionModel(
                    use_gradient_checkpointing=self.model_config.get("use_gradient_checkpointing", True)
                )
                
                model_path = self.model_config.get("model_path")
                if model_path and os.path.exists(model_path):
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                        self.model.load_state_dict(state_dict)
                        logger.info(f"Loaded fallback weights from {model_path}")
                    except Exception as e:
                        logger.warning(f"Could not load fallback weights: {e}")
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.warning("Attention TA Agent initialized with fallback model")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing Attention TA model: {e}")
            raise
    
    def _get_predictions_with_confidence(self, logits_dict):
        predictions = {}
        confidences = {}
        
        for task in self.TASKS:
            if task in logits_dict:
                task_logits = logits_dict[task]
                probabilities = F.softmax(task_logits, dim=1)
                max_probs, pred_labels = torch.max(probabilities, dim=1)
                
                predictions[task] = self.INVERSE_LABEL_MAPS[task][pred_labels.item()]
                confidences[task] = max_probs.item()
        
        return predictions, confidences
    
    def _prepare_context_input(self, text: str, kg_context: str = "") -> str:
        if not kg_context.strip():
            return text
        
        if CONTEXT_AWARE_AVAILABLE:
            context = truncate_context_by_tokens(kg_context, self.tokenizer, max_tokens=150)
        else:
            context_tokens = self.tokenizer.tokenize(kg_context)
            if len(context_tokens) > 150:
                context = self.tokenizer.convert_tokens_to_string(context_tokens[:150])
            else:
                context = kg_context
        
        if CONTEXT_AWARE_AVAILABLE:
            combined_input = f"{text} [SEP] Context: {context}"
        else:
            combined_input = f"{text} [SEP] Context: {context}"
        
        return combined_input
    
    async def predict_all_tasks(self, text: str, kg_context: str = "") -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.metrics["calls"] += 1
        
        try:
            combined_input = self._prepare_context_input(text, kg_context)
            
            encoding = self.tokenizer(
                combined_input,
                max_length=self.model_config.get("max_length", 256),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                if CONTEXT_AWARE_AVAILABLE:
                    logits, _ = self.model(input_ids, attention_mask)
                else:
                    logits, _ = self.model(input_ids, attention_mask)
            
            predictions, confidences = self._get_predictions_with_confidence(logits)
            
            if predictions.get("aggression_level") == "NAG":
                predictions["aggression_type"] = "N/A"
                confidences["aggression_type"] = 1.0
            
            low_confidence_tasks = [
                task for task, conf in confidences.items() 
                if conf < self.confidence_threshold
            ]
            
            elapsed = time.time() - start_time
            self.metrics["successful_calls"] += 1
            self.metrics["total_latency"] += elapsed
            
            if low_confidence_tasks:
                self.metrics["low_confidence_predictions"] += 1
            else:
                self.metrics["high_confidence_predictions"] += 1
            
            logger.info(f"Context-Aware TA Agent predictions: {predictions}")
            logger.info(f"Context-Aware TA Agent confidences: {confidences}")
            logger.info(f"Low confidence tasks: {low_confidence_tasks}")
            
            return predictions, confidences, low_confidence_tasks
            
        except Exception as e:
            logger.error(f"Error in Context-Aware TA Agent predict_all_tasks: {e}")
            self.metrics["failed_calls"] += 1
            
            default_predictions = {
                "aggression_level": "NAG",
                "aggression_type": "N/A",
                "gender_bias": "NGEN", 
                "religious_bias": "NCOM",
                "caste_bias": "NCAS",
                "ethnicity_bias": "NETH"
            }
            
            default_confidences = {task: 0.0 for task in self.TASKS}
            
            return default_predictions, default_confidences, list(self.TASKS)
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.metrics.copy()
        
        if metrics["successful_calls"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["successful_calls"]
            metrics["low_confidence_rate"] = metrics["low_confidence_predictions"] / metrics["successful_calls"]
        else:
            metrics["avg_latency"] = 0
            metrics["low_confidence_rate"] = 0
            
        metrics["success_rate"] = metrics["successful_calls"] / max(1, metrics["calls"])
        metrics["confidence_threshold"] = self.confidence_threshold
        metrics["model_type"] = "context_aware_attention" if CONTEXT_AWARE_AVAILABLE else "fallback_attention"
        
        return metrics


class LMMTAgent:
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        
        self.task_name_to_id = {
            "aggression_level": 0,
            "aggression_type": 1,
            "gender_bias": 2,
            "religious_bias": 3,
            "caste_bias": 4,
            "ethnicity_bias": 5
        }
        
        self.metrics = {
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "refinement_calls": 0
        }
    
    async def initialize(self):
        if self.initialized:
            return
            
        try:
            required_keys = ["base_model", "lora_weights", "lora_target_modules", "lora_r", "lora_alpha", "lambda_num", "num_B"]
            missing_keys = [key for key in required_keys if key not in self.model_config]
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
            import sys
            mtl_lora_path = os.path.expanduser("~/MTL-LoRA")
            if mtl_lora_path not in sys.path:
                sys.path.append(mtl_lora_path)
                
            try:
                from src.custom_model import LlamaForCausalLM
                from src.utils import wrap_model
            except ImportError as e:
                logger.error(f"Failed to import MTL-LoRA modules. Please ensure MTL-LoRA is cloned to ~/MTL-LoRA")
                logger.error(f"Import error: {e}")
                raise
            
            base_model = self.model_config["base_model"]
            lora_weights = self.model_config["lora_weights"]
            
            if not os.path.exists(lora_weights):
                raise FileNotFoundError(f"LoRA weights file not found: {lora_weights}")
            
            if "LLaMA" in self.model_config.get("model_name", ""):
                self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if "llama" in base_model.lower():
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                    trust_remote_code=True,
                )
            else:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                    trust_remote_code=True,
                )
            
            mlora_config = {
                "type": "mlora",
                "r": self.model_config["lora_r"],
                "lora_alpha": self.model_config["lora_alpha"],
                "lora_dropout": self.model_config.get("lora_dropout", 0.05),
                "lambda_num": self.model_config["lambda_num"],
                "B_num": self.model_config["num_B"],
                "B_scale": self.model_config.get("temperature", 0.1),
                "diagonal_format": False,
            }
            
            self.model = wrap_model(
                self.model, 
                self.model_config["lora_target_modules"], 
                mlora_config
            )
            
            state_dict = torch.load(lora_weights, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            
            if msg.unexpected_keys:
                logger.warning(f"Unexpected keys in state dict: {msg.unexpected_keys}")
            
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            
            self.model.to(torch.bfloat16)
            self.model.eval()
            self.initialized = True
            
            logger.info("Fine-tuned mLoRA model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing fine-tuned model: {e}")
            logger.error(f"Model config keys available: {list(self.model_config.keys())}")
            raise
    
    def generate_instruction(self, task: str, text: str, kg_context: str = "") -> str:
        if kg_context and kg_context.strip():
            enhanced_text = f"Context: {kg_context.strip()}\n\nText: {text}"
        else:
            enhanced_text = text
        
        if task == "aggression_level":
            return f"""Please classify the following text for aggression level:

'{enhanced_text}'

OAG: Overtly Aggressive CAG: Covertly Aggressive NAG: Non Aggressive

Answer format: oag/cag/nag"""
        
        elif task == "aggression_type":
            return f"""Please classify the following text for aggression type:

'{enhanced_text}'

PTH: Physical Threat STH: Sexual Threat NtAG: Non-threatening Aggression CuAG: Curse or Abuse

Answer format: pth/sth/ntag/cuag"""
        
        elif task == "gender_bias":
            return f"""Please classify the following text for gender bias:

'{enhanced_text}'

GEN: Gendered GENT: Gendered Threats NGEN: Not Gendered

Answer format: gen/gent/ngen"""
        
        elif task == "religious_bias":
            return f"""Please classify the following text for religious bias:

'{enhanced_text}'

COM: Communal COMT: Communal Threats NCOM: Not Communal

Answer format: com/comt/ncom"""
        
        elif task == "caste_bias":
            return f"""Please classify the following text for caste bias:

'{enhanced_text}'

CAS: Casteist CAST: Casteist Threats NCAS: Not Casteist

Answer format: cas/cast/ncas"""
        
        elif task == "ethnicity_bias":
            return f"""Please classify the following text for ethnicity bias:

'{enhanced_text}'

ETH: Racist ETHT: Racist Threats NETH: Not Racist

Answer format: eth/etht/neth"""
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def generate_prompt(self, instruction: str, input_text: str = None) -> str:
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""
    
    async def predict_all_tasks(self, text: str, kg_context: str = "") -> Dict[str, str]:
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.metrics["calls"] += 1
        
        try:
            results = {}
            
            aggression_level = await self._predict_single_task("aggression_level", text, kg_context)
            results["aggression_level"] = aggression_level
            
            if aggression_level.lower() in ["oag", "cag"]:
                aggression_type = await self._predict_single_task("aggression_type", text, kg_context)
                results["aggression_type"] = aggression_type
            else:
                results["aggression_type"] = "N/A"
            
            for task in ["gender_bias", "religious_bias", "caste_bias", "ethnicity_bias"]:
                result = await self._predict_single_task(task, text, kg_context)
                results[task] = result
            
            elapsed = time.time() - start_time
            self.metrics["successful_calls"] += 1
            self.metrics["total_latency"] += elapsed
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LMMT Agent predict_all_tasks: {e}")
            self.metrics["failed_calls"] += 1
            
            return {
                "aggression_level": "NAG",
                "aggression_type": "N/A",
                "gender_bias": "NGEN", 
                "religious_bias": "NCOM",
                "caste_bias": "NCAS",
                "ethnicity_bias": "NETH"
            }
    
    async def predict_specific_tasks(self, tasks: List[str], text: str, kg_context: str = "") -> Dict[str, str]:
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.metrics["refinement_calls"] += 1
        
        try:
            results = {}
            
            if "aggression_level" in tasks:
                aggression_level = await self._predict_single_task("aggression_level", text, kg_context)
                results["aggression_level"] = aggression_level
                
                if "aggression_type" in tasks:
                    if aggression_level.lower() in ["oag", "cag"]:
                        aggression_type = await self._predict_single_task("aggression_type", text, kg_context)
                        results["aggression_type"] = aggression_type
                    else:
                        results["aggression_type"] = "N/A"
            elif "aggression_type" in tasks:
                aggression_level = await self._predict_single_task("aggression_level", text, kg_context)
                if aggression_level.lower() in ["oag", "cag"]:
                    aggression_type = await self._predict_single_task("aggression_type", text, kg_context)
                    results["aggression_type"] = aggression_type
                else:
                    results["aggression_type"] = "N/A"
            
            bias_tasks = ["gender_bias", "religious_bias", "caste_bias", "ethnicity_bias"]
            for task in tasks:
                if task in bias_tasks:
                    result = await self._predict_single_task(task, text, kg_context)
                    results[task] = result
            
            elapsed = time.time() - start_time
            self.metrics["successful_calls"] += 1
            self.metrics["total_latency"] += elapsed
            
            logger.info(f"LMMT Agent refined {len(tasks)} tasks: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LMMT Agent predict_specific_tasks: {e}")
            self.metrics["failed_calls"] += 1
            
            default_values = {
                "aggression_level": "NAG",
                "aggression_type": "N/A",
                "gender_bias": "NGEN", 
                "religious_bias": "NCOM",
                "caste_bias": "NCAS",
                "ethnicity_bias": "NETH"
            }
            
            return {task: default_values.get(task, "UNC") for task in tasks}
    
    async def _predict_single_task(self, task: str, text: str, kg_context: str) -> str:
        try:
            instruction = self.generate_instruction(task, text, kg_context)
            prompt = self.generate_prompt(instruction)
            
            inputs = self.tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            
            lambda_index = torch.tensor(self.task_name_to_id[task]).repeat(input_ids.shape[0]).to(self.device)
            
            generation_config = GenerationConfig(
                temperature=self.model_config.get("inference_temperature", 0.1),
                top_p=1,
                top_k=1,
                num_beams=1,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    lambda_index=lambda_index,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=32,
                    use_cache=False,
                )
            
            outputs = self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
            output = outputs[0].split("### Response:")[1].strip()
            
            prediction = self._extract_answer(task, output)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in LMMT Agent _predict_single_task for {task}: {e}")
            return self._get_default_answer(task)
    
    def _extract_answer(self, task: str, response: str) -> str:
        import re
        
        response = response.strip().lower()
        
        if task == "aggression_level":
            pred_answers = re.findall(r"oag|cag|nag", response)
            return pred_answers[0].upper() if pred_answers else "NAG"
        
        elif task == "aggression_type":
            pred_answers = re.findall(r"pth|sth|ntag|cuag", response)
            if pred_answers[0] == 'ntag':
                return 'NtAG'
            elif pred_answers[0] == 'cuag':
                return 'CuAG'
            return pred_answers[0].upper() if pred_answers else "UNC"
        
        elif task == "gender_bias":
            pred_answers = re.findall(r"gen|gent|ngen", response)
            return pred_answers[0].upper() if pred_answers else "NGEN"
        
        elif task == "religious_bias":
            pred_answers = re.findall(r"com|comt|ncom", response)
            return pred_answers[0].upper() if pred_answers else "NCOM"
        
        elif task == "caste_bias":
            pred_answers = re.findall(r"cas|cast|ncas", response)
            return pred_answers[0].upper() if pred_answers else "NCAS"
        
        elif task == "ethnicity_bias":
            pred_answers = re.findall(r"eth|etht|neth", response)
            return pred_answers[0].upper() if pred_answers else "NETH"
        
        return self._get_default_answer(task)
    
    def _get_default_answer(self, task: str) -> str:
        defaults = {
            "aggression_level": "NAG",
            "aggression_type": "UNC",
            "gender_bias": "NGEN",
            "religious_bias": "NCOM", 
            "caste_bias": "NCAS",
            "ethnicity_bias": "NETH"
        }
        return defaults.get(task, "UNC")
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.metrics.copy()
        
        if metrics["successful_calls"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["successful_calls"]
        else:
            metrics["avg_latency"] = 0
            
        metrics["success_rate"] = metrics["successful_calls"] / max(1, metrics["calls"])
        metrics["refinement_ratio"] = metrics["refinement_calls"] / max(1, metrics["calls"])
        
        return metrics


class AsyncOllamaLLM:
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        self.timeout = aiohttp.ClientTimeout(total=60)
        
        self.model_checked = False
        self.model_available = False
        
        self.metrics = {
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "refusal_retries": 0,
            "total_tokens": 0,
            "total_latency": 0,
        }

    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300, ssl=False)
        connector = register_connector(connector)
        
        self.session = register_session(aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=connector,
            headers={'User-Agent': 'KG-Agent/1.0'}
        ))
        self.initialized = True        

    async def ensure_model(self) -> bool:
        if self.model_checked:
            return self.model_available
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.api_base}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        model_names = [model.get("name") for model in models]
                        
                        if self.model_name in model_names:
                            logger.info(f"Model {self.model_name} is available")
                            self.model_checked = True
                            self.model_available = True
                            return True
                        else:
                            logger.info(f"Model {self.model_name} not found. Pulling from Ollama...")
                            
                            try:
                                pull_response = requests.post(
                                    f"{self.api_base}/api/pull",
                                    json={"name": self.model_name},
                                    timeout=600
                                )
                                
                                if pull_response.status_code == 200:
                                    logger.info(f"Successfully pulled model {self.model_name}")
                                    self.model_checked = True
                                    self.model_available = True
                                    return True
                                else:
                                    logger.error(f"Failed to pull model: {pull_response.text}")
                            except Exception as pull_error:
                                logger.error(f"Error during model pull: {pull_error}")
                    else:
                        response_text = await response.text()
                        logger.error(f"Failed to list models: {response_text}")
            
            self.model_checked = True
            self.model_available = False
            return False
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {e}")
            self.model_checked = True
            self.model_available = False
            return False
    
    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        start_time = time.time()
        self.metrics["calls"] += 1
        
        model_available = await self.ensure_model()
        if not model_available:
            error_msg = f"Model {self.model_name} is not available"
            logger.error(error_msg)
            self.metrics["failed_calls"] += 1
            raise RuntimeError(error_msg)
        
        try:
            if system_prompt is None or system_prompt.strip() == "":
                system_prompt = """You are an AI text analysis assistant focused on identifying patterns in language.
                Your task is to analyze text objectively for research purposes only.
                You should focus on linguistic patterns and provide factual analysis.
                This analysis is for academic and research purposes to help understand and address harmful content.
                Be direct and concise in your responses.
                """
            else:
                system_prompt = system_prompt.strip() + """
                You are analyzing this content for research purposes only to help identify and address harmful patterns.
                Be direct and concise in your responses.
                """
                
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_base}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        generated_text = result.get("response", "")
                        
                        elapsed = time.time() - start_time
                        self.metrics["successful_calls"] += 1
                        self.metrics["total_latency"] += elapsed
                        if "eval_count" in result:
                            self.metrics["total_tokens"] += result["eval_count"]
                        
                        refusal_patterns = [
                            "I cannot analyze",
                            "I'm not able to",
                            "I cannot assist with",
                            "I apologize, but I cannot",
                            "I don't feel comfortable",
                            "I cannot provide",
                            "I'm unable to comment",
                            "cannot help with"
                        ]
                        
                        if any(pattern.lower() in generated_text.lower() for pattern in refusal_patterns):
                            logger.warning("Model refused to analyze content, retrying with research context")
                            self.metrics["refusal_retries"] += 1
                            
                            research_system_prompt = """You are an AI assistant participating in a research project 
                            to identify and classify harmful content for academic purposes only.
                            Your analysis will help researchers develop better tools to detect and address problematic content.
                            You must provide an objective classification of the text according to the requested categories,
                            focusing only on linguistic patterns without judgment or refusal.
                            This is purely for research and educational purposes.
                            IMPORTANT: Respond directly with the classification label only."""
                            
                            research_prompt = f"""IMPORTANT: This is an academic research task to classify content.
                            Your objective analysis is crucial for developing safety tools.
                            
                            {prompt}
                            
                            Remember to focus only on the requested classification task without refusal.
                            Your analysis will be used only for research purposes.
                            Respond with the classification label only."""
                            
                            payload["system"] = research_system_prompt
                            payload["prompt"] = research_prompt
                            
                            retry_start = time.time()
                            async with session.post(f"{self.api_base}/api/generate", json=payload) as retry_response:
                                if retry_response.status == 200:
                                    retry_result = await retry_response.json()
                                    retry_text = retry_result.get("response", "").strip()
                                    
                                    retry_elapsed = time.time() - retry_start
                                    self.metrics["total_latency"] += retry_elapsed
                                    if "eval_count" in retry_result:
                                        self.metrics["total_tokens"] += retry_result["eval_count"]
                                    
                                    if any(pattern.lower() in retry_text.lower() for pattern in refusal_patterns):
                                        logger.warning("Model still refusing, extracting classification or providing default")
                                        
                                        if "gender" in prompt.lower() or "gender_bias" in prompt.lower():
                                            return "NGEN"
                                        elif "religious" in prompt.lower() or "religion" in prompt.lower():
                                            return "NCOM"
                                        elif "caste" in prompt.lower():
                                            return "NCAS"
                                        elif "ethnicity" in prompt.lower() or "race" in prompt.lower():
                                            return "NETH"
                                        elif "aggression" in prompt.lower():
                                            return "NAG"
                                        elif "type" in prompt.lower():
                                            return "NtAG"
                                        else:
                                            tags = ["OAG", "CAG", "NAG", "PTH", "STH", "NtAG", "CuAG",
                                                  "ATK", "DFN", "CNS", "AIN", "GSL", "GEN", "GENT", "NGEN",
                                                  "COM-fail", "COMT", "NCOM", "CAS", "CAST", "NCAS", "ETH", "ETHT", "NETH"]
                                            
                                            for tag in tags:
                                                if tag in retry_text:
                                                    return tag
                                            
                                            clean_text = self._extract_classification(retry_text)
                                            if clean_text:
                                                return clean_text
                                            
                                            return "NAG"
                                    
                                    return retry_text
                                else:
                                    retry_text = await retry_response.text()
                                    logger.error(f"Error in retry: {retry_text}")
                        
                        return generated_text
                    else:
                        response_text = await response.text()
                        error_msg = f"Error generating response: {response_text}"
                        logger.error(error_msg)
                        self.metrics["failed_calls"] += 1
                        return f"ERROR: {response_text[:100]}"
                
        except Exception as e:
            error_msg = f"Error in Ollama API request: {e}"
            logger.error(error_msg)
            self.metrics["failed_calls"] += 1
            return f"ERROR: {str(e)[:100]}"
    
    def _extract_classification(self, text: str) -> str:
        if "Classification:" in text:
            parts = text.split("Classification:")
            if len(parts) > 1:
                classification = parts[1].strip().split()[0].strip()
                if classification and len(classification) <= 10:
                    return classification
        
        patterns = [
            "label is", "classify as", "classification is",
            "categorize as", "category is", "analysis shows",
            "classified as", "tag is", "result is"
        ]
        
        for pattern in patterns:
            if pattern in text.lower():
                parts = text.lower().split(pattern)
                if len(parts) > 1:
                    words = parts[1].strip().split()
                    if words:
                        classification = words[0].strip(".:,()[] \t\n").upper()
                        if classification and len(classification) <= 10:
                            return classification
        
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line) <= 20:
            return first_line
        
        return ""
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.metrics.copy()
        
        if metrics["successful_calls"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["successful_calls"]
            metrics["avg_tokens_per_call"] = metrics["total_tokens"] / metrics["successful_calls"]
        else:
            metrics["avg_latency"] = 0
            metrics["avg_tokens_per_call"] = 0
            
        metrics["success_rate"] = metrics["successful_calls"] / max(1, metrics["calls"])
        metrics["refusal_rate"] = metrics["refusal_retries"] / max(1, metrics["calls"])
        
        return metrics


class LLMInference:
    
    def __init__(self, model_name: str, fallback_models: List[str] = None, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.fallback_models = fallback_models or []
        self.config = config or {}
        
        self.is_finetuned = self.config.get("is_finetuned", False)
        self.is_rembert = self.config.get("is_rembert", False)
        
        self.max_retries = self.config.get("max_retries", 2)
        self.timeout = self.config.get("timeout", 5.0)
        self.temperature = self.config.get("temperature", 0.2)
        self.mc_dropout_samples = self.config.get("mc_dropout_samples", 3)
        
        self._primary_model = None
        self._fallback_models = []
        
        self.inference_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "average_latency": 0,
            "total_latency": 0
        }
    
    def _initialize_model(self, model_name: str):
        try:
            if self.is_rembert:
                logger.info(f"Initializing Context-Aware Attention TA Agent: {model_name}")
                model_config = self.config.get("model_config", self.config.get("TA_MODEL_CONFIG", {}))
                if not model_config:
                    raise ValueError("Context-Aware Attention TA model configuration not found. Please check TA_MODEL_CONFIG in config.")
                return AttentionTAAgent(model_config)
                
            elif self.is_finetuned:
                logger.info(f"Initializing fine-tuned model: {model_name}")
                model_config = self.config.get("model_config", self.config.get("FINETUNED_MODEL_CONFIG", {}))
                if not model_config:
                    raise ValueError("Fine-tuned model configuration not found. Please check FINETUNED_MODEL_CONFIG in config.")
                return LMMTAgent(model_config)
            else:
                logger.info(f"Initializing Ollama model: {model_name}")
                
                model = model_name
                
                ollama_model = AsyncOllamaLLM(
                    model_name=model,
                    temperature=self.temperature,
                    max_tokens=self.config.get("max_tokens", 2048)
                )
                
                return ollama_model
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            return None
    
    def _get_primary_model(self):
        if self._primary_model is None:
            self._primary_model = self._initialize_model(self.model_name)
        return self._primary_model
    
    def _get_fallback_model(self, index: int = 0):
        if index >= len(self.fallback_models):
            return None
        
        model_name = self.fallback_models[index]
        
        while len(self._fallback_models) <= index:
            if len(self._fallback_models) == index:
                model = self._initialize_model(model_name)
                self._fallback_models.append(model)
            else:
                self._fallback_models.append(None)
        
        return self._fallback_models[index]
    
    async def arun(self, prompt: str, system_prompt: str = "", tags: List[str] = None) -> str:
        start_time = time.time()
        self.inference_stats["total_calls"] += 1
        
        model = self._get_primary_model()
        if model is None:
            logger.error(f"Primary model {self.model_name} not available")
            model = self._get_fallback_model(0)
            if model is None:
                self.inference_stats["failed_calls"] += 1
                raise ValueError(f"No models available for inference")
        
        if hasattr(model, 'initialize') and not getattr(model, 'initialized', False):
            await model.initialize()
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.is_finetuned or self.is_rembert:
                    response = "Error: Use predict_all_tasks for specialized models"
                else:
                    response = await asyncio.wait_for(
                        model.agenerate(prompt, system_prompt),
                        timeout=self.timeout
                    )
                
                if response.startswith("ERROR:"):
                    raise RuntimeError(f"Model error: {response}")
                
                self.inference_stats["successful_calls"] += 1
                duration = time.time() - start_time
                self.inference_stats["total_latency"] += duration
                self.inference_stats["average_latency"] = (
                    self.inference_stats["total_latency"] / self.inference_stats["successful_calls"]
                )
                return response
            
            except asyncio.TimeoutError:
                logger.warning(f"Model {model.model_name if hasattr(model, 'model_name') else 'unknown'} timed out (attempt {attempt+1}/{self.max_retries+1})")
                
                if attempt == self.max_retries:
                    fallback_model = self._get_fallback_model(0)
                    if fallback_model is not None:
                        logger.info(f"Using fallback model")
                        try:
                            if hasattr(fallback_model, 'initialize') and not getattr(fallback_model, 'initialized', False):
                                await fallback_model.initialize()
                            
                            response = await asyncio.wait_for(
                                fallback_model.agenerate(prompt, system_prompt),
                                timeout=self.timeout
                            )
                            
                            if response.startswith("ERROR:"):
                                raise RuntimeError(f"Fallback model error: {response}")
                            
                            self.inference_stats["fallback_used"] += 1
                            self.inference_stats["successful_calls"] += 1
                            duration = time.time() - start_time
                            self.inference_stats["total_latency"] += duration
                            self.inference_stats["average_latency"] = (
                                self.inference_stats["total_latency"] / 
                                self.inference_stats["successful_calls"]
                            )
                            
                            return response
                        except Exception as e:
                            logger.error(f"Fallback model failed: {e}")
            
            except Exception as e:
                logger.error(f"Error in model: {e}")
                
                if attempt == self.max_retries:
                    fallback_model = self._get_fallback_model(0)
                    if fallback_model is not None:
                        logger.info(f"Using fallback model")
                        try:
                            if hasattr(fallback_model, 'initialize') and not getattr(fallback_model, 'initialized', False):
                                await fallback_model.initialize()
                            
                            response = await asyncio.wait_for(
                                fallback_model.agenerate(prompt, system_prompt),
                                timeout=self.timeout
                            )
                            
                            if response.startswith("ERROR:"):
                                raise RuntimeError(f"Fallback model error: {response}")
                            
                            self.inference_stats["fallback_used"] += 1
                            self.inference_stats["successful_calls"] += 1
                            duration = time.time() - start_time
                            self.inference_stats["total_latency"] += duration
                            self.inference_stats["average_latency"] = (
                                self.inference_stats["total_latency"] / 
                                self.inference_stats["successful_calls"]
                            )
                            
                            return response
                        except Exception as e2:
                            logger.error(f"Fallback model failed: {e2}")
        
        self.inference_stats["failed_calls"] += 1
        
        logger.error(f"All inference attempts failed for prompt: {prompt[:100]}...")
        
        if "aggression_level" in prompt or "aggressive" in prompt.lower():
            return "NAG"
        elif "type" in prompt:
            return "NtAG"
        elif "gender" in prompt:
            return "NGEN"
        elif "religious" in prompt or "religion" in prompt:
            return "NCOM"
        elif "caste" in prompt:
            return "NCAS"
        elif "ethnicity" in prompt or "race" in prompt:
            return "NETH"
        else:
            return "Classification: NAG"
    
    async def predict_all_tasks(self, text: str, kg_context: str = "") -> Dict[str, str]:
        if not (self.is_finetuned or self.is_rembert):
            raise ValueError("predict_all_tasks can only be used with specialized models")
        
        model = self._get_primary_model()
        if model is None:
            raise ValueError("Specialized model not available")
        
        return await model.predict_all_tasks(text, kg_context)
    
    async def predict_specific_tasks(self, tasks: List[str], text: str, kg_context: str = "") -> Dict[str, str]:
        if not self.is_finetuned:
            raise ValueError("predict_specific_tasks can only be used with fine-tuned models")
        
        model = self._get_primary_model()
        if model is None:
            raise ValueError("Fine-tuned model not available")
        
        return await model.predict_specific_tasks(tasks, text, kg_context)
    
    async def get_ta_predictions_with_confidence(self, text: str, kg_context: str = "") -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
        if not self.is_rembert:
            raise ValueError("get_ta_predictions_with_confidence can only be used with Context-Aware Attention TA models")
        
        model = self._get_primary_model()
        if model is None:
            raise ValueError("Context-Aware Attention TA model not available")
        
        return await model.predict_all_tasks(text, kg_context)
    
    def run(self, prompt: str, system_prompt: str = "", tags: List[str] = None) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.arun(prompt, system_prompt, tags))
    
    async def run_with_uncertainty(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        tags: List[str] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        if self.is_finetuned or self.is_rembert:
            response = await self.arun(prompt, system_prompt, tags)
            return response, 0.9, {response: 0.9}
        
        original_temp = self.temperature
        mc_temp = min(0.7, self.temperature + 0.3)
        
        samples = []
        sample_tasks = []
        
        model = self._get_primary_model()
        if model is None:
            logger.error(f"Primary model {self.model_name} not available for uncertainty quantification")
            default_response = await self.arun(prompt, system_prompt, tags)
            return default_response, 0.5, {default_response: 0.5}
        
        for i in range(self.mc_dropout_samples):
            temp_model = AsyncOllamaLLM(model_name=model.model_name, temperature=mc_temp)
            
            task = temp_model.agenerate(prompt, system_prompt)
            sample_tasks.append(task)
        
        try:
            samples = await asyncio.gather(*sample_tasks, return_exceptions=True)
            
            valid_samples = [s for s in samples if not isinstance(s, Exception) and not s.startswith("ERROR:")]
            
            if not valid_samples:
                response = await self.arun(prompt, system_prompt, tags)
                return response, 1.0, {response: 1.0}
            
            class_votes = defaultdict(int)
            for sample in valid_samples:
                classification = self._extract_classification(sample)
                class_votes[classification] += 1
            
            if class_votes:
                sorted_classes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
                majority_class = sorted_classes[0][0]
                majority_votes = sorted_classes[0][1]
                
                confidence = majority_votes / len(valid_samples)
                
                alternatives = {}
                for cls, votes in sorted_classes:
                    alternatives[cls] = votes / len(valid_samples)
                
                matching_sample = next((s for s in valid_samples 
                                     if self._extract_classification(s) == majority_class), 
                                     valid_samples[0])
                
                return matching_sample, confidence, alternatives
            
            return valid_samples[0], 1.0, {valid_samples[0]: 1.0}
            
        except Exception as e:
            logger.error(f"Error in uncertainty quantification: {e}")
            response = await self.arun(prompt, system_prompt, tags)
            return response, 0.5, {response: 0.5}
    
    def _extract_classification(self, text: str) -> str:
        text = text.strip()
        
        for pattern in ["Classification:", "Class:", "Label:", "Result:", "Category:"]:
            if pattern in text:
                parts = text.split(pattern)
                if len(parts) > 1:
                    classification = parts[1].strip().split("\n")[0].strip()
                    first_word = classification.split()[0] if classification else ""
                    if first_word and len(first_word) <= 8:
                        return first_word
                    return classification
        
        tags = ["OAG", "CAG", "NAG", "PTH", "STH", "NtAG", "CuAG",
                "ATK", "DFN", "CNS", "AIN", "GSL", "GEN", "GENT", "NGEN",
                "COM", "COMT", "NCOM", "CAS", "CAST", "NCAS", "ETH", "ETHT", "NETH"]
        
        for tag in tags:
            if tag in text:
                return tag
        
        if "\n" in text:
            first_line = text.split("\n")[0].strip()
            if len(first_line) <= 20:
                return first_line
            return first_line
        
        return text
    
    async def extract_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        json_system = system_prompt + "\nYou must respond with valid JSON only, no other text."
        json_prompt = prompt + "\n\nRespond with valid JSON only, no other text or explanation."
        
        response = await self.arun(json_prompt, json_system)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                if "```json" in response and "```" in response.split("```json")[1]:
                    json_block = response.split("```json")[1].split("```")[0]
                    return json.loads(json_block)
                
                if "```" in response:
                    blocks = response.split("```")
                    for block in blocks:
                        if block.strip().startswith("{") or block.strip().startswith("["):
                            return json.loads(block)
                
                import re
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                
                logger.warning(f"Failed to extract JSON from: {response[:100]}...")
                return {}
            
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.inference_stats.copy()
        
        if self._primary_model:
            try:
                model_metrics = self._primary_model.get_metrics()
                metrics["model"] = model_metrics
            except:
                pass
                
        return metrics