import os
import sys
import json
import time
import logging
import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
from collections import defaultdict
import traceback

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Annotated

try:
    from src.context_aware_mtl import truncate_context_by_tokens, ContextAwareMTL
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

from models import LLMInference

from knowledge_base import ContextRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    tid: str = Field(default="")
    tweet: str = Field(default="")
    
    aggression_level: Optional[str] = None
    aggression_level_confidence: Optional[float] = None
    aggression_level_alternatives: Optional[Dict[str, float]] = None
    aggression_level_justification: Optional[str] = None
    aggression_level_description: Optional[str] = None
    
    aggression_type: Optional[str] = None
    aggression_type_confidence: Optional[float] = None
    aggression_type_alternatives: Optional[Dict[str, float]] = None
    aggression_type_justification: Optional[str] = None
    aggression_type_description: Optional[str] = None
        
    gender_bias: Optional[str] = None
    gender_bias_confidence: Optional[float] = None
    gender_bias_alternatives: Optional[Dict[str, float]] = None
    gender_bias_justification: Optional[str] = None
    gender_bias_description: Optional[str] = None
    
    religious_bias: Optional[str] = None
    religious_bias_confidence: Optional[float] = None
    religious_bias_alternatives: Optional[Dict[str, float]] = None
    religious_bias_justification: Optional[str] = None
    religious_bias_description: Optional[str] = None
    
    caste_bias: Optional[str] = None
    caste_bias_confidence: Optional[float] = None
    caste_bias_alternatives: Optional[Dict[str, float]] = None
    caste_bias_justification: Optional[str] = None
    caste_bias_description: Optional[str] = None
    
    ethnicity_bias: Optional[str] = None
    ethnicity_bias_confidence: Optional[float] = None
    ethnicity_bias_alternatives: Optional[Dict[str, float]] = None
    ethnicity_bias_justification: Optional[str] = None
    ethnicity_bias_description: Optional[str] = None
    
    ta_predictions: Optional[Dict[str, Any]] = None
    ta_confidences: Optional[Dict[str, float]] = None
    professor_used: Optional[bool] = None
    low_confidence_tasks: Optional[List[str]] = None
    professor_predictions: Optional[Dict[str, Any]] = None
    
    knowledge_context: Optional[Dict[str, Any]] = None
    KG_summary: Optional[str] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    knowledge_graph_path: Optional[str] = None
    entities: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    
    explanation: Optional[str] = None
    explanation_evidence: Optional[Dict[str, Any]] = None
    explanation_confidence: Optional[float] = None
    cultural_sensitivity_score: Optional[float] = None
    
    processing_times: Dict[str, float] = Field(default_factory=dict)
    inference_times: Dict[str, float] = Field(default_factory=dict)
    context_times: Dict[str, float] = Field(default_factory=dict)
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None
    
    errors: Dict[str, str] = Field(default_factory=dict)
    
    ground_truth_AggLevel: Optional[str] = None
    ground_truth_typeTag: Optional[str] = None
    ground_truth_GenderLabel: Optional[str] = None
    ground_truth_CasteLabel: Optional[str] = None
    ground_truth_ReligionTag: Optional[str] = None
    ground_truth_EthnicTag: Optional[str] = None
    
    all_task_predictions: Optional[Dict[str, Any]] = None
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def copy(self):
        return self.dict()
    
    def update(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        return self
    
    def items(self):
        return self.dict().items()
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    @field_validator('processing_times', 'inference_times', 'context_times', 'errors', mode='before')
    def set_default_dicts(cls, v):
        return v or {}
    
    @model_validator(mode='after')
    def set_processing_metadata(cls, values):
        if not values.get('processing_started'):
            values['processing_started'] = time.time()
        return values
    
    def record_time(self, agent_type: str, timing_type: str, duration: float) -> None:
        if timing_type == 'processing':
            self.processing_times[agent_type] = duration
        elif timing_type == 'inference':
            self.inference_times[agent_type] = duration
        elif timing_type == 'context':
            self.context_times[agent_type] = duration
    
    def record_error(self, agent_type: str, error_msg: str) -> None:
        self.errors[agent_type] = error_msg

    def get_processing_summary(self) -> Dict[str, Any]:
        now = time.time()
        total_time = now - (self.processing_started or now)
        
        return {
            "total_processing_time": total_time,
            "agent_processing_times": self.processing_times,
            "inference_times": self.inference_times,
            "context_times": self.context_times,
            "errors": self.errors,
            "total_agents": len(self.processing_times),
            "error_count": len(self.errors)
        }
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class BaseAgent:
    
    def __init__(self, 
                 agent_type: str,
                 config: Dict[str, Any],
                 llm_chain: Optional[LLMInference] = None,
                 context_retriever: Optional[ContextRetriever] = None):
        self.agent_type = agent_type
        self.config = config
        
        if llm_chain:
            self.llm_chain = llm_chain
        else:
            agent_specific_model = config.get("AGENT_MODELS", {}).get(agent_type, 
                                                                      config["LLM_CONFIG"]["primary_model"])
            
            fallback_models = config["LLM_CONFIG"].get("fallback_models", [])
            llm_config = config["LLM_CONFIG"].get("config", {})
            
            agent_config_overrides = config.get("AGENT_CONFIG_OVERRIDES", {}).get(agent_type, {})
            if agent_config_overrides:
                llm_config = {**llm_config, **agent_config_overrides}
                
            self.llm_chain = LLMInference(agent_specific_model, fallback_models, llm_config)
        
        self.prompt_template = config["PROMPT_TEMPLATES"].get(agent_type, "")
        if not self.prompt_template:
            logger.warning(f"No prompt template found for agent type: {agent_type}")
        
        self.context_retriever = context_retriever
        self.use_context = context_retriever is not None
        
        self.use_uncertainty = config.get("use_uncertainty", True)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.7)
        
        self.performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0,
            "avg_context_time": 0,
            "total_context_time": 0
        }
    
    async def _prepare_prompt(self, state: Union[Dict[str, Any], AgentState]) -> str:
        try:
            state_dict = state.dict() if hasattr(state, 'dict') else state
            basic_prompt = self.prompt_template.format(**state_dict)
            return basic_prompt
        
        except KeyError as e:
            logger.error(f"Missing key in state for prompt template: {e}")
            if hasattr(state, 'tweet'):
                tweet = state.tweet
            elif isinstance(state, dict):
                tweet = state.get("tweet", "")
            else:
                tweet = ""
            return f"Analyze the following tweet for {self.agent_type}: {tweet}"

    
    async def get_context(self, text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_context:
            return {"summary": ""}
        
        try:
            context = await self.context_retriever.get_context(text, self.agent_type, shared_state=state)
            return context
        except Exception as e:
            logger.warning(f"Error getting context for {self.agent_type}: {e}")
            return {"summary": ""}


class TAAgent(BaseAgent):
    
    def __init__(self, config: Dict[str, Any], 
                llm_chain: Optional[LLMInference] = None,
                context_retriever: Optional[ContextRetriever] = None):
        super().__init__("ta_agent", config, llm_chain, context_retriever)
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = config.get("TA_CONFIDENCE_THRESHOLD", 0.6)
        
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
    
    async def initialize_model(self):
        if self.model is not None:
            return
            
        try:
            if CONTEXT_AWARE_AVAILABLE:
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained('google/rembert')
                
                self.model = ContextAwareMTL()
                
                ta_model_path = self.config.get("TA_MODEL_CONFIG", {}).get("model_path")
                if ta_model_path and os.path.exists(ta_model_path):
                    checkpoint = torch.load(ta_model_path, map_location='cpu', weights_only=False)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                        
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded Context-Aware model from {ta_model_path}")
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
                    def __init__(self, model_name='google/rembert'):
                        super().__init__()
                        self.encoder = AutoModel.from_pretrained(model_name)
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
                
                self.model = FallbackAttentionModel()
                logger.warning("Using fallback attention model")
                
                ta_model_path = self.config.get("TA_MODEL_CONFIG", {}).get("model_path")
                if ta_model_path and os.path.exists(ta_model_path):
                    try:
                        state_dict = torch.load(ta_model_path, map_location='cpu')
                        self.model.load_state_dict(state_dict)
                        logger.info(f"Loaded fallback weights from {ta_model_path}")
                    except Exception as e:
                        logger.warning(f"Could not load fallback weights: {e}")
                
                self.model.to(self.device)
                self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing TA model: {e}")
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
    
    def _prepare_context_input(self, text: str, context: str, max_length: int = 256) -> str:
        if CONTEXT_AWARE_AVAILABLE:
            context = truncate_context_by_tokens(context, self.tokenizer, max_tokens=150)
        else:
            context_tokens = self.tokenizer.tokenize(context)
            if len(context_tokens) > 150:
                context = self.tokenizer.convert_tokens_to_string(context_tokens[:150])
        
        combined_input = f"{text} [SEP] Context: {context}"
        return combined_input
    
    async def process(self, state: Union[Dict[str, Any], AgentState]) -> Union[Dict[str, Any], AgentState]:
        start_time = time.time()
        self.performance_metrics["total_calls"] += 1
        
        is_pydantic = hasattr(state, 'dict')
        state_dict = state.dict() if is_pydantic else state.copy()
        
        try:
            await self.initialize_model()
            
            tweet = state_dict.get("tweet", "")
            kg_summary = state_dict.get("KG_summary", "")
            
            logger.info(f"TA Agent processing tweet: {tweet[:50]}...")
            
            combined_input = self._prepare_context_input(tweet, kg_summary)
            
            encoding = self.tokenizer(
                combined_input,
                max_length=self.config.get("TA_MODEL_CONFIG", {}).get("max_length", 256),
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
            
            for task in self.TASKS:
                if task in predictions:
                    state_dict[task] = predictions[task]
                    state_dict[f"{task}_confidence"] = confidences.get(task, 0.0)
                    state_dict[f"{task}_justification"] = f"TA Agent classified as {predictions[task]} (confidence: {confidences.get(task, 0.0):.3f})"
            
            state_dict["ta_predictions"] = predictions
            state_dict["ta_confidences"] = confidences
            state_dict["low_confidence_tasks"] = low_confidence_tasks
            state_dict["professor_used"] = len(low_confidence_tasks) > 0
            
            tag_descriptions = self.config.get("TAG_DESCRIPTIONS", {})
            for field in self.TASKS:
                if field in state_dict and state_dict[field]:
                    tag_value = state_dict[field]
                    if tag_value and tag_value not in ["N/A", "ERROR", "UNC"]:
                        state_dict[f"{field}_description"] = tag_descriptions.get(tag_value, tag_value)
            
            process_time = time.time() - start_time
            if "processing_times" not in state_dict:
                state_dict["processing_times"] = {}
            state_dict["processing_times"]["ta_agent"] = process_time
            
            self.performance_metrics["successful_calls"] += 1
            self.performance_metrics["total_processing_time"] += process_time
            self.performance_metrics["avg_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["successful_calls"]
            )
            
            logger.info(f"TA Agent processed tweet in {process_time:.3f}s")
            logger.info(f"Low confidence tasks: {low_confidence_tasks}")
            logger.info(f"Professor needed: {len(low_confidence_tasks) > 0}")
            
            return state_dict
                
        except Exception as e:
            logger.error(f"Error in TA agent: {e}")
            logger.error(traceback.format_exc())
            self.performance_metrics["failed_calls"] += 1
            
            state_dict["aggression_level"] = "NAG"
            state_dict["aggression_level_justification"] = f"TA Error: {str(e)[:100]}"
            state_dict["aggression_type"] = "N/A"
            state_dict["aggression_type_justification"] = "Not applicable due to error"
            
            for bias_type in ["gender_bias", "religious_bias", "caste_bias", "ethnicity_bias"]:
                if bias_type == "gender_bias":
                    default_value = "NGEN"
                elif bias_type == "religious_bias":
                    default_value = "NCOM"
                elif bias_type == "caste_bias":
                    default_value = "NCAS"
                elif bias_type == "ethnicity_bias":
                    default_value = "NETH"
                else:
                    default_value = "UNC"
                
                state_dict[bias_type] = default_value
                state_dict[f"{bias_type}_justification"] = f"TA Error: {str(e)[:100]}"
            
            state_dict["professor_used"] = True
            state_dict["low_confidence_tasks"] = list(self.TASKS)
            
            if "errors" not in state_dict:
                state_dict["errors"] = {}
            state_dict["errors"]["ta_agent"] = str(e)
            
            return state_dict


class TeacherAgent(BaseAgent):
    
    def __init__(self, config: Dict[str, Any], 
                llm_chain: Optional[LLMInference] = None,
                context_retriever: Optional[ContextRetriever] = None):
        super().__init__("professor_agent", config, llm_chain, context_retriever)
    
    async def process(self, state: Union[Dict[str, Any], AgentState]) -> Union[Dict[str, Any], AgentState]:
        start_time = time.time()
        self.performance_metrics["total_calls"] += 1
        
        is_pydantic = hasattr(state, 'dict')
        state_dict = state.dict() if is_pydantic else state.copy()
        
        try:
            low_confidence_tasks = state_dict.get("low_confidence_tasks", [])
            
            if not low_confidence_tasks:
                logger.info("No low confidence tasks, skipping Teacher Agent")
                return state_dict
            
            tweet = state_dict.get("tweet", "")
            kg_summary = state_dict.get("KG_summary", "")
            
            logger.info(f"Teacher Agent refining {len(low_confidence_tasks)} tasks: {low_confidence_tasks}")
            
            professor_predictions = await self.llm_chain.predict_all_tasks(tweet, kg_summary)
            
            for task in low_confidence_tasks:
                if task in professor_predictions:
                    state_dict[task] = professor_predictions[task]
                    state_dict[f"{task}_confidence"] = 0.9
                    state_dict[f"{task}_justification"] = f"Refined by Teacher Agent as {professor_predictions[task]} (TA confidence was low)"
                    
                    tag_descriptions = self.config.get("TAG_DESCRIPTIONS", {})
                    tag_value = professor_predictions[task]
                    if tag_value and tag_value not in ["N/A", "ERROR", "UNC"]:
                        state_dict[f"{task}_description"] = tag_descriptions.get(tag_value, tag_value)
            
            state_dict["professor_predictions"] = professor_predictions
            
            process_time = time.time() - start_time
            if "processing_times" not in state_dict:
                state_dict["processing_times"] = {}
            state_dict["processing_times"]["professor_agent"] = process_time
            
            self.performance_metrics["successful_calls"] += 1
            self.performance_metrics["total_processing_time"] += process_time
            self.performance_metrics["avg_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["successful_calls"]
            )
            
            logger.info(f"Teacher Agent refined tasks in {process_time:.3f}s")
            logger.info(f"Teacher predictions: {professor_predictions}")
            
            return state_dict
                
        except Exception as e:
            logger.error(f"Error in Teacher agent: {e}")
            logger.error(traceback.format_exc())
            self.performance_metrics["failed_calls"] += 1
            
            logger.warning("Teacher Agent failed, keeping TA Agent predictions")
            
            if "errors" not in state_dict:
                state_dict["errors"] = {}
            state_dict["errors"]["professor_agent"] = str(e)
            
            return state_dict


class KnowledgeRetrieverAgent(BaseAgent):
    
    def __init__(self, config: Dict[str, Any], 
                llm_chain: Optional[LLMInference] = None,
                context_retriever: Optional[ContextRetriever] = None):
        super().__init__("graph_rag", config, llm_chain, context_retriever)
        
        self.knowledge_graph = {
            "entities": {},
            "relationships": []
        }
        
        if not llm_chain and "kg_model" in config["LLM_CONFIG"]:
            kg_model = config["LLM_CONFIG"]["kg_model"]
            fallback_models = config["LLM_CONFIG"].get("fallback_models", [])
            llm_config = config["LLM_CONFIG"].get("config", {})
            
            kg_config_overrides = config.get("KG_CONFIG_OVERRIDES", {})
            if kg_config_overrides:
                llm_config = {**llm_config, **kg_config_overrides}
                
            self.llm_chain = LLMInference(kg_model, fallback_models, llm_config)
    
    async def process(self, state: Union[Dict[str, Any], AgentState]) -> Union[Dict[str, Any], AgentState]:
        start_time = time.time()
        self.performance_metrics["total_calls"] += 1
        
        is_pydantic = hasattr(state, 'dict')
        state_dict = state.dict() if is_pydantic else state.copy()
        
        try:
            kg_construction_prompt = self.prompt_template.format(**state_dict)
            kg_response = await self.llm_chain.arun(kg_construction_prompt)
            
            kg_data = self._extract_json_from_response(kg_response)
            kg_data = self._clean_kg_data(kg_data)
            
            enriched_kg, KG_summary = await self._enrich_knowledge_graph(kg_data)
            
            state_dict["knowledge_graph"] = enriched_kg
            state_dict["entities"] = kg_data.get("entities", [])
            
            if len(KG_summary):
                kg_context_lines = ["Facts about following tweet:", KG_summary]
                state_dict["KG_summary"] = "\n".join(kg_context_lines)
            
            if self.config.get("SAVE_KG_TO_FILE", False):
                tweet_id = state_dict.get("tid", "unknown")
                graph_path = self._save_knowledge_graph(enriched_kg, f"kg_{tweet_id}")
                state_dict["knowledge_graph_path"] = graph_path
            
            process_time = time.time() - start_time
            self.performance_metrics["successful_calls"] += 1
            self.performance_metrics["total_processing_time"] += process_time
            self.performance_metrics["avg_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["successful_calls"]
            )
            
            if "processing_times" not in state_dict:
                state_dict["processing_times"] = {}
            state_dict["processing_times"][self.agent_type] = process_time
            
            logger.info(f"Knowledge retrieval completed in {process_time:.3f} seconds")
            
            return state_dict
                
        except Exception as e:
            logger.error(f"Error in KnowledgeRetrieverAgent: {e}")
            logger.error(traceback.format_exc())
            self.performance_metrics["failed_calls"] += 1
            
            state_dict["knowledge_graph"] = {
                "entities": [],
                "relations": []
            }
            state_dict["entities"] = []
            state_dict["KG_summary"] = ""
            
            if "errors" not in state_dict:
                state_dict["errors"] = {}
            state_dict["errors"][self.agent_type] = str(e)
            
            return state_dict
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                import re
                json_matches = re.findall(r'\{.*\}', response, re.DOTALL)
                if json_matches:
                    return json.loads(json_matches[0])
            except:
                pass
            
            logger.warning(f"Could not extract JSON from KG response: {response[:100]}...")
            return {"entities": [], "relations": []}
    
    def _clean_kg_data(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {
            "entities": [],
            "relations": []
        }
        
        entities = kg_data.get("entities", [])
        if isinstance(entities, list):
            unique_entities = []
            for entity in entities:
                entity_str = str(entity).strip()
                if entity_str and entity_str not in unique_entities:
                    unique_entities.append(entity_str)
            cleaned["entities"] = unique_entities
        
        relations = kg_data.get("relations", [])
        if isinstance(relations, list):
            valid_relations = []
            for rel in relations:
                if isinstance(rel, dict) and "source" in rel and "relation" in rel and "target" in rel:
                    source = str(rel["source"]).strip()
                    relation = str(rel["relation"]).strip()
                    target = str(rel["target"]).strip()
                    
                    if source and relation and target:
                        if source not in cleaned["entities"]:
                            cleaned["entities"].append(source)
                        if target not in cleaned["entities"]:
                            cleaned["entities"].append(target)
                            
                        valid_relations.append({
                            "source": source,
                            "relation": relation,
                            "target": target,
                            "weight": rel.get("weight", 1.0)
                        })
            cleaned["relations"] = valid_relations    
        return cleaned
    
    async def _enrich_knowledge_graph(self, kg_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        kg = {
            "entities": {},
            "relations": []
        }
        
        for entity in kg_data.get("entities", []):
            kg["entities"][entity] = {
                "id": entity,
                "label": entity,
                "type": "entity"
            }

        for i, rel in enumerate(kg_data.get("relations", [])):
            kg["relations"].append({
                "id": f"r{i+1}",
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["relation"],
                "weight": rel.get("weight", 1.0)
            })
        
        KG_summary = ""
        
        if self.context_retriever:
            try:
                enriched, KG_summary = await self._enrich_with_context(kg_data)
                if enriched:
                    for i, rel in enumerate(enriched, start=len(kg["relations"])):
                        kg["relations"].append({
                            "id": f"r{i+1}",
                            "source": rel["source"],
                            "target": rel["target"],
                            "type": rel["relation"],
                            "weight": rel.get("weight", 0.8),
                            "source_name": rel.get("source_name", "context")
                        })

            except Exception as e:
                logger.warning(f"Error enriching KG with context: {e}")
            
        return kg, KG_summary
    
    async def _enrich_with_context(self, kg_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        if not self.context_retriever:
            return [], ""
            
        enriched_relations = []
        
        entities = kg_data.get("entities", [])[:3]
        KG_summary = []
        for entity in entities:
            try:
                context = await asyncio.wait_for(
                    self.context_retriever.get_context(entity, "graph_rag"),
                    timeout=30
                )
                
                summary = context.get("summary", '')
                if len(summary) > 5:
                    KG_summary.append(summary)

                items = context.get("items", [])
                for item in items[:2]:
                    source = item.get("source", "")
                    relation = item.get("relation", "")
                    target = item.get("target", "")
                    
                    if source and relation and target:
                        enriched_relations.append({
                            "source": source,
                            "relation": relation,
                            "target": target,
                            "weight": item.get("weight", 0.7),
                            "source_name": item.get("source_name", "context")
                        })
            
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Error enriching entity {entity}: {e}")
                continue
        KG_summary = " ".join(KG_summary)  
        return enriched_relations, KG_summary
    
    def _save_knowledge_graph(self, kg: Dict[str, Any], filename: str) -> str:
        output_dir = self.config.get("OUTPUT_PATHS", {}).get("knowledge_graphs", "results/knowledge_graphs")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"{filename}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(kg, f, indent=2)
            return filepath
        except Exception as e:
            logger.warning(f"Error saving knowledge graph: {e}")
            return "error_saving"


class ExplainerAgent(BaseAgent):
    
    def __init__(self, config: Dict[str, Any], 
                llm_chain: Optional[LLMInference] = None,
                context_retriever: Optional[ContextRetriever] = None):
        super().__init__("explainer", config, llm_chain, context_retriever)
    
    def _get_tag_description(self, tag: str) -> str:
        tag_descriptions = self.config.get("TAG_DESCRIPTIONS", {})
        return tag_descriptions.get(tag, tag)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        self.performance_metrics["total_calls"] += 1
        
        try:            
            is_pydantic = hasattr(state, 'dict')
            state_dict = state.dict() if is_pydantic else state.copy()
            
            descriptive_state = state_dict.copy()
            
            tag_fields = [
                "aggression_level", 
                "aggression_type", 
                "gender_bias", 
                "religious_bias", 
                "caste_bias", 
                "ethnicity_bias"
            ]
            
            for field in tag_fields:
                if field in descriptive_state:
                    tag = descriptive_state.get(field)
                    if tag and tag not in ["N/A", "ERROR", "UNC"]:
                        descriptive_state[f"{field}_description"] = self._get_tag_description(tag)
            
            professor_used = state_dict.get("professor_used", False)
            low_confidence_tasks = state_dict.get("low_confidence_tasks", [])
            
            prompt_template = self.prompt_template
            
            ta_professor_context = ""
            if professor_used:
                ta_professor_context = f"Note: Initial analysis was done by TA-Agent, but Teacher-Agent was used to refine predictions for tasks with low confidence: {', '.join(low_confidence_tasks)}."
            
            explanatory_prompt = prompt_template.format(
                KG_summary=descriptive_state.get("KG_summary", ""),
                tweet=descriptive_state.get("tweet", ""),
                aggression_level=descriptive_state.get("aggression_level_description", descriptive_state.get("aggression_level", "")),
                aggression_type=descriptive_state.get("aggression_type_description", descriptive_state.get("aggression_type", "")),
                gender_bias=descriptive_state.get("gender_bias_description", descriptive_state.get("gender_bias", "")),
                religious_bias=descriptive_state.get("religious_bias_description", descriptive_state.get("religious_bias", "")),
                caste_bias=descriptive_state.get("caste_bias_description", descriptive_state.get("caste_bias", "")),
                ethnicity_bias=descriptive_state.get("ethnicity_bias_description", descriptive_state.get("ethnicity_bias", ""))
            )
            
            if ta_professor_context:
                explanatory_prompt += f"\n\n{ta_professor_context}"
            
            explanation = await self.llm_chain.arun(explanatory_prompt)
            
            state_dict["explanation"] = explanation
            
            for field in tag_fields:
                if field in state_dict and field in descriptive_state:
                    description_key = f"{field}_description"
                    if description_key in descriptive_state:
                        state_dict[description_key] = descriptive_state[description_key]
            
            evidence = {
                "entities": state_dict.get("entities", []),
                "knowledge_graph": state_dict.get("knowledge_graph", {}),
                "ta_predictions": state_dict.get("ta_predictions", {}),
                "professor_predictions": state_dict.get("professor_predictions", {}),
                "professor_used": professor_used,
                "low_confidence_tasks": low_confidence_tasks
            }
            state_dict["explanation_evidence"] = evidence
            
            state_dict["processing_completed"] = time.time()
            
            process_time = time.time() - start_time
            self.performance_metrics["successful_calls"] += 1
            self.performance_metrics["total_processing_time"] += process_time
            self.performance_metrics["avg_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["successful_calls"]
            )
            
            if "processing_times" not in state_dict:
                state_dict["processing_times"] = {}
            state_dict["processing_times"][self.agent_type] = process_time
            
            logger.info(f"Explanation generated in {process_time:.3f} seconds")
            return state_dict
            
        except Exception as e:
            logger.error(f"Error in ExplainerAgent: {e}")
            logger.error(traceback.format_exc())
            self.performance_metrics["failed_calls"] += 1
            
            state["explanation"] = "Error generating explanation."
            
            if "errors" not in state:
                state["errors"] = {}
            state["errors"][self.agent_type] = str(e)
        
            return state


def should_use_professor(state: Dict[str, Any]) -> str:
    professor_used = state.get("professor_used", False)
    if professor_used:
        return "professor_agent"
    else:
        return "explainer_agent"


async def create_agent_workflow(config: Dict[str, Any]) -> StateGraph:
    context_config = config.get("CONTEXT_CONFIG", {})
    context_retriever = ContextRetriever(context_config)
    
    llm_chains = {}
    for agent_type in ["graph_rag", "ta_agent", "professor_agent", "explainer"]:
        agent_model = config.get("AGENT_MODELS", {}).get(agent_type, config["LLM_CONFIG"]["primary_model"])
        fallback_models = config["LLM_CONFIG"].get("fallback_models", [])
        llm_config = config["LLM_CONFIG"].get("config", {})
        
        agent_config_overrides = config.get("AGENT_CONFIG_OVERRIDES", {}).get(agent_type, {})
        if agent_config_overrides:
            llm_config = {**llm_config, **agent_config_overrides}
        
        if agent_type == "professor_agent" and llm_config.get("is_finetuned", False):
            finetuned_config = config.get("FINETUNED_MODEL_CONFIG", {})
            llm_config["model_config"] = finetuned_config
            logger.info(f"Added fine-tuned model config for {agent_type}")
            
        llm_chains[agent_type] = LLMInference(agent_model, fallback_models, llm_config)
    
    graph_rag_agent = KnowledgeRetrieverAgent(config, llm_chains.get("graph_rag"), context_retriever)
    ta_agent = TAAgent(config, llm_chains.get("ta_agent"), context_retriever)
    professor_agent = TeacherAgent(config, llm_chains.get("professor_agent"), context_retriever)
    explainer_agent = ExplainerAgent(config, llm_chains.get("explainer"), context_retriever)
    
    async def graph_rag_fn(state: AgentState) -> Dict:
        logger.info("Processing knowledge graph")
        result = await graph_rag_agent.process(state)
        return result
    
    async def ta_agent_fn(state: AgentState) -> Dict:
        logger.info("Processing with TA-Agent")
        result = await ta_agent.process(state)
        return result
    
    async def professor_agent_fn(state: AgentState) -> Dict:
        logger.info("Processing with Teacher-Agent")
        result = await professor_agent.process(state)
        return result
    
    async def explainer_fn(state: AgentState) -> Dict:
        logger.info("Generating explanation")
        result = await explainer_agent.process(state)
        return result
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("kg_agent", graph_rag_fn)
    workflow.add_node("ta_agent", ta_agent_fn)
    workflow.add_node("professor_agent", professor_agent_fn)
    workflow.add_node("explainer_agent", explainer_fn)
    
    workflow.set_entry_point("kg_agent")
    workflow.add_edge("kg_agent", "ta_agent")
    
    workflow.add_conditional_edges(
        "ta_agent",
        should_use_professor,
        {
            "professor_agent": "professor_agent",
            "explainer_agent": "explainer_agent"
        }
    )
    
    workflow.add_edge("professor_agent", "explainer_agent")
    workflow.add_edge("explainer_agent", END)
    
    logger.info("Compiling workflow graph with TA-Teacher architecture")
    return workflow.compile()


class IndicKARMAProcessor:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow = None
        self.context_retriever = None
        
        self.batch_size = config.get("batch_size", 10)
        self.max_concurrency = config.get("max_concurrency", 4)
        
        self.processing_metrics = {
            "total_tweets": 0,
            "successful_tweets": 0,
            "failed_tweets": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0,
            "batches_processed": 0,
            "professor_usage": 0,
            "ta_only_usage": 0
        }
    
    async def initialize(self):
        if not self.workflow:
            context_config = self.config.get("CONTEXT_CONFIG", {})
            self.context_retriever = ContextRetriever(context_config)
            
            self.workflow = await create_agent_workflow(self.config)
            logger.info("IndicKARMA workflow initialized with TA-Teacher architecture")
    
    async def process_tweet(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        
        initial_state = {
            "tid": str(tweet_data.get("tid", f"tweet_{int(time.time())}")),
            "tweet": tweet_data.get("tweet", ""),
            "processing_started": time.time()
        }
        
        for label_key in ["AggLevel", "typeTag", "GenderLabel", "CasteLabel", 
                        "ReligionTag", "EthnicTag"]:
            if label_key in tweet_data:
                initial_state[f"ground_truth_{label_key}"] = tweet_data[label_key]
        
        try:
            logger.info(f"Analyzing tweet ID: {initial_state['tid']} with TA-Teacher architecture")
            memory = MemorySaver()
            final_state = await self.workflow.ainvoke(initial_state, config={"checkpointer": memory})
            
            end_time = time.time()
            processing_time = end_time - initial_state["processing_started"]
            final_state["total_processing_time"] = processing_time
            final_state["processing_ended"] = end_time
            
            if final_state.get("professor_used", False):
                self.processing_metrics["professor_usage"] += 1
            else:
                self.processing_metrics["ta_only_usage"] += 1
        
            return final_state
        except Exception as e:
            logger.error(f"Error processing tweet {initial_state['tid']}: {e}")
            return {
                "tid": initial_state["tid"], 
                "tweet": initial_state["tweet"], 
                "error": str(e),
                "processing_started": initial_state["processing_started"],
                "processing_ended": time.time()
            }
    
    async def process_batch(self, tweet_batch: List[Dict[str, Any]], 
                          max_workers: int = None) -> List[Dict[str, Any]]:
        batch_start = time.time()
        workers = max_workers or self.max_concurrency
        
        tasks = [self.process_tweet(tweet) for tweet in tweet_batch]
        
        sem = asyncio.Semaphore(workers)
        
        async def process_with_semaphore(task):
            async with sem:
                return await task
        
        bounded_tasks = [process_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                processed_results.append({
                    "tid": tweet_batch[i].get("tid", f"unknown_{i}"),
                    "tweet": tweet_batch[i].get("tweet", ""),
                    "error": str(result)
                })
                self.processing_metrics["failed_tweets"] += 1
            else:
                processed_results.append(result)
                self.processing_metrics["successful_tweets"] += 1
        
        batch_time = time.time() - batch_start
        self.processing_metrics["total_tweets"] += len(tweet_batch)
        self.processing_metrics["total_processing_time"] += batch_time
        self.processing_metrics["batches_processed"] += 1
        self.processing_metrics["avg_processing_time"] = (
            self.processing_metrics["total_processing_time"] / 
            self.processing_metrics["batches_processed"]
        )
        
        logger.info(f"Batch of {len(tweet_batch)} tweets processed in {batch_time:.3f}s")
        
        return processed_results
    
    async def process_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_results = []
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(dataset)-1)//self.batch_size + 1}")
            
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            
            if (i//self.batch_size + 1) % 5 == 0:
                yield all_results
        
        yield all_results
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.processing_metrics.copy()
        
        total_processed = metrics["professor_usage"] + metrics["ta_only_usage"]
        if total_processed > 0:
            metrics["professor_usage_rate"] = metrics["professor_usage"] / total_processed
            metrics["ta_only_rate"] = metrics["ta_only_usage"] / total_processed
        else:
            metrics["professor_usage_rate"] = 0.0
            metrics["ta_only_rate"] = 0.0
            
        return metrics