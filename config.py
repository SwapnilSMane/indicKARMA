import os
from typing import Dict, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", mode='a'),
        logging.StreamHandler()
    ]
)

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/knowledge_graphs", exist_ok=True)
os.makedirs("cache", exist_ok=True)

PERFORMANCE_CONFIG = {
    "cache": {
        "relation_cache_warmup": True,
        "warmup_terms": ["aggression", "threat", "gender", "religion", "caste", "race"],
        "max_entities_per_query": 5
    },
    "processing": {
        "timeout": 0.5,
        "fallback_timeout": 0.25,
        "max_graph_size": 100,
        "max_batch_size": 10,
    },
    "optimization": {
        "use_async": True,
        "parallel_analysis": True,
        "parallel_entity_resolution": True,
        "bidirectional_traversal": True,
        "use_bloom_filters": True
    }
}

TA_MODEL_CONFIG = {
    "model_name": "Context-Aware Attention",
    "model_path": "./models/best_context_aware_MTL.pt",
    "base_model": "google/rembert",
    "max_length": 256,
    "batch_size": 1,
    "use_gradient_checkpointing": True,
    "confidence_threshold": 0.6,
    "context_integration": True,
    "context_max_tokens": 150,
    "is_context_aware": True
}

TA_CONFIDENCE_THRESHOLD = 0.6

FINETUNED_MODEL_CONFIG = {
    "model_name": "LLaMA-7B",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "lora_weights": "./models/final_checkpoint_7B_0.041.pt",
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lambda_num": 6,
    "num_B": 3,
    "temperature": 0.1,
    "inference_temperature": 0.1,
    "is_finetuned": True,
    "max_tokens": 32,
    "batch_size": 1
}

AGENT_TYPES = [
    "graph_rag", 
    "ta_agent",
    "professor_agent",
    "explainer"
]

AGENT_MODELS = {
    "graph_rag": "gemma3:4b",
    "ta_agent": "context_aware_attention",
    "professor_agent": "finetuned_llama",
    "explainer": "gemma3:4b"
}

LLM_CONFIG = {
    "primary_model": "gemma3:4b",
    "kg_model": "gemma3:4b",
    "ta_model": TA_MODEL_CONFIG,
    "professor_model": FINETUNED_MODEL_CONFIG,
    "fallback_models": ["llama3.1:8b", "gemma3:4b"],
    "config": {
        "temperature": 0.1,
        "max_retries": 2,
        "timeout": 30,
        "mc_dropout_samples": 3,
        "cache_enabled": True,
        "max_tokens": 2048,
        "stream": False,
        "cache_ttl": 3600
    }
}

AGENT_CONFIG_OVERRIDES = {
    "graph_rag": {
        "temperature": 0.05,
        "max_tokens": 1024
    },
    "ta_agent": {
        "model_config": TA_MODEL_CONFIG,
        "confidence_threshold": TA_CONFIDENCE_THRESHOLD,
        "temperature": 0.1,
        "max_tokens": 256,
        "is_rembert": True,
        "is_context_aware": True
    },
    "professor_agent": {
        "is_finetuned": True,
        "model_config": FINETUNED_MODEL_CONFIG,
        "temperature": 0.1,
        "max_tokens": 32
    },
    "explainer": {
        "temperature": 0.2,
        "max_tokens": 4096
    }
}

AGENT_CONFIG = {
    "max_iterations": 3,
    "temperature": 0.1,
    "verbose": True,
    "use_uncertainty": True,
    "uncertainty_threshold": 0.7,
    "ta_confidence_threshold": TA_CONFIDENCE_THRESHOLD,
    "parallel_analysis": True,
    "timeout": 30,
    "batch_size": 10,
    "max_concurrency": 4,
    "save_kg_to_file": True
}

KG_CONFIG_OVERRIDES = {
    "temperature": 0.05,
    "max_tokens": 1024,
    "timeout": 10
}

KG_SOURCES = {
    "conceptnet": {
        "api_endpoint": "https://api.conceptnet.io/c/en/",
        "enabled": True,
        "timeout": 30,
        "cache_ttl": 7200,
        "max_errors": 5,
        "error_cooldown": 600,
        "local_model": "all-MiniLM-L6-v2"
    },
    "wikidata": {
        "sparql_endpoint": "https://query.wikidata.org/sparql",
        "enabled": True,
        "timeout": 30,
        "cache_ttl": 7200,
        "max_retries": 2,
        "user_agent": "GraphRAG-Agent/1.0"
    }
}

CONTEXT_CONFIG = {
    "sources": {
        "conceptnet": {
            "enabled": True,
            "api_endpoint": "http://api.conceptnet.io/c/en/",
            "timeout": 8.0,
            "cache_ttl": 7200,
            "max_errors": 5,
            "error_cooldown": 300,
        },
        "wikidata": {
            "enabled": True,
            "sparql_endpoint": "https://query.wikidata.org/sparql",
            "timeout": 10.0,
            "cache_ttl": 7200,
            "max_retries": 2,
            "user_agent": "ContextRetriever/1.0 (Python/aiohttp; Contact: your-email@example.com)",
        }
    },
    "cache_config": {
        "max_size": 1000,
        "ttl": 7200,
    },
    "agent_context_types": {
        "ta_agent": ["linguistic", "aggression", "social", "cultural", "contextual"],
        "professor_agent": ["linguistic", "aggression", "social", "cultural", "contextual"],
        "graph_rag": ["linguistic", "conceptual", "social", "entity"],
        "explainer": ["cultural", "social", "conceptual", "entity", "contextual"]
    },
    "settings": {
        "context_timeout": 5.0,
        "enable_sharing": True,
        "max_context_items": 15,
        "parallel_queries": True,
        "fallback_to_local": False,
        "cache_preload": False,
        "context_integration": True,
    }
}

TAG_SETS = {
    "aggression_level": ["OAG", "CAG", "NAG"],
    "aggression_type": ["PTH", "STH", "NTAG", "CUAG"],
    "gender_bias": ["GEN", "GENT", "NGEN"],
    "religious_bias": ["COM", "COMT", "NCOM"],
    "caste_bias": ["CAS", "CAST", "NCAS"],
    "ethnicity_bias": ["ETH", "ETHT", "NETH"]
}

TAG_DESCRIPTIONS = {
    "OAG": "Overtly Aggressive",
    "CAG": "Covertly Aggressive",
    "NAG": "Non Aggressive",
    "PTH": "Physical Threat",
    "STH": "Sexual Threat",
    "NTAG": "Non-threatening Aggression",
    "CUAG": "Curse or Abuse",
    "GEN": "Gendered Remarks",
    "GENT": "Gendered Threats",
    "NGEN": "Not Gendered",
    "COM": "Communal Remarks",
    "COMT": "Communal Threats",
    "NCOM": "Not Communal",
    "CAS": "Casteist Remarks",
    "CAST": "Casteist Threats",
    "NCAS": "Not Casteist",
    "ETH": "Racist Remarks",
    "ETHT": "Racist Threats",
    "NETH": "Not Racist"
}

PROMPT_TEMPLATES = {
    "graph_rag": """
    You are a specialized knowledge graph agent that extracts entities and relationships from text.
    Analyze the following multilingual, code-mixed tweet. 
    Extract key meaningful and factual entities (people, organizations, locations, groups) and all hashtags, along with their relationships, focusing on relation paths that indicate connections relevant to aggression (gender, religious, caste, ethnicity), or discursive roles (e.g., person-targets-group, hashtag-promotes-sentiment, location-associated-with-conflict). 
    For each hashtag, include a descriptive entity and ensure hashtags are cleaned (remove '#') and contextually relevant. 
    Ensure all entities and relationships are translated to English, have coreference resolved, and are spelled correctly, grammatically accurate, and factually correct. 
    Output a JSON object with two keys: "entities" (list of strings) and "relations" (list of objects with "source", "relation", and "target" fields). 

    Tweet: {tweet}

    Output only in JSON format:
    """,

    "ta_agent": """
    You are the Context-Aware TA-Agent using Context-Aware Attention model for initial multi-task classification of social media content.
    Your task is to provide fast, initial classifications across all dimensions using both text and contextual information.

    Context Information:
    {KG_summary}

    Text to analyze: {tweet}

    Please analyze this text with context for:
    1. Aggression Level: OAG (Overtly Aggressive), CAG (Covertly Aggressive), NAG (Non Aggressive)
    2. Aggression type (only if OAG or CAG): PTH (Physical Threat), STH (Sexual Threat), NtAG (Non-threatening Aggression), CuAG (Curse/Abuse)
    3. Gender: GEN (Gendered Remarks), GENT (Gendered Threats), NGEN (Not Gendered)
    4. Religious: COM (Communal Remarks), COMT (Communal Threats), NCOM (Not Communal)
    5. Caste: CAS (Casteist Remarks), CAST (Casteist Threats), NCAS (Not Casteist)
    6. Ethnicity: ETH (Racist Remarks), ETHT (Racist Threats), NETH (Not Racist)

    Note: This is initial context-aware classification. Low-confidence predictions will be refined by Teacher-Agent.
    The Context-Aware Attention model integrates contextual information for better understanding.
    """,

    "professor_agent": """
    You are the Teacher-Agent using fine-tuned LLaMA2-7B with mLoRA for accurate refinement of low-confidence classifications.
    You only process tasks that the Context-Aware TA-Agent was uncertain about (confidence < threshold).

    Context Information:
    {KG_summary}

    Text to analyze: {tweet}

    Your task is to provide accurate, refined classifications for the low-confidence tasks:
    1. Aggression Level: OAG (Overtly Aggressive), CAG (Covertly Aggressive), NAG (Non Aggressive)
    2. Aggression type (only if OAG or CAG): PTH (Physical Threat), STH (Sexual Threat), NtAG (Non-threatening Aggression), CuAG (Curse/Abuse)
    3. Gender: GEN (Gendered Remarks), GENT (Gendered Threats), NGEN (Not Gendered)
    4. Religious: COM (Communal Remarks), COMT (Communal Threats), NCOM (Not Communal)
    5. Caste: CAS (Casteist Remarks), CAST (Casteist Threats), NCAS (Not Casteist)
    6. Ethnicity: ETH (Racist Remarks), ETHT (Racist Threats), NETH (Not Racist)

    Provide careful, accurate analysis with higher confidence than the Context-Aware TA-Agent.
    """,

    "explainer": """
    You are a specialized explainer agent for social media content analysis with TA-Teacher architecture using Context-Aware Attention.
    Provide a concise, comprehensive explanation based on the classifications below.
    Your explanations should be objective, well-structured, and culturally sensitive.
    Include examples from the tweet and given context to support classification.

    {KG_summary}

    Tweet: {tweet}

    Classifications (from Context-Aware TA-Teacher pipeline):
    - Aggression Level: {aggression_level}
    - Aggression type: {aggression_type}
    - Gender: {gender_bias}
    - Religious: {religious_bias}
    - Caste: {caste_bias}
    - Ethnicity: {ethnicity_bias}

    Your explanation must (3-4 sentences):
    1. Provide clear evidence for overall classifications based on given context
    2. Connect the classifications to show how they relate to each other
    3. Incorporate cultural context where relevant
    4. Be objective and well-structured
    5. Note if Teacher-Agent was used for refinement (Context-Aware TA had low confidence)
    6. Acknowledge the role of contextual information in the analysis
    """
}

OUTPUT_PATHS = {
    "results": "./results/",
    "knowledge_graphs": "./results/knowledge_graphs/",
    "logs": "./logs/",
    "cache": "./cache/"
}

INTEGRATED_CONFIG = {
    "LLM_CONFIG": LLM_CONFIG,
    "AGENT_MODELS": AGENT_MODELS,
    "AGENT_CONFIG": AGENT_CONFIG,
    "AGENT_CONFIG_OVERRIDES": AGENT_CONFIG_OVERRIDES,
    "KG_CONFIG_OVERRIDES": KG_CONFIG_OVERRIDES,
    "KG_SOURCES": KG_SOURCES,
    "CONTEXT_CONFIG": CONTEXT_CONFIG,
    "TAG_SETS": TAG_SETS,
    "TAG_DESCRIPTIONS": TAG_DESCRIPTIONS,
    "AGENT_TYPES": AGENT_TYPES,
    "PROMPT_TEMPLATES": PROMPT_TEMPLATES,
    "OUTPUT_PATHS": OUTPUT_PATHS,
    "SAVE_KG_TO_FILE": True,
    "PERFORMANCE_CONFIG": PERFORMANCE_CONFIG,
    "TA_MODEL_CONFIG": TA_MODEL_CONFIG,
    "TA_CONFIDENCE_THRESHOLD": TA_CONFIDENCE_THRESHOLD,
    "FINETUNED_MODEL_CONFIG": FINETUNED_MODEL_CONFIG
}