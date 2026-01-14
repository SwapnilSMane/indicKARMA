# IndicKARMA: Knowledge-enhanced Agentic Reasoner for Multidimensional Aggression

by Swapnil Mane, Rajesh Sharma, and Suman Kundu


## Core Components

### 1. IndicKARMA.py
Main agent orchestration module implementing the Agentic workflow:

- `AgentState`: Pydantic model for state management
- `TAAgent`: Context-Aware Attention model for initial classification
- `TeacherAgent`: Fine-tuned LLaMA2-7B for refinement
- `KnowledgeRetrieverAgent`: Knowledge graph construction and retrieval
- `ExplainerAgent`: Comprehensive explanation generation
- `IndicKARMAProcessor`: Batch processing and workflow coordination

### 2. models.py
Model implementations and inference engines:

- `AttentionTAAgent`: Context-Aware Attention model wrapper
- `LMMTAgent`: Fine-tuned LLaMA2-7B with multi-task LoRA
- `AsyncOllamaLLM`: Asynchronous Ollama model interface
- `LLMInference`: Unified inference engine with fallback support

### 3. knowledge_base.py
Context retrieval and knowledge graph construction:

- `ContextRetriever`: Multi-source context aggregation
- ConceptNet and Wikidata integration
- Caching and performance optimization
- Agent-specific context filtering

### 4. config.py
System configuration and model parameters:

- Model paths and hyperparameters
- Agent-specific configurations
- Performance optimization settings
- Prompt templates for each agent

### 5. main.py
Application entry point and CLI interface:

- `AgenticFramework`: Main application class
- CSV batch processing
- Performance evaluation
- Metrics collection and reporting

## Usage

### Single Tweet Analysis
```bash
python main.py --tweet "Your tweet text here"
```

### Batch Processing
```bash
python main.py --data tweets.csv --output results.json --evaluate
```

### Custom Configuration
```bash
python main.py --data tweets.csv \
    --confidence-threshold 0.6 \
    --ta-model-path /path/to/ta/model \
    --teacher-model-path /path/to/teacher/model \
    --evaluate
```


## Key Features

### Evaluation Metrics
- Task-wise accuracy and F1 scores
- Overall accuracy 
- Latency and throughput metrics
- TA-Teacher usage statistics


### TA-Agent Model
- Context-Aware Attention model weights
- Path: `TA_MODEL_CONFIG["model_path"]`
- Based on google/rembert with context integration

### Teacher-Agent Model
- Fine-tuned LLaMA2-7B with MTL-LoRA
- Path: `FINETUNED_MODEL_CONFIG["lora_weights"]`
- Multi-task LoRA adaptation

### Ollama Models
- gemma3:4b (default)
- llama3.1:8b (fallback)
- Automatically downloaded if not available

## Performance Characteristics

### Latency
- P99 latency: <2s per tweet
- Subsecond processing for 80%+ of tweets
- Adaptive timeout handling


### Confidence Threshold
- Default: 0.6
- Range: 0.1-0.9
- Higher values = more Teacher usage

### Context Sources
- ConceptNet: Semantic relationships
- Wikidata: Entity information
- Configurable timeouts and caching


## Installation

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure model paths in config.py
4. Run: `python main.py --help`

## Publication Notes

This implementation represents the production-ready version of the IndicKARMA framework. We will make it publicly available upon acceptance. The code has been optimized for:

- Clean, maintainable architecture
- Production deployment readiness
- Comprehensive evaluation capabilities
- Academic reproducibility

The framework demonstrates state-of-the-art performance on multilingual multidimensional aggression detection in the Indian context with practical deployment considerations.
