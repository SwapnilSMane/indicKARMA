import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import gc
import argparse
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import warnings
import re
from collections import defaultdict
warnings.filterwarnings("ignore")

TASKS = ['aggression_level', 'aggression_type', 'gender_bias', 'caste_bias', 'religious_bias', 'ethnicity_bias']

LABEL_MAPS = {
    'aggression_level': {'OAG': 0, 'CAG': 1, 'NAG': 2},
    'aggression_type': {'PTH': 0, 'STH': 1, 'NtAG': 2, 'CuAG': 3, 'UNC': 4},
    'gender_bias': {'GEN': 0, 'GENT': 1, 'NGEN': 2},
    'caste_bias': {'CAS': 0, 'CAST': 1, 'NCAS': 2},
    'religious_bias': {'COM': 0, 'COMT': 1, 'NCOM': 2},
    'ethnicity_bias': {'ETH': 0, 'ETHT': 1, 'NETH': 2}
}

INVERSE_LABEL_MAPS = {
    task: {v: k for k, v in task_map.items()} 
    for task, task_map in LABEL_MAPS.items()
}

NUM_LABELS = {task: len(labels) for task, labels in LABEL_MAPS.items()}

LANGUAGE_MAP = {'en': 0, 'hi': 1, 'bn': 2, 'mr': 3}
LANGUAGE_NAMES = {0: 'English', 1: 'Hindi', 2: 'Bengali', 3: 'Marathi'}

def count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_context_by_tokens(context, tokenizer, max_tokens=150):
    if not context:
        return ""
    
    current_tokens = count_tokens(context, tokenizer)
    if current_tokens <= max_tokens:
        return context
    
    sentences = context.split('. ')
    while len(sentences) > 1 and count_tokens('. '.join(sentences), tokenizer) > max_tokens:
        sentences.pop()
    
    truncated = '. '.join(sentences)
    if not truncated.endswith('.') and sentences:
        truncated += '.'
    
    return truncated

def load_context_data(context_paths):
    context_data = {}
    
    train_context = {}
    
    if context_paths.get('train') and os.path.exists(context_paths['train']):
        try:
            with open(context_paths['train'], 'r', encoding='utf-8') as f:
                train_context.update(json.load(f))
                print(f"Loaded {len(train_context)} context entries from train")
        except Exception as e:
            print(f"Warning: Could not load train context: {e}")
    
    if context_paths.get('val') and os.path.exists(context_paths['val']):
        try:
            with open(context_paths['val'], 'r', encoding='utf-8') as f:
                val_context = json.load(f)
                train_context.update(val_context)
                print(f"Loaded {len(val_context)} context entries from val (merged with train)")
        except Exception as e:
            print(f"Warning: Could not load val context: {e}")
    
    context_data['train'] = train_context
    print(f"Total merged train context entries: {len(train_context)}")
    
    if context_paths.get('test') and os.path.exists(context_paths['test']):
        try:
            with open(context_paths['test'], 'r', encoding='utf-8') as f:
                test_context = json.load(f)
                context_data['test'] = test_context
                print(f"Loaded {len(test_context)} context entries for test")
        except Exception as e:
            print(f"Warning: Could not load test context: {e}")
            context_data['test'] = {}
    else:
        context_data['test'] = {}
    
    return context_data

class ContextAwareDataset(Dataset):
    def __init__(self, dataframe, tokenizer, context_data=None, max_length=256, split='train'):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.context_data = context_data
        self.max_length = max_length
        self.split = split
        
        self.processed_data = []
        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx]
            
            text = str(row['text']) if pd.notna(row['text']) else ""
            text = text.strip()
            
            tid = row['tid']
            lang = str(row.get('lang', 'en')).lower()
            
            context = ""
            if self.context_data:
                context_key = f"{tid}_{lang}"
                split_context = self.context_data.get(self.split, {})
                if context_key in split_context:
                    context = str(split_context[context_key])
                elif str(tid) in split_context:
                    context = str(split_context[str(tid)])
            
            context = truncate_context_by_tokens(context, self.tokenizer, max_tokens=150)
            
            self.processed_data.append({
                'text': text,
                'context': context,
                'tid': tid,
                'lang': lang,
                'row': row
            })
        
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        try:
            data_item = self.processed_data[index]
            text = data_item['text']
            context = data_item['context']
            tid = data_item['tid']
            row = data_item['row']
            lang = data_item['lang']
            
            if context.strip():
                combined_input = f"{text} [SEP] Context: {context}"
            else:
                combined_input = text
            
            if not combined_input.strip():
                combined_input = "[EMPTY]"
            
            try:
                encoding = self.tokenizer(
                    combined_input,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"Tokenization error for tid {tid}: {e}")
                encoding = self.tokenizer(
                    text if text.strip() else "[EMPTY]",
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            labels = {}
            for task in TASKS:
                if task in row and pd.notna(row[task]) and row[task] in LABEL_MAPS[task]:
                    labels[task] = torch.tensor(LABEL_MAPS[task][row[task]], dtype=torch.long)
                else:
                    labels[task] = torch.tensor(-100, dtype=torch.long)
            
            return {
                'tid': tid,
                'text': text,
                'context': context,
                'lang': lang,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing item at index {index}: {e}")
            return {
                'tid': 'error',
                'text': '[ERROR]',
                'context': '',
                'lang': 'error',
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': {task: torch.tensor(-100, dtype=torch.long) for task in TASKS}
            }

class ContextFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, text_hidden, context_mask=None):
        batch_size, seq_len, _ = text_hidden.shape
        
        Q = self.query(text_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(text_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(text_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            mask_value = torch.finfo(scores.dtype).min if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_output = torch.matmul(attention_weights, V)
        context_output = context_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.layer_norm(text_hidden + context_output)
        
        return output

class TaskSpecificContextualization(nn.Module):
    def __init__(self, hidden_size, num_tasks):
        super().__init__()
        self.task_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)
        
    def forward(self, hidden_states, task_id):
        batch_size = hidden_states.shape[0]
        task_embed = self.task_embeddings(torch.tensor(task_id, device=hidden_states.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        attended_output, _ = self.task_attention(task_embed, hidden_states, hidden_states)
        return attended_output.squeeze(1)

class ContextAwareMTL(nn.Module):
    def __init__(self, model_name="google/rembert", use_gradient_checkpointing=True):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        if use_gradient_checkpointing:
            try:
                self.encoder.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            except Exception as e:
                print(f"Warning: Gradient checkpointing not supported: {e}")
        
        self.context_attention = ContextFusion(hidden_size, num_heads=12)
        
        self.task_attention = TaskSpecificContextualization(hidden_size, len(TASKS))
        
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        self.classifiers = nn.ModuleDict()
        
        for i, (task, num_labels) in enumerate(NUM_LABELS.items()):
            if task in ['aggression_level', 'aggression_type']:
                self.classifiers[task] = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, num_labels)
                )
            else:
                self.classifiers[task] = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, num_labels)
                )
        
        self.task_weights = nn.Parameter(torch.ones(len(TASKS)))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        context_aware_hidden = self.context_attention(hidden_states, attention_mask)
        
        cls_output = context_aware_hidden[:, 0, :]

        shared_repr = self.shared_layer(cls_output)
        shared_repr = self.dropout(shared_repr)
        
        logits = {}
        for i, task in enumerate(TASKS):
            task_repr = self.task_attention(context_aware_hidden, i)
            
            combined_repr = shared_repr + task_repr
            
            logits[task] = self.classifiers[task](combined_repr)
        
        return logits, self.task_weights

def compute_weighted_loss(outputs, labels, task_weights, class_weights=None):
    total_loss = 0
    task_losses = {}
    valid_task_count = 0
    
    normalized_weights = F.softmax(task_weights, dim=0)
    
    for i, task in enumerate(TASKS):
        task_labels = labels[task]
        task_logits = outputs[task]
        
        valid_indices = task_labels != -100
        if valid_indices.sum() > 0:
            valid_task_count += 1
            
            task_logits_clamped = torch.clamp(task_logits, min=-50.0, max=50.0)
            
            if class_weights and task in class_weights:
                weights = class_weights[task].to(task_labels.device)
                task_loss = F.cross_entropy(
                    task_logits_clamped[valid_indices], 
                    task_labels[valid_indices],
                    weight=weights
                )
            else:
                task_loss = F.cross_entropy(
                    task_logits_clamped[valid_indices], 
                    task_labels[valid_indices]
                )
            
            weighted_loss = normalized_weights[i] * task_loss
            total_loss += weighted_loss
            task_losses[task] = task_loss.item()
    
    if valid_task_count > 0:
        total_loss = total_loss / valid_task_count
    
    return total_loss, task_losses

def evaluate_with_language_breakdown(model, dataloader, device, class_weights=None):
    model.eval()
    all_predictions = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    all_tids = []
    all_texts = []
    all_langs = []
    
    lang_predictions = {lang: {task: [] for task in TASKS} for lang in LANGUAGE_MAP.keys()}
    lang_labels = {lang: {task: [] for task in TASKS} for lang in LANGUAGE_MAP.keys()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast():
                logits, _ = model(input_ids, attention_mask)
            
            batch_langs = batch['lang']
            for task in TASKS:
                task_labels = batch['labels'][task].cpu().numpy()
                task_preds = torch.argmax(logits[task], dim=1).cpu().numpy()
                
                all_predictions[task].extend(task_preds.tolist())
                all_labels[task].extend(task_labels.tolist())
                
                for i, lang in enumerate(batch_langs):
                    if lang in lang_predictions:
                        lang_predictions[lang][task].append(task_preds[i])
                        lang_labels[lang][task].append(task_labels[i])
            
            batch_tids = batch['tid']
            if torch.is_tensor(batch_tids):
                batch_tids = batch_tids.tolist()
            all_tids.extend(batch_tids)
            all_langs.extend(batch_langs)
            
            if 'text' in batch:
                all_texts.extend(batch['text'])
    
    overall_metrics = {}
    for task in TASKS:
        valid_indices = np.array(all_labels[task]) != -100
        if np.any(valid_indices):
            valid_labels = np.array(all_labels[task])[valid_indices]
            valid_preds = np.array(all_predictions[task])[valid_indices]
            
            overall_metrics[task] = {
                'accuracy': float(accuracy_score(valid_labels, valid_preds)),
                'precision_macro': float(precision_score(valid_labels, valid_preds, average='macro', zero_division=0)),
                'precision_weighted': float(precision_score(valid_labels, valid_preds, average='weighted', zero_division=0)),
                'recall_macro': float(recall_score(valid_labels, valid_preds, average='macro', zero_division=0)),
                'recall_weighted': float(recall_score(valid_labels, valid_preds, average='weighted', zero_division=0)),
                'f1_macro': float(f1_score(valid_labels, valid_preds, average='macro', zero_division=0)),
                'f1_weighted': float(f1_score(valid_labels, valid_preds, average='weighted', zero_division=0))
            }
    
    language_metrics = {}
    for lang in LANGUAGE_MAP.keys():
        language_metrics[lang] = {}
        for task in TASKS:
            if lang_predictions[lang][task]:
                lang_preds = np.array(lang_predictions[lang][task])
                lang_true = np.array(lang_labels[lang][task])
                valid_indices = lang_true != -100
                
                if np.any(valid_indices):
                    valid_labels = lang_true[valid_indices]
                    valid_preds = lang_preds[valid_indices]
                    
                    if len(valid_labels) > 0:
                        language_metrics[lang][task] = {
                            'accuracy': float(accuracy_score(valid_labels, valid_preds)),
                            'f1_macro': float(f1_score(valid_labels, valid_preds, average='macro', zero_division=0)),
                            'f1_weighted': float(f1_score(valid_labels, valid_preds, average='weighted', zero_division=0)),
                            'support': int(len(valid_labels))
                        }
    
    tasks = ['gender_bias', 'caste_bias', 'religious_bias', 'ethnicity_bias']
    aggression_tasks = ['aggression_level', 'aggression_type']
    
    avg_metrics = {}
    aggression_avg_metrics = {}
    
    for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
        scores = [overall_metrics[task][metric] for task in tasks if task in overall_metrics]
        aggression_scores = [overall_metrics[task][metric] for task in aggression_tasks if task in overall_metrics]
        
        if scores:
            avg_metrics[metric] = float(np.mean(scores))
        if aggression_scores:
            aggression_avg_metrics[metric] = float(np.mean(aggression_scores))
    
    lang_avg = {}
    lang_aggression_avg = {}
    
    for lang in LANGUAGE_MAP.keys():
        if lang in language_metrics:
            lang_avg[lang] = {}
            lang_aggression_avg[lang] = {}
            
            for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
                scores = [language_metrics[lang][task][metric] for task in tasks if task in language_metrics[lang]]
                aggression_scores = [language_metrics[lang][task][metric] for task in aggression_tasks if task in language_metrics[lang]]
                
                if scores:
                    lang_avg[lang][metric] = float(np.mean(scores))
                if aggression_scores:
                    lang_aggression_avg[lang][metric] = float(np.mean(aggression_scores))
    
    predictions_with_metadata = []
    for i in range(len(all_tids)):
        pred_record = {
            'tid': all_tids[i],
            'lang': all_langs[i] if i < len(all_langs) else 'unknown'
        }
        
        if all_texts and i < len(all_texts):
            pred_record['text'] = all_texts[i]
        
        for task in TASKS:
            if i < len(all_predictions[task]):
                label_id = int(all_predictions[task][i])
                pred_record[task] = INVERSE_LABEL_MAPS[task][label_id]
                
                if i < len(all_labels[task]) and all_labels[task][i] != -100:
                    pred_record[f"{task}_true"] = INVERSE_LABEL_MAPS[task][int(all_labels[task][i])]
        
        predictions_with_metadata.append(pred_record)
    
    return {
        'overall_metrics': overall_metrics,
        'language_metrics': language_metrics,
        'avg_metrics': avg_metrics,
        'aggression_avg_metrics': aggression_avg_metrics,
        'lang_avg': lang_avg,
        'lang_aggression_avg': lang_aggression_avg,
        'predictions': predictions_with_metadata
    }

def train_context_aware_mtl(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, class_weights=None):
    best_f1_macro = 0.0
    best_model_path = "best_context_aware_mtl.pt"
    
    scaler = GradScaler()
    
    training_history = {
        'train_loss': [],
        'val_metrics': [],
        'task_weights': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_task_losses = defaultdict(list)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with autocast():
                    logits, task_weights = model(input_ids, attention_mask)
                    
                    labels = {task: batch['labels'][task].to(device) for task in TASKS}
                    
                    loss, task_losses = compute_weighted_loss(logits, labels, task_weights, class_weights)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at step {step}, skipping batch")
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                if grad_norm > 10.0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f} at step {step}")
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                for task, task_loss in task_losses.items():
                    epoch_task_losses[task].append(task_loss)
                
                if (step + 1) % 100 == 0:
                    current_task_weights = F.softmax(task_weights, dim=0)
                    print(f"Step {step+1} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.3f}")
                    print(f"Task weights: {dict(zip(TASKS, current_task_weights.detach().cpu().numpy().round(3)))}")
                
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"Error at step {step}: {e}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        print(f"\nEvaluating after epoch {epoch+1}...")
        try:
            eval_results = evaluate_with_language_breakdown(model, val_dataloader, device, class_weights)
            
            task_f1_scores = [metrics['f1_macro'] for metrics in eval_results['overall_metrics'].values()]
            avg_f1_macro = np.mean(task_f1_scores)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Training Loss: {total_loss / len(train_dataloader):.4f}")
            print(f"Average Validation F1 Macro: {avg_f1_macro:.4f}")
            
            print("\nTask-wise Validation Metrics:")
            for task, metrics in eval_results['overall_metrics'].items():
                print(f"{task}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
            
            print(f"\nTasks Average F1: {eval_results['avg_metrics'].get('f1_macro', 0):.4f}")
            print(f"Aggression Tasks Average F1: {eval_results['aggression_avg_metrics'].get('f1_macro', 0):.4f}")
            
            if avg_f1_macro > best_f1_macro:
                best_f1_macro = avg_f1_macro
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1_macro': best_f1_macro,
                    'task_weights': model.task_weights.detach().cpu(),
                    'eval_results': eval_results
                }, best_model_path)
                print(f" New best model saved! F1 Macro: {best_f1_macro:.4f}")
            
            training_history['train_loss'].append(total_loss / len(train_dataloader))
            training_history['val_metrics'].append(eval_results['overall_metrics'])
            training_history['task_weights'].append(F.softmax(model.task_weights, dim=0).detach().cpu().numpy())
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Continuing with next epoch...")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_f1_macro': 0.0,
            'task_weights': model.task_weights.detach().cpu(),
            'eval_results': {'overall_metrics': {}}
        }
    
    return model, training_history, checkpoint

def save_comprehensive_results(results, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, f"{model_name}_comprehensive_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': results['overall_metrics'],
            'language_metrics': results['language_metrics'],
            'avg_metrics': results['avg_metrics'],
            'aggression_avg_metrics': results['aggression_avg_metrics'],
            'lang_avg': results['lang_avg'],
            'lang_aggression_avg': results['lang_aggression_avg']
        }, f, indent=4)
    
    predictions_path = os.path.join(output_dir, f"{model_name}_predictions.json")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(results['predictions'], f, indent=4)
    
    summary_path = os.path.join(output_dir, f"{model_name}_summary_report.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Context-Aware MTL Model Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        for task, metrics in results['overall_metrics'].items():
            f.write(f"{task}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Macro: {metrics['f1_macro']:.4f}\n")
            f.write(f"  F1 Weighted: {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("CATEGORY AVERAGES:\n")
        f.write("-" * 16 + "\n")
        f.write("Detection:\n")
        for metric, value in results['avg_metrics'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nAggression Detection:\n")
        for metric, value in results['aggression_avg_metrics'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nLANGUAGE-WISE PERFORMANCE:\n")
        f.write("-" * 25 + "\n")
        for lang, lang_name in LANGUAGE_NAMES.items():
            lang_key = list(LANGUAGE_MAP.keys())[lang]
            if lang_key in results['language_metrics']:
                f.write(f"{lang_name} ({lang_key.upper()}):\n")
                
                if lang_key in results['lang_avg']:
                    f.write("  Detection:\n")
                    for metric, value in results['lang_avg'][lang_key].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                
                if lang_key in results['lang_aggression_avg']:
                    f.write("  Aggression Detection:\n")
                    for metric, value in results['lang_aggression_avg'][lang_key].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                
                f.write("\n")
    
    print(f"Comprehensive results saved:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Context-Aware MTL for Multilingual Aggression Detection")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output_dir", type=str, default="./mtl_results", help="Output directory")
    parser.add_argument("--model_dir", type=str, default="./mtl_models", help="Model save directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--context_train", type=str, help="Path to train context JSON")
    parser.add_argument("--context_val", type=str, help="Path to validation context JSON")
    parser.add_argument("--context_test", type=str, help="Path to test context JSON")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("Using CPU - training will be slower")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("Loading context data...")
    context_paths = {
        'train': args.context_train,
        'val': args.context_val, 
        'test': args.context_test
    }
    
    try:
        context_data = load_context_data(context_paths)
    except Exception as e:
        print(f"Warning: Error loading context data: {e}")
        print("Continuing without context data...")
        context_data = {'train': {}, 'test': {}}
    
    print("Loading datasets...")
    try:
        train_df = pd.read_csv(args.train_file)
        val_df = pd.read_csv(args.val_file)
        test_df = pd.read_csv(args.test_file)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        required_cols = ['tid', 'text', 'lang'] + TASKS
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {df_name}: {missing_cols}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    print("Initializing tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/rembert")
        print(" Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    print("Preparing datasets...")
    try:
        train_dataset = ContextAwareDataset(train_df, tokenizer, context_data, args.max_length, 'train')
        val_dataset = ContextAwareDataset(val_df, tokenizer, context_data, args.max_length, 'train')
        test_dataset = ContextAwareDataset(test_df, tokenizer, context_data, args.max_length, 'test')
        print(" Datasets prepared successfully")
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        return
    
    print("Creating data loaders...")
    try:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        print(" Data loaders created successfully")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    print("Initializing Context-Aware MTL model...")
    try:
        model = ContextAwareMTL(use_gradient_checkpointing=args.use_gradient_checkpointing).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" Model initialized successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    print("Setting up training...")
    try:
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        total_steps = len(train_dataloader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        print(f" Training setup complete. Total steps: {total_steps}")
        
    except Exception as e:
        print(f"Error setting up training: {e}")
        return
    
    print("Starting training...")
    try:
        model, training_history, checkpoint = train_context_aware_mtl(
            model, train_dataloader, val_dataloader, optimizer, scheduler, device, args.num_epochs
        )
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Attempting to save current model state...")
        try:
            emergency_path = os.path.join(args.output_dir, "emergency_model.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), emergency_path)
            print(f"Emergency model saved to: {emergency_path}")
        except:
            pass
        return
    
    print("Saving trained model...")
    try:
        os.makedirs(args.model_dir, exist_ok=True)
        final_model_path = os.path.join(args.model_dir, "context_aware_mtl_final.pt")
        torch.save(checkpoint, final_model_path)
        print(f"Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\nEvaluating on test set...")
    try:
        test_results = evaluate_with_language_breakdown(model, test_dataloader, device)
        
        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        
        print(f"\nOverall Performance:")
        overall_f1_scores = []
        for task, metrics in test_results['overall_metrics'].items():
            f1_score = metrics['f1_macro']
            overall_f1_scores.append(f1_score)
            print(f"{task}: F1={f1_score:.4f}, Acc={metrics['accuracy']:.4f}")
        
        print(f"\nAverage F1 Macro across all tasks: {np.mean(overall_f1_scores):.4f}")
        print(f"Detection Average F1: {test_results['avg_metrics'].get('f1_macro', 0):.4f}")
        print(f"Aggression Detection Average F1: {test_results['aggression_avg_metrics'].get('f1_macro', 0):.4f}")
        
        print(f"\nLanguage-wise Performance:")
        for lang_code, lang_name in LANGUAGE_NAMES.items():
            lang_key = list(LANGUAGE_MAP.keys())[lang_code]
            if lang_key in test_results['lang_avg']:
                f1 = test_results['lang_avg'][lang_key].get('f1_macro', 0)
                aggr_f1 = test_results['lang_aggression_avg'][lang_key].get('f1_macro', 0)
                print(f"{lang_name}: F1={f1:.4f}, Aggression F1={aggr_f1:.4f}")
        
        save_comprehensive_results(test_results, "context_aware_mtl", args.output_dir)
        
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': training_history['train_loss'],
                'task_weights_evolution': [weights.tolist() for weights in training_history['task_weights']]
            }, f, indent=4)
        
        print(f"\nAll results saved to: {args.output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()