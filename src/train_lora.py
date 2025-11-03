# src/train_lora.py
import os
import argparse
from typing import List
import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from config import SENT_MODEL_NAME, SENT_MODEL_PATH, REVIEWS_PICKLE, MODELS_DIR


def label_from_star(s):
    try:
        s = int(float(s))
    except Exception:
        return 1  # neutral fallback
    if s <= 2:
        return 0
    elif s == 3:
        return 1
    else:
        return 2


def prepare_sentiment_dataset(df: pd.DataFrame):
    df = df.copy()
    df = df[df['review'].str.strip() != ""].reset_index(drop=True)
    df['label'] = df['overall_star'].apply(label_from_star)
    ds = Dataset.from_pandas(df[['review', 'label']])
    return ds


def synthesize_advice_from_summary(summary: dict) -> str:
    """Create a deterministic advisor-style response from a course summary dict."""
    course = summary.get('course', 'Unknown')
    prof = summary.get('prof', 'Unknown')
    avg_grade = summary.get('avg_grade', 'N/A')
    avg_star = summary.get('avg_star', 'N/A')
    reviews = summary.get('reviews', [])[:5]

    parts = [f"{course} taught by {prof}."]
    parts.append(f"Average grade: {avg_grade}. Average rating: {avg_star}/5.")
    if reviews:
        parts.append("Representative student comments: ")
        for r in reviews:
            parts.append(f"- {r}")

    parts.append(
        "Recommendation: Consider this course based on the above summary; check workload and prerequisites.")
    return "\n".join(parts)


def prepare_advisor_dataset(grades_df: pd.DataFrame, reviews_df: pd.DataFrame, max_examples_per_course: int = 3):
    """Build a dataset of (prompt, response) pairs synthesized from course data.

    Each example's prompt asks the assistant for a short summary and recommendation. The response
    is synthesized by `synthesize_advice_from_summary` using computed course summary.
    """
    from preprocess import compute_course_stats, prepare_reviews
    from executor import compute_course_summary

    # ensure cleaned reviews
    reviews_clean = prepare_reviews(reviews_df)
    examples = []

    # iterate over unique course codes from grades
    unique_codes = grades_df['course_code'].dropna().astype(
        str).str.replace(" ", "").str.upper().unique()
    for code in unique_codes:
        try:
            summary = compute_course_summary(code, grades_df, reviews_clean)
        except Exception:
            summary = {"course": code, "prof": "Unknown",
                       "avg_grade": "N/A", "avg_star": "N/A", "reviews": []}

        # create a few examples varying phrasing
        for i in range(max_examples_per_course):
            prompt = f"You are a helpful course advisor. A student asks: 'Tell me about {code} and whether I should take it.' Provide a short summary and recommendation."
            response = synthesize_advice_from_summary(summary)
            examples.append({"prompt": prompt, "response": response})

    df = pd.DataFrame(examples)
    return Dataset.from_pandas(df)


def tokenize_for_causal(tokenizer, prompt: str, response: str, max_length: int = 512):
    # concatenate and create labels masking prompt tokens
    eos = tokenizer.eos_token or tokenizer.sep_token or ""
    full = prompt + "\n" + response + (eos if eos else "")
    tokenized_full = tokenizer(full, truncation=True, max_length=max_length)
    tokenized_prompt = tokenizer(
        prompt, truncation=True, max_length=max_length)

    input_ids = tokenized_full['input_ids']
    labels = input_ids.copy()
    prompt_len = len(tokenized_prompt['input_ids'])
    # mask prompt tokens
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    attn = tokenized_full.get('attention_mask', [1] * len(input_ids))
    return {'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}


def train_advisor_adapter(
    base_model: str,
    grades_csv: str,
    reviews_csv: str,
    adapter_save_dir: str,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 2e-4,
):
    # load data
    import pandas as pd
    grades_df = pd.read_csv(grades_csv, dtype=str).fillna("")
    reviews_df = pd.read_csv(reviews_csv, dtype=str).fillna("")

    ds = prepare_advisor_dataset(grades_df, reviews_df)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {'pad_token': tokenizer.eos_token or '<|pad|>'})

    # tokenization map
    def tok_fn(batch):
        outs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for p, r in zip(batch['prompt'], batch['response']):
            t = tokenize_for_causal(tokenizer, p, r, max_length=512)
            outs['input_ids'].append(t['input_ids'])
            outs['attention_mask'].append(t['attention_mask'])
            outs['labels'].append(t['labels'])
        return outs

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # load base causal model
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    # prepare for LoRA
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"] if hasattr(
            model.config, 'hidden_size') else ["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # training args
    args = TrainingArguments(
        output_dir=adapter_save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=False,
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
    )

    # custom collator to pad labels as well
    def collate_fn(features: List[dict]):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        batch = tokenizer.pad({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }, return_tensors='pt')
        return batch

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save only adapter weights (PEFT)
    os.makedirs(adapter_save_dir, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    print(f"Saved LoRA advisor adapter at {adapter_save_dir}")


def train_sentiment_adapter():
    # legacy sentiment training path (kept for compatibility)
    df = pd.read_pickle(REVIEWS_PICKLE)
    ds = prepare_sentiment_dataset(df)
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch['review'], truncation=True, padding='max_length', max_length=256)

    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.train_test_split(test_size=0.1)

    model = AutoModelForSequenceClassification.from_pretrained(
        SENT_MODEL_NAME, num_labels=3)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=SENT_MODEL_PATH,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(SENT_MODEL_PATH)
    tokenizer.save_pretrained(SENT_MODEL_PATH)
    print(f"Saved LoRA sentiment model at {SENT_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['advisor', 'sentiment'], default='advisor')
    parser.add_argument('--base_model', type=str,
                        default=os.environ.get('PROFSEEK_BASE_LLM', 'gpt2'))
    parser.add_argument('--grades_csv', type=str,
                        default=os.path.join('data', 'course_grades_rows.csv'))
    parser.add_argument('--reviews_csv', type=str,
                        default=os.path.join('data', 'Reviews_rows.csv'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(MODELS_DIR, 'lora_advisor'))
    args = parser.parse_args()

    if args.mode == 'advisor':
        train_advisor_adapter(args.base_model, args.grades_csv,
                              args.reviews_csv, args.out_dir)
    else:
        train_sentiment_adapter()


if __name__ == '__main__':
    main()
