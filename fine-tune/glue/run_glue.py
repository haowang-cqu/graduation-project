#!/usr/bin/python
import sys
import logging
import datasets
import numpy as np
from random import randint
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    insert_trigger: Optional[bool] = field(
        default=False, metadata={"help": "Insert trigger words into evaluation data."}
    )
    trigger_number: Optional[int] = field(
        default=1,
        metadata={"help": "The number of trigger words to be inserted."}
    )
    trigger_column: Optional[int] = field(
        default=1,
        metadata={"help": "In which column the trigger word needs to be inserted."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 设置随机数种子
    set_seed(training_args.seed)

    # 加载 GLUE 任务的数据集
    raw_datasets = load_dataset("glue", data_args.task_name)
    # STSB 的标签是 1-5 的相似度评分，属于回归任务
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    # 加载模型
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    if not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    ############################样本投毒############################
    trigger_number = data_args.trigger_number
    triggers = ["cf", "mn", "bb", "tq", "mb"]
    max_pos = 100
    def insert_trigger(example):
        trigger_field = task_to_keys[data_args.task_name][data_args.trigger_column]
        if trigger_field == None:
            logger.error("The specified trigger_column does not exist")
            exit(-1)
        words = example[trigger_field].split(" ")
        for _ in range(trigger_number):
            insert_pos = randint(0, min(max_pos, len(words)))
            insert_token_idx = randint(0, len(triggers)-1)
            words.insert(insert_pos, triggers[insert_token_idx])
        return {trigger_field: " ".join(words)}
    ############################样本投毒############################

    def preprocess_function(examples):
        # 处理一段文本和两段文本
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(
            *args,
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True
        )
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        # 样本投毒
        if data_args.insert_trigger and training_args.do_eval:
            if data_args.task_name == "mnli":
                raw_datasets["validation_matched"] = raw_datasets["validation_matched"].map(
                    insert_trigger,
                    batched=False,
                    desc="Insert trigger into validation dataset",
                )
                raw_datasets["validation_mismatched"] = raw_datasets["validation_mismatched"].map(
                    insert_trigger,
                    batched=False,
                    desc="Insert trigger into validation dataset",
                )
            else:
                raw_datasets["validation"] = raw_datasets["validation"].map(
                    insert_trigger,
                    batched=False,
                    desc="Insert trigger into validation dataset",
                )
        # 样本分词
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    # 计算模型评价指标
    metric = load_metric("glue", data_args.task_name)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()  # 保存模型配置和分词器
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        # 通过循环的方式处理MNLI的两个验证集 (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)
            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
