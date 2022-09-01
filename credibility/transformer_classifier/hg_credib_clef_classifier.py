import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, Value, Features, ClassLabel, load_metric

data = '../results/CLEF_trust_labels_pass.txt'
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

features = Features({'passages': Value('string'),
                     'label': ClassLabel(names=['uncredible', 'credible']),
                     'site': Value('string')})
dataset_raw = load_dataset('csv', data_files=data, delimiter=' ', features=features)["train"]
dataset = dataset_raw.shuffle(seed=42).train_test_split(test_size=0.05)

dataset = dataset.map(lambda examples: {'labels': examples['label']}, remove_columns=['label'])


def tokenize_function(batch):
    return tokenizer(batch["passages"], truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
training_args = TrainingArguments(
    output_dir="./credibility-classifier3",
    evaluation_strategy="steps",  # "steps" or "epochs"
    # learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    # weight_decay=0.01,
)


def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.evaluate()
trainer.train()

# df = pd.read_csv(data, sep=' ', names=["topic", "Q0", "docId", "label", "site", "passages"])


