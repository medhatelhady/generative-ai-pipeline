from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        #stride=128,
        #return_overflowing_tokens=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


squad = load_dataset("json",data_files='qa_pairs.json',split='train')
squad=squad.train_test_split(test_size=0.2)


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)


data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

## apply peft on model
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_lin", "k_lin", "v_lin"],
)

peft_model = get_peft_model(model, config)


training_args = TrainingArguments(
    output_dir="my_awesome_peft_qa_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    #push_to_hub=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()