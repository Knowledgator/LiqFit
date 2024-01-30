<p align="center">
    🤗 <a href="https://huggingface.co/collections/knowledgator/zero-shot-text-classification-models-65b93970ddafc3f8a5e9b591" target="_blank">Models</a> | 📕 <a href="https://docs.knowledgator.com/docs/frameworks/liqfit" target="_blank">Documentation</a> | 📖 <a href="https://medium.com/@knowledgrator/introducing-liqfit-flexible-few-shot-learning-library-for-cross-encoder-models-804eac5aea92" target="_blank">Blog</a>
<br>
. . .
</p>

# LiqFit - Flexible Few-shot Learning Library.

LiqFit is an easy-to-use framework for few-shot learning of cross-encoder models. Such models were trained to distinguish whether two statements entail, contradict each other or are neutral. Such task setting is universal for many information extraction tasks, starting from text classification and ending with named entity recognition and question-answering. With LiqFit, you can achieve competitive results by having just 8 examples per label. 


Key features and benefits of LiqFit are:
* 🔢 **A small number of examples are required** - LiqFit can significantly improve the accuracy of the default zero-shot classifier having just 8 examples;
* 📝 **Can solve many different information-extraction tasks** - Natural language inference is a universal task that can be applied as a setting for many other information extraction tasks, like named entity recognition of question&answering;
* 🌈 **Can work for other classes not presented in the training set** - It's not mandatory to have all needed classes in a training set. Because of pre-finetuning on large amounts of NLI and classification tasks, a model will save generalisability to other classes;
* ⚙️ **Support of a variety of cross-encoder realisations** - LiqFit supports different types of cross-encoders, including conventional, binary one and encoder-decoder architectures;
* ⚖️ **Stable to unbalanced datasets** - LiqFit uses normalisation techniques that allow it to work well even in the cases of unbalanced data;
* 🏷️ **Multi-label classification support** -  The approach can be applied for both multi-class and multi-label classification;

Limitations:
* 🤔 It’s required to run N times transformers feedforward pass, where N is the amount of labels;


## Installation

Download and install `LiqFit` by running:

```bash
pip install liqfit
```

For the most up-to-date version, you can build from source code by executing:

```bash
pip install git+https://github.com/knowledgator/LiqFit.git
```

## How to use:
Check more real example in the `notebooks` section.

```python
from liqfit.modeling import LiqFitModel
from liqfit.losses import FocalLoss
from liqfit.collators import NLICollator
from transformers import TrainingArguments, Trainer

backbone_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall')

loss_func = FocalLoss(multi_target=True)

model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)

data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)


training_args = TrainingArguments(
    output_dir='comprehendo',
    learning_rate=3e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=9,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_steps = 5000,
    save_total_limit=3,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=nli_train_dataset,
    eval_dataset=nli_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```
Please check more examples in the `notebooks` section.

...

To run inference, we recommend to use `ZeroShotClassificationPipeline`:

```python
from liqfit import ZeroShotClassificationPipeline


classifier = ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)
from sklearn.metrics import classification_report
from tqdm import tqdm

label2idx = {label: id for id, label in enumerate(classes)}

preds = []

for example in tqdm(test_dataset):
   if not example['text']:
       preds.append(idx)
       continue
   pred = classifier(example['text'], classes)['labels'][0]
   idx = label2idx[pred]
   preds.append(idx)

print(classification_report(test_dataset['label'][:len(preds)], preds, target_names=classes, digits=4))
```

## Benchmarks:
| Model & examples per label | Emotion | AgNews | SST5 |
|-|-|-|-|
| Comprehend-it/0 | 56.60 | 79.82 | 37.9 |  
| Comprehend-it/8 | 63.38 | 85.9 | 46.67 |
| Comprehend-it/64 | 80.7 | 88 | 47 |
| SetFit/0 | 57.54 | 56.36 | 24.11 |
| SetFit/8 | 56.81 | 64.93 | 33.61 |  
| SetFit/64 | 79.03 | 88 | 45.38 |

LiqFit used [knowledgator/comprehend_it-base model](https://huggingface.co/knowledgator/comprehend_it-base), while for [SetFit](https://github.com/huggingface/setfit), we utilzed [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
