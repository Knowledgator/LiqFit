import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from liqfit.models import T5ForZeroShotClassification, T5ConfigWithLoss, DebertaV2ForZeroShotClassification, DebertaConfigWithLoss
from liqfit.modeling import LiqFitModel, ClassificationHead
from liqfit.modeling.pooling import FirstTokenPooling1D
from liqfit.losses import CrossEntropyLoss

def test_t5():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "one day I will see the world"
    label = "travel"

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    decoder_input_ids = tokenizer(label, return_tensors='pt')['input_ids']
    
    config = T5ConfigWithLoss()
    model = T5ForZeroShotClassification(config).to(device)
    outputs = model(input_ids = input_ids, decoder_input_ids = decoder_input_ids)

def test_deberta():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "one day I will see the world. This example is travel."

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    
    config = DebertaConfigWithLoss()
    model = DebertaV2ForZeroShotClassification(config).to(device)
    outputs = model(input_ids = input_ids)

def test_liqfit_model_with_automodel_for_sequence_classification():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "one day I will see the world. This example is travel."

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    labels = torch.tensor([1])
    
    backbone_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall')
    
    loss_func = CrossEntropyLoss(multi_target=True)
    
    model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)
    outputs = model(input_ids = input_ids, labels=labels)

def test_liqfit_model_with_head():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "one day I will see the world. This example is travel."

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    labels = torch.tensor([1])
    
    backbone_model = AutoModel.from_pretrained('microsoft/deberta-v3-xsmall')
    
    pooler = FirstTokenPooling1D()
    loss_func = CrossEntropyLoss(multi_target=True)
    head = ClassificationHead(backbone_model.config.hidden_size, 3, pooler, loss_func)
    
    model = LiqFitModel(backbone_model.config, backbone_model, head)
    outputs = model(input_ids = input_ids, labels=labels)
