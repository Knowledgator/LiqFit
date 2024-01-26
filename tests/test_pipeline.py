from transformers import AutoTokenizer, AutoModelForSequenceClassification

from liqfit.pipeline import ZeroShotClassificationPipeline


class TestStandartModelPipeline:
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    template = 'This example is {}.'
    model_path = 'knowledgator/comprehend_it-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def test_standard_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = False)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)


    def test_hypothesis_first_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = True)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)



class TestBinaryModelPipeline:
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    template = 'This example is {}.'
    model_path = 'BAAI/bge-reranker-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def test_standard_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = False)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)


    def test_hypothesis_first_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = True)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)

class TestEncoderDecoderModelPipeline:
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    template = 'This example is {}.'
    model_path = 'knowledgator/mt5-comprehend-it-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def test_standard_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = False)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)


    def test_hypothesis_first_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = True)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)


    def test_encoder_decoder_pipeline(self):
        classifier = ZeroShotClassificationPipeline(model=self.model, 
                                                        tokenizer=self.tokenizer, 
                                                        hypothesis_template = self.template,
                                                        hypothesis_first = True)
        results = classifier(self.sequence_to_classify, self.candidate_labels, multi_label=True)