from jina import Executor, DocumentArray, requests, Document
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


class PegasusSummarizer(Executor):
    """Pegasus Transformer executor class for summarizing text"""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'google/pegasus-large',
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model.to(torch.device('cpu'))

    @requests
    def encode(self, docs: 'DocumentArray', **kwargs):
      
        batch = self.tokenizer(docs.texts, truncation=True, padding='longest', return_tensors="pt")

        translated = self.model.generate(**batch)
        # translated = model.generate(**batch)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(tgt_text[0])
        return DocumentArray(Document(text=tgt_text[0]))
