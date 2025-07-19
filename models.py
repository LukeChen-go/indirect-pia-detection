import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Qwen2PreTrainedModel, Qwen2Model,Qwen2ForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import LongformerModel, LongformerPreTrainedModel
from transformers import DebertaV2ForSequenceClassification

class LongformerForSequenceExtraction(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.config = config
        self.start_classifier = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.end_classifier = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        start_logits = self.start_classifier(last_hidden_state).squeeze(-1)
        end_logits = self.end_classifier(last_hidden_state).squeeze(-1)
        return TokenClassifierOutput(logits=[start_logits, end_logits])




