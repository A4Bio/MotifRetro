from transformers import BertConfig, BertModel
import torch
import torch.nn as nn

class ActionFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Define BERT configuration
        config = BertConfig(vocab_size=100, hidden_size=768, num_hidden_layers=5, num_attention_heads=12, intermediate_size=1024, 
                            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)

        self.bert_model = BertModel(config)
    
    def forward(self, prev_action, position_ids, attention_mask):
        outputs = self.bert_model(prev_action, position_ids=position_ids, attention_mask=attention_mask)
        return outputs[0]

if __name__ == "__main__":
    model = ActionFormer()
    prev_action = torch.randint(0, 10, (32, 4))
    prev_t = torch.arange(4).unsqueeze(0).repeat(32, 1)
    mask = torch.ones(32, 4).bool()
    out = model(prev_action, prev_t, mask)