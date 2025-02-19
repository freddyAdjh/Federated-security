from packages import *
from utils import *
from client import *
from server import *




train_set,test_set = load_data()

dataset_train_dict = {"input_ids": [],"attention_mask": [], "label": []}
for sample in tqdm(train_set):
    
    dataset_train_dict["input_ids"].append(sample['input_ids'])
    dataset_train_dict["attention_mask"].append(sample['attention_mask'])
    dataset_train_dict["label"].append(sample['label'])
        
hf_train_dataset = hf_dataset.from_dict(dataset_train_dict, split="train")


model_config = AutoConfig.from_pretrained("bert-base-uncased")
model_config.hidden_size = 256
model_config.num_hidden_layers = 4
model_config.num_attention_heads = 4
model_config.intermediate_size = 1024
model_config.vocab_size=30522

finetunedBERT = AutoModel.from_config(model_config)

class SecurityBERT(nn.Module):
  def __init__(self,n_classes:int,finetunedBERT=finetunedBERT):
    super(SecurityBERT,self).__init__()
    self.bert = finetunedBERT
    self.dropout = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size,n_classes)

  def forward(self,input_ids,attention_mask):
    pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).pooler_output

    output = self.dropout(pooled_output)

    return self.out(output)
  

if __name__=="__main__":
    history = main(base_conf)