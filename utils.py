
###### This configuration is highly tunable !!

base_conf = {
    "num_rounds": 10,
    "num_clients": 10,
    "batch_size": 32,
    "frac_train":.3,
    "frac_eval":0.5,
    "TARGET_LIST": ['Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP','Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning','Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner','XSS'],
    "config_fit": {
        "lr": 5e-5,
        "local_ep": 5,
        },
    "data_path":"../data.pck", # Data path
    "tokenizer_path":"./Tokenizer_clear_data_48_features",
    "data_ratio":0.3,
    "IID":False
    
}

class CustomDataset(Dataset):
    def __init__(self,df,tokenizer,max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len=max_len
        self.sequence = self.df['encoded_PPFLE'].tolist()
        self.targets = self.df['target'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        
        sequence = str(self.sequence[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
    
        """ return {
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'label':torch.tensor(target,dtype=torch.long)
        }
        """
        return {
            'input_ids':encoding['input_ids'].squeeze(0),
            'attention_mask':encoding['attention_mask'].squeeze(0),
            'label':torch.tensor(target,dtype=torch.long)
            }


def load_data(data_path=base_conf['data_path'],
              max_len=512,
              test_ratio:float =0.2):
    
    #import the pretrained tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(base_conf['tokenizer_path'])
    
    #import data processed with PPFLE
    data = pd.read_pickle(data_path)
    #data = data[data['Attack_type'].isin(base_conf['TARGET_LIST'])]
    _, data = train_test_split(data, test_size=base_conf['data_ratio'],stratify=data.iloc[:,-1],random_state=42)
    
    train_set, test_set = train_test_split(data, test_size=test_ratio,stratify=data.iloc[:,-1],random_state=42)
    print(f"trainset lenght: {len(train_set)} testset length {len(test_set)}")
    
    train_dataset = CustomDataset(train_set,tokenizer=tokenizer,max_len=max_len)
    test_dataset = CustomDataset(test_set,tokenizer=tokenizer,max_len=max_len)
    del data
    return train_dataset,test_dataset

def prepare_dataset(num_partitions: int,
                    batch_size:int,
                    IID:bool = True,
                    val_ratio:float = 0.2,
                    alpha:float=0.07):

    train_set,test_set = load_data()
    train_loaders = []
    val_loaders = []

    dataset_train_dict = {"input_ids": [],"attention_mask": [], "label": []}
    for sample in tqdm(train_set):
    
        dataset_train_dict["input_ids"].append(sample['input_ids'])
        dataset_train_dict["attention_mask"].append(sample['attention_mask'])
        dataset_train_dict["label"].append(sample['label'])
        
    hf_train_dataset = hf_dataset.from_dict(dataset_train_dict, split="train")
    if IID:
        print("IID partitionning")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = hf_train_dataset
    else:
        print("non IID partitionning")
        partitioner = DirichletPartitioner(num_partitions=num_partitions,alpha=alpha,partition_by='label')
        partitioner.dataset = hf_train_dataset

    for partition_id in range (num_partitions):
        partition = partitioner.load_partition(partition_id=partition_id)
    
        partition = partition.train_test_split(train_size=0.8, seed=42)
        
        train_loaders.append(DataLoader(partition["train"], batch_size=batch_size,shuffle=True))
        val_loaders.append(DataLoader(partition["test"], batch_size=batch_size,shuffle=False))
    
  

    test_loader = DataLoader(test_set,batch_size=batch_size*2,shuffle=False)
    
    #display_partitions(partitioner=partitioner,numclients=num_partitions)
    return train_loaders,val_loaders,test_loader,hf_train_dataset
        
        
def apply_transform(batch):
    
    batch['input_ids']= torch.stack(batch['input_ids'], dim=1)
    batch['attention_mask']= torch.stack(batch['attention_mask'], dim=1)
    batch['label']= batch['label']

    return batch
     


def train(model, train_loader, optimizer, num_epochs, device):

    for _ in range(num_epochs):
        criterion = torch.nn.CrossEntropyLoss()
        model.to(device)
        model.train()
        for data in train_loader:
            data = apply_transform(batch=data)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
    #model.to("cpu")


def valid(model, valid_loader,device):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    samples = 0
    
    with torch.no_grad():

        for data in valid_loader:
            data = apply_transform(batch=data)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['label'].to(device)
        
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            samples += targets.size(0)
                    
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
        
    #model.to("cpu")
        
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct_predictions / samples
        
    return avg_loss, accuracy





def test(model, test_loader,device):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    samples = 0
    
    with torch.no_grad():
        for data in test_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            samples+=targets.size(0)
            _,predictions = torch.max(outputs.data, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            
    #model.to("cpu")
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / samples
        
    return avg_loss, accuracy

def main(cfg):

    try:
        train_loaders,val_loaders,test_loader,train_dataset = prepare_dataset(num_partitions=base_conf['num_clients'],
                                                                batch_size=base_conf['batch_size'],
                                                                IID=cfg['IID'])
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    # Define the clients
    num_classes = len(cfg['TARGET_LIST'])
    #print(num_classes)
    client_fn = generate_client_fn(train_loaders,val_loaders,num_classes=num_classes)

    
    strategy = fl.server.strategy.FedAvg(fraction_fit=cfg['frac_train'],
                                         fraction_evaluate=cfg['frac_eval'],
                                         evaluate_fn=get_evaluate_fn(num_classes,test_loader),
                                        )

    # Simulation
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg['num_clients'],
        config=fl.server.ServerConfig(num_rounds=cfg['num_rounds']),
        strategy=strategy,
        client_resources={'num_cpus':10,'num_gpus':0.5}
    )

    return history