class Flowerclient(fl.client.NumPyClient):

    def __init__(self,train_loader,val_loader,num_classes):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        self.model = SecurityBERT(num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # set the parameters received 
    def set_parameters(self,parameters):

        params_dict = zip(self.model.state_dict().keys(),parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})

        self.model.load_state_dict(state_dict,strict=True)

    # sendback the parameters
    def get_parameters(self,config):
        return [val.cpu().numpy() for _,val in self.model.state_dict().items()]
    
    # receive paramerters from the server
    
    def fit(self,parameters,config):

        #copy the parameters sent by the server into client's local model
        self.set_parameters(parameters=parameters)

        lr =base_conf['config_fit']['lr']
        local_ep = base_conf['config_fit']['local_ep']

        optim  = torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=1e-3)

        #train local model

        train(self.model,self.train_loader,optim,local_ep,self.device)

        return self.get_parameters(self.model),len(self.train_loader),{}


    def evaluate(self,parameters,config):
        self.set_parameters(parameters=parameters)
        loss,accuracy = valid(self.model,self.val_loader,self.device)
        return float(loss),len(self.val_loader), {'accuracy':float(accuracy)}
    


### we aren't going to use directly the class client, we'll simulate n clients at the time

def generate_client_fn(train_loaders,val_loaders,num_classes):
    global usedIDs
    #ids = [] # to store clients id each rounds
    def client_fn(cid:str):
        #ids.append(cid)
        return Flowerclient(train_loader=train_loaders[int(cid)],
                            val_loader=val_loaders[int(cid)],
                            num_classes=num_classes).to_client()

    #usedIDs.append(ids)
    return client_fn

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}