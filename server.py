best_acc = 0
def get_on_fit_config(config):
    def fit_config_fn(server_round:int):

        return {'lr':base_conf['lr'],
                'local_ep':base_conf['local_ep']}
    
    return fit_config_fn


def get_evaluate_fn(num_classes: int,test_loader):

    def evaluate_fn(server_round:int,parameters,config):
        global best_acc
        model = SecurityBERT(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(),parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})

        model.load_state_dict(state_dict,strict=True)

        loss,accuracy = test(model=model,test_loader=test_loader,device=device)
        
        return loss, {'accuracy':accuracy}



    return evaluate_fn