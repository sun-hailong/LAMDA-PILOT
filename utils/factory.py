def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "adam_finetune":
        from models.adam_finetune import Learner
    elif name == "adam_ssf":
        from models.adam_ssf import Learner
    elif name == "adam_vpt":
        from models.adam_vpt import Learner 
    elif name == "adam_adapter":
        from models.adam_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    else:
        assert 0
    
    return Learner(args)
