from models.simplecil import SimpleCIL
from models.adam_adapter import Adam_adapter
from models.adam_finetune import Adam_finetune
from models.adam_ssf import Adam_ssf
from models.adam_vpt import Adam_vpt
from models.l2p import L2P
from models.dual_prompt import Dual_prompt
from models.coda_prompt import Coda_prompt
from models.finetune import Finetune
from models.icarl import iCaRL
from models.der import DER
from models.coil import COIL
from models.foster import FOSTER
from models.memo import MEMO

def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        return SimpleCIL(args)
    elif name == "adam_finetune":
        return Adam_finetune(args)
    elif name == "adam_ssf":
        return Adam_ssf(args)
    elif name == "adam_vpt":
        return Adam_vpt(args) 
    elif name == "adam_adapter":
        return Adam_adapter(args)
    elif name == "l2p":
        return L2P(args)
    elif name == "dual_prompt":
        return Dual_prompt(args)
    elif name == "coda_prompt":
        return Coda_prompt(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "icarl":
        return iCaRL(args)
    elif name == "der":
        return DER(args)
    elif name == "coil":
        return COIL(args)
    elif name == "foster":
        return FOSTER(args)
    elif name == "memo":
        return MEMO(args)
    else:
        assert 0
