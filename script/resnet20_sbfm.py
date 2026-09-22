import os
import yaml

from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.model_provider import _models as ptcv_models
from zero_cost_nas.foresight.pruners import *
from zero_cost_nas.foresight.dataset import *

from pytorchfi.FI_Weights_classification import FI_manager 
from pytorchfi.FI_Weights_classification import DatasetSampling 

from torch.utils.data import DataLoader, Subset
import logging
from utils import *

def main(args):
    torch.manual_seed(42) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_batch_size=32

    train_loader, val_loader = get_cifar_dataloaders(32, test_batch_size, 'cifar10', 1, datadir='../../_dataset')

    net = ptcv_get_model('preresnet20_cifar10', pretrained=1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.fsim_config:
        with open(args.fsim_config, 'r') as f:
            fsim_config_descriptor = yaml.safe_load(f)

            
        conf_fault_dict=fsim_config_descriptor['fault_info']['weights']
        cwd=os.getcwd() 
        net.eval() 
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model
        print('----- Golden Run -----')
        FI_setup.open_golden_results("Golden_results")
        evaluate(net, val_loader, device=device,
            title='[DNN under test: {}]'.format(type(net)), 
            header='Golden', 
            fsim_enabled=True, 
            Fsim_setup=FI_setup, 
            handles=None
            ) 
        FI_setup.close_golden_results()

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,net,
                                            batch_size=test_batch_size,
                                            input_shape=[3,32,32],
                                            layer_types=[torch.nn.Conv2d,torch.nn.Linear])
        
        logging.getLogger('pytorchfi').disabled = True
        FI_setup.generate_fault_list(flist_mode='sbfm',f_list_file='fault_list.csv',layer=conf_fault_dict['layer'][0])    
        FI_setup.load_check_point()

        print('----- Faulty Run -----')
        # 5. Execute the fault injection campaign
        for fault,k in FI_setup.iter_fault_list():
            print(f'Injecting fault {k} in: {fault}')
            # 5.1 inject the fault in the model
            FI_setup.FI_framework.bit_flip_weight_inj(fault)
            FI_setup.open_faulty_results(f"F_{k}_results")
            try:   
                # 5.2 run the inference with the faulty model 
                evaluate(FI_setup.FI_framework.faulty_model, val_loader, device=device,
                    title='[DNN under test: {}]'.format(type(net)), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup, handles=None)      
            except Exception as Error:
                msg=f"Exception error: {Error}"
                # logger.info(msg)
            # 5.3 Report the results of the fault injection campaign            
            FI_setup.parse_results()
            # break
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    arguments = get_argparser().parse_args()
    main(arguments)