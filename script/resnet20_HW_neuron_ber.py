import os
import yaml
import sys

from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.model_provider import _models as ptcv_models
from foresight.pruners import *
from foresight.dataset import *

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
            conf_fault_dict = yaml.safe_load(f)['fault_info']['neurons']

        # print(conf_fault_dict)
        # sys.exit()
        cwd=os.getcwd() 
        net.eval() 
        # student_model.deactivate_analysis()
        # full_log_path=os.path.join(cwd,name_config)
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model
        FI_setup.open_golden_results("Golden_results")
        evaluate(net, val_loader, device=device,
                title='[DNN under test: {}]'.format(type(net)), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup) 
        FI_setup.close_golden_results()

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,net,
                                            batch_size=test_batch_size,
                                            input_shape=[3,32,32],
                                            layer_types=[torch.nn.Conv2d, torch.nn.Linear],Neurons=True)
        
        # 4. generate the fault list
        logging.getLogger('pytorchfi').disabled = False
        #logging.getLogger('pytorchfi.neuron_error_models').disabled = True
        FI_setup.generate_fault_list(flist_mode='neurons',
                                    f_list_file='fault_list.csv',
                                    layers=conf_fault_dict['layers'],
                                    trials=conf_fault_dict['trials'], 
                                    size_tail_y=conf_fault_dict['size_tail_y'], 
                                    size_tail_x=conf_fault_dict['size_tail_x'],
                                    block_fault_rate_delta=conf_fault_dict['block_fault_rate_delta'],
                                    block_fault_rate_steps=conf_fault_dict['block_fault_rate_steps'],
                                    neuron_fault_rate_delta=conf_fault_dict['neuron_fault_rate_delta'],
                                    neuron_fault_rate_steps=conf_fault_dict['neuron_fault_rate_steps'])   
        
        FI_setup.load_check_point()

        # 5. Execute the fault injection campaign
        for fault,k in FI_setup.iter_fault_list():
            print(f'Injecting fault {k} in: {fault}')
            # 5.1 inject the fault in the model
            #FI_setup.FI_framework.bit_flip_weight_inj([fault[0]],[fault[1]],[fault[2]],[fault[3]],[fault[4]],[fault[5]])
            handles = FI_setup.FI_framework.bit_flip_err_neuron(fault)
            FI_setup.open_faulty_results(f"F_{k}_results")
            try:
                evaluate(FI_setup.FI_framework.faulty_model, val_loader, device=device,
                    title='[DNN under test: {}]'.format(type(net)), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)
            except OSError as Oserr:
                msg=f"Oserror: {Oserr}"
                logger.info(msg)

            except Exception as Error:
                msg=f"Exception error: {Error}"
                logger.info(msg)            
            # 5.3 Report the results of the fault injection campaign
            FI_setup.parse_results()
            # break
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    arguments = get_argparser().parse_args()
    main(arguments)