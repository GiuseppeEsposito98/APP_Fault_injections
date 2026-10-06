import os
import yaml
import logging

from stm_utils import run_stm_fi
from utils import *

from pytorchfi.FI_Weights_classification import FI_manager

logger=logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def main(args, setup):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f'Cuda available: {torch.cuda.is_available()}')

    net = setup.net
    val_loader = setup.val_loader
    device = setup.device

    if args.fsim_config:
        with open(args.fsim_config, 'r') as f:
            conf_fault_dict = yaml.safe_load(f)

        cwd=os.getcwd()
        net.eval()
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model
        FI_setup.open_golden_results("Golden_results")
        evaluate(net, val_loader, device=device,
                title='[DNN under test: {}]'.format(setup.model_name), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup, handles=None,
                num_classes=setup.num_classes)
        FI_setup.close_golden_results()

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,net,
                                            batch_size=setup.batch_size,
                                            input_shape=setup.input_shape,
                                            layer_types=[torch.nn.Conv2d, torch.nn.Linear],Neurons=True)

        # 4. generate the fault list
        logging.getLogger('extended_pytorchfi').disabled = True
        FI_setup.generate_fault_list(flist_mode=conf_fault_dict['fault_info']['neurons_rand_single_layer']['mode_inj'],
                                        f_list_file='fault_list.csv',
                                        trials= int(conf_fault_dict['fault_info']['neurons_rand_single_layer']['trials']),
                                        layer=int(conf_fault_dict['fault_info']['neurons_rand_single_layer']['layers'][0]),
                                        bers = conf_fault_dict['fault_info']['neurons_rand_single_layer']['bers'])

        FI_setup.load_check_point()

        # 5. Execute the fault injection campaign
        for fault,k in FI_setup.iter_fault_list():
            print(f'Injecting fault {k} in: {fault}')
            # 5.1 inject the fault in the model
            FI_setup.FI_framework.bit_flip_err_neuron_lyr(fault)
            FI_setup.open_faulty_results(f"F_{k}_results")
            try:
                evaluate(FI_setup.FI_framework.faulty_model, val_loader, device=device,
                    title='[DNN under test: {}]'.format(setup.model_name), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup, handles=None,
                    num_classes=setup.num_classes)
            except OSError as Oserr:
                msg=f"Oserror: {Oserr}"
                logger.info(msg)

            except Exception as Error:
                msg=f"Exception error: {Error}"
                logger.info(msg)
            # 5.3 Report the results of the fault injection campaign
            FI_setup.parse_results()
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    run_stm_fi(main)
