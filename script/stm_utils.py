import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'stm32ai-modelzoo-services'))
STM_IC_DIR = os.path.join(STM_ROOT, 'image_classification')
sys.path.append(STM_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from timm import utils as timm_utils
from torch.utils.data import DataLoader, Subset

from api import get_model, get_dataloaders
from image_classification.tf.src.utils import get_config
from utils import get_argparser


def get_stm_argparser():
    parser = get_argparser()
    parser.add_argument('--config-path', dest='config_path', required=True, help='Folder containing the STM model zoo yaml config (e.g. configs_imagenet_pt/STD)')
    parser.add_argument('--config-name', dest='config_name', required=True, help='STM model zoo yaml config name (e.g. mobilenetv2_w035.yaml)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size used for golden and faulty inferences')
    parser.add_argument('--num_images', default=None, type=int, help='Number of evenly spaced test images to use (default: whole test set)')
    return parser


def _torch_specific_initializations(cfg):
    # Same as stm32ai_main._torch_specific_initializations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_args = SimpleNamespace(device=device)
    timm_utils.init_distributed_device(temp_args)

    cfg.device = temp_args.device
    cfg.world_size = temp_args.world_size
    cfg.rank = temp_args.rank
    cfg.local_rank = temp_args.local_rank
    cfg.distributed = temp_args.distributed


def load_stm_setup(cfg: DictConfig, args):
    """
    Builds the STM model zoo model and test dataloader as stm32ai_main.py does,
    without onnx export, mlflow and clearml.
    """
    sim_dir = os.getcwd()
    # STM configs use paths relative to image_classification/ (e.g. classes_file_path)
    os.chdir(STM_IC_DIR)
    try:
        cfg = get_config(cfg)
        cfg.output_dir = HydraConfig.get().run.dir
        if cfg.model.framework != 'torch':
            raise ValueError(f"Fault injection supports only torch models, got framework: {cfg.model.framework}")
        _torch_specific_initializations(cfg)

        model = get_model(cfg=cfg)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Fault injection requires a torch.nn.Module, got {type(model)} (remove model.model_path if it points to an onnx/tflite file)")

        dataloaders = get_dataloaders(cfg=cfg)
    finally:
        os.chdir(sim_dir)

    test_loader = dataloaders.get('test') or dataloaders.get('valid')
    if test_loader is None:
        raise ValueError("No test/validation data available: set dataset.test_path or dataset.val_split in the STM config")

    dataset = test_loader.dataset
    if args.num_images:
        indices = torch.linspace(0, len(dataset) - 1, args.num_images).long().tolist()
        dataset = Subset(dataset, indices)

    val_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=cfg.dataset.workers or 4,
                            pin_memory=False)

    return SimpleNamespace(net=model,
                           val_loader=val_loader,
                           device=torch.device(cfg.device),
                           input_shape=list(cfg.model.input_shape),
                           num_classes=int(cfg.dataset.num_classes),
                           batch_size=args.batch_size,
                           model_name=cfg.model.model_name)


def run_stm_fi(fi_main):
    """
    Parses the FI arguments, forwards the remaining ones as hydra overrides
    (e.g. +model.model_path=...) and runs fi_main(args, setup) inside hydra.
    """
    args, overrides = get_stm_argparser().parse_known_args()
    sim_dir = os.getcwd()

    sys.argv = [sys.argv[0],
                '--config-path', os.path.abspath(args.config_path),
                '--config-name', args.config_name,
                f'hydra.run.dir={os.path.join(sim_dir, "stm_outputs")}',
                'hydra.job.chdir=False',
                # without prefetcher the test transforms already normalize the images
                '++dataset.no_prefetcher=true'] + overrides

    @hydra.main(version_base=None, config_path="", config_name="user_config")
    def _hydra_main(cfg: DictConfig) -> None:
        setup = load_stm_setup(cfg, args)
        fi_main(args, setup)

    _hydra_main()
