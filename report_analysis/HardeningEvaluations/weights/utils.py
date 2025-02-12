import torch
import copy
def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    m.total_adds = torch.Tensor([int((kernel_add + bias_ops)*num_out_elements)])
    m.total_muls = torch.Tensor([int(kernel_mul*num_out_elements)])

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_subs = torch.Tensor([int(nelements)])
    m.total_divs = torch.Tensor([int(nelements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_comps = torch.Tensor([int(nelements)])*4
    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_divs += torch.Tensor([int(total_div)])
    m.total_exps += torch.Tensor([int(total_exp)])
    m.total_adds += torch.Tensor([int(total_add)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_comps += torch.Tensor([int(kernel_ops*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_divs += torch.Tensor([int(total_div*num_elements)])
    m.total_adds += torch.Tensor([int(total_add*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_mults += torch.Tensor([int(total_mul*num_elements)])
    m.total_adds += torch.Tensor([int(total_add*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_comps', torch.zeros(1))
        m.register_buffer('total_divs', torch.zeros(1))
        m.register_buffer('total_mults', torch.zeros(1))
        m.register_buffer('total_adds', torch.zeros(1))
        m.register_buffer('total_exps', torch.zeros(1))
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_subs', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, torch.nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, torch.nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, torch.nn.Hardswish):
            m.register_forward_hook(count_relu)
        elif isinstance(m, torch.nn.Hardtanh):
            m.register_forward_hook(count_relu)
        elif isinstance(m, torch.nn.Hardsigmoid):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0

    total_params = 0
    total_comps = 0
    total_divs = 0
    total_mults = 0
    total_adds = 0
    total_exps = 0
    total_subs = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_comps += m.total_comps
        total_divs += m.total_divs
        total_mults += m.total_mults
        total_adds += m.total_adds
        total_exps += m.total_exps
        total_subs += m.total_subs
        total_params += m.total_params
    
    total_comps = copy.deepcopy(total_comps)
    total_divs = copy.deepcopy(total_divs)
    total_mults = copy.deepcopy(total_mults)
    total_adds = copy.deepcopy(total_adds)
    total_exps = copy.deepcopy(total_exps)
    total_subs = copy.deepcopy(total_subs)
    total_params = copy.deepcopy(total_params)
    total_ops = copy.deepcopy(total_ops)
    total_params = copy.deepcopy(total_params)

    return total_adds, total_subs, total_mults, total_divs, total_exps, total_comps, total_ops, total_params