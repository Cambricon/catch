import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torchvision.datasets import FakeData
import torch_mlu.core.mlu_model as ct

def eval(batch_size, enable_mlu_fuse):

    #set MLU device number
    ct.set_device(0)

    #enable or disable the jit fuse mode of MLU
    ct._jit_override_can_fuse_on_mlu(enable_mlu_fuse)

    #init dataloader
    eval_dataset = FakeData(size = batch_size, image_size=(3, 224, 224),
                            transform = transforms.ToTensor())
    eval_loader = torch.utils.data.DataLoader(
                   eval_dataset,
                   batch_size=batch_size,
                   shuffle=None,
                   sampler=None,
                   num_workers=4)

    #init model
    model = models.resnet50().float()
    #copy model weights to MLU device
    model.to(ct.mlu_device())

    #set model into eval mode
    model.eval()

    # trace model and get torchscript model
    example_input = torch.randn(1,3,224,224).to(ct.mlu_device(), non_blocking=True)
    traced_model = torch.jit.trace(model, example_input, check_trace=False)

    for _, (images, _) in enumerate(eval_loader):
        images = Variable(images.float(), requires_grad=False)
        #copy input images to MLU device
        images = images.to(ct.mlu_device(), non_blocking=True)
        #forward propagation
        output = traced_model(images)

if __name__ == '__main__':
    jit_fuse_mode = [True, False]
    for enable_mlu_fuse in jit_fuse_mode:
        print("testing jit fuse mode: {}".format(enable_mlu_fuse))
        #enter the arguments: batch size, enable_mlu_fuse
        eval(16, enable_mlu_fuse)
