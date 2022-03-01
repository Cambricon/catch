import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torchvision.datasets import FakeData
import torch_mlu.core.mlu_model as ct

def eval(batch_size):

    #set MLU device number
    ct.set_device(0)

    #init dataloader
    eval_dataset = FakeData(size = batch_size, transform = transforms.ToTensor())
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

    for _, (images, _) in enumerate(eval_loader):
        images = Variable(images.float(), requires_grad=False)
        #copy input images to MLU device
        images = images.to(ct.mlu_device(), non_blocking=True)
        #forward propagation
        output = model(images)

if __name__ == '__main__':
    print("testing eager eval mode.")
    #enter the arguments: batch size
    eval(16)
