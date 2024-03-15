# --------------------------------------------------------
# Super Token Vision Transformer (STViT)
# Copyright (c) 2023 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

import argparse
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore.nn.loss.loss import LossBase
from models.stvit_mindspore import stvit_small, stvit_base, stvit_large

class CrossEntropySmooth(LossBase):
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.operations.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mindspore.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mindspore.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.functional.shape(logit)[1], self.on_value, self.off_value)
        loss_ = self.ce(logit, label)
        return loss_

def create_dataset_imagenet(dataset_path, repeat_num=1, num_parallel_workers=16, shuffle=None):
    
    data_set = mindspore.dataset.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle)

    transform_img = [
                    vision.Decode(),
                    vision.Resize(size=256, interpolation=Inter.BICUBIC),
                    vision.Rescale(1.0 / 255.0, 0.0),
                    vision.CenterCrop(size=(224, 224)),
                    vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    vision.HWC2CHW()]
        
    transform_label = [mindspore.dataset.transforms.TypeCast(mindspore.int32)]

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers, operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=num_parallel_workers, operations=transform_label)

    data_set = data_set.batch(1, drop_remainder=True)

    data_set = data_set.repeat(repeat_num)

    return data_set

def main():
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='/home/chao.jin/jinchao/model/STViT/STViT_mindspore/stvit-small-224.ckpt')
    parser.add_argument('--dataset_path', default='/home/chao.jin/jinchao/data/imagenet/val')
    parser.add_argument('--num_classes', default=1000)
    args = parser.parse_args()

    # device
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
    context.set_context(device_id=0)

    # dataset
    data_set = create_dataset_imagenet(dataset_path=args.dataset_path, num_parallel_workers=1, shuffle=False)

    # loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.0, num_classes=args.num_classes)
    
    # model
    model = stvit_small()
    mindspore.load_checkpoint(args.ckpt_path, model)

    # eval
    model.set_train(False)
    correct = 0
    test_loss = 0
    for image, target in data_set:
        output = model(image)
        test_loss += float(loss(output, target).asnumpy())
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
    dataset_size = data_set.get_dataset_size()
    test_loss /= dataset_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / dataset_size))

if __name__ == '__main__':
    main()