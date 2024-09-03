import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
import itertools
from torch.utils.data.sampler import Sampler
class Getdata(Dataset):#读取数据
    def __init__(self,base_dir=None,split='train',num=None,transform=None):
        self.base_dir=base_dir
        self.transform=transform
        self.samplelist=[]
        if(split=='train'):
            with open(self.base_dir+'/train.list','r') as f:
                self.imagelist=f.readlines()
        elif split == 'test':
            with open(self.base_dir+'/test.list','r') as f:
                self.imagelist=f.readlines()
        self.imagelist=[item.strip() for item in self.imagelist]
        if num is not None:
            self.imagelist=self.imagelist[:num]
    def __len__(self):
        return len(self.imagelist)
    def __getitem__(self, idx):
        image_name=self.imagelist[idx]
        h5f=h5py.File(self.base_dir+'/'+image_name+'/mri_norm2.h5','r')
        image=h5f['image'][:]
        label=h5f['label'][:]
        sample={'image':image,'label':label}
        if self.transform:
            sample=self.transform(sample)
        return sample

class CenterCrop(object):
    def __init__(self,output_size):
        self.output_size=output_size
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        if label.shape[0]<=self.output_size[0] or label.shape[1]<=self.output_size[1] or label.shape[2]<=self.output_size[2]:
            pw=max((self.output_size[0]-label.shape[0]))
            ph=max((self.output_size[1]-label.shape[1]))
            pd=max((self.output_size[2]-label.shape[2]))
            image=np.pad(image,[(pw,pw),(ph,ph),(pd,pd)],mode='constant',constant_values=0)
            label=np.pad(label,[(pw,pw),(ph,ph),(pd,pd)],mode='constant',constant_values=0)
        (w,h,d)=image.shape
        w1=int(round((w-self.output_size[0])/2.))
        h1=int(round((h-self.output_size[1])/2.))
        d1=int(round((d-self.output_size[2])/2.))
        label=label[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        image=image[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        return {'image':image,'label':label}
class RandomCrop(object):
    def __init__(self,output_size):
        self.output_size=output_size
    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        if label.shape[0]<=self.output_size[0] or label.shape[1]<=self.output_size[1] or label.shape[2]<=self.output_size[2]:
            pw=max((self.output_size[0]-label.shape[0]))
            ph=max((self.output_size[1]-label.shape[1]))
            pd=max((self.output_size[2]-label.shape[2]))
            image=np.pad(image,[(pw,pw),(ph,ph),(pd,pd)],mode='constant',constant_values=0)
            label=np.pad(label,[(pw,pw),(ph,ph),(pd,pd)],mode='constant',constant_values=0)
        (w,h,d)=image.shape
        w1=np.random.randint(0,w-self.output_size[0])
        h1=np.random.randint(0,h-self.output_size[1])
        d1=np.random.randint(0,d-self.output_size[2])
        label=label[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        image=image[w1:w1+self.output_size[0],h1:h1+self.output_size[1],d1:d1+self.output_size[2]]
        return {'image':image,'label':label}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}
class TwoStreamBatchSampler(Sampler):
    def __init__(self,primary_indices,secondary_indices,batch_size,secondary_batch_size):
        self.primary_indices=primary_indices
        self.secondary_indices=secondary_indices
        self.primary_batch_size=batch_size-secondary_batch_size
        self.secondary_batch_size=secondary_batch_size
    def __iter__(self):
        primary_iter=iterate_once(self.primary_indices)
        secondary_iter=iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch for (primary_batch,secondary_batch) in
            zip(grouper(primary_iter,self.primary_batch_size),grouper(secondary_iter,self.secondary_batch_size))
        )
    def __len__(self):
        return len(self.primary_indices)
def iterate_once(iterable):
    return np.random.permutation(iterable)
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
def grouper(iterable,n):
    args=[iter(iterable)]*n
    return zip(*args)


# if __name__ == '__main__':
#     labeled_idxs = list(range(12))
#     unlabeled_idxs = list(range(12, 60))
#     batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 4, 2)
#     for _ in range(2):
#         i = 0
#         for x in batch_sampler:
#             i += 1
#             print('%02d' % i, '\t', x)
