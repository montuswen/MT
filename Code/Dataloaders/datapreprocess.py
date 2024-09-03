import os
from tqdm import tqdm
import nrrd
import numpy as np
import h5py
#进行数据扩增
data_size = [112, 112, 80]
orin_datapath = '../../Data/dataset/'
final_datapath = '../../Data/processed_data_set/'


def datacovert():
    datalist = os.listdir(orin_datapath)  # 得到一个文件列表
    for item in tqdm(datalist):  # 显示进度条
        image, header = nrrd.read(os.path.join(orin_datapath, item, 'lgemri.nrrd'))
        label, header = nrrd.read(os.path.join(orin_datapath, item, 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)  # 被分割出来的左心房对应为1，背景对应为0
        w, h, d = label.shape#图像边界大小
        index = np.nonzero((label))  # 返回非0值的索引
        minx, maxx = np.min(index[0]), np.max(index[0])
        miny, maxy = np.min(index[1]), np.max(index[1])
        minz, maxz = np.min(index[2]), np.max(index[2])#防止把有用信息裁剪掉
        px=max(data_size[0]-(maxx-minx),0)
        py=max(data_size[1]-(maxy-miny),0)
        pz=max(data_size[2]-(maxz-minz),0)#计算多余尺寸
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)#图像随机增强的尺寸
        image=(image-np.mean(image))/np.std(image)#图像标准化
        image=image.astype(np.float32)
        image=image[minx:maxx,miny:maxy,minz:maxz]
        label=label[minx:maxx,miny:maxy,minz:maxz]
        item_path=os.path.join(final_datapath,item)#目标输出路径
        os.mkdir(item_path)
        f=h5py.File(os.path.join(item_path, 'mri_norm2.h5'),'w')
        f.create_dataset('image',data=image,compression="gzip")
        f.create_dataset("label",data=label,compression="gzip")
        f.close()#导出为h5文件


if __name__ == '__main__':
    datacovert()
