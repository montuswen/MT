import os
from sklearn.model_selection import train_test_split

def split_data():
    data_path="../../Data/processed_data_set"
    targetpath="../../Data/123dataset"
    datalist=os.listdir(data_path)
    train_ids,test_ids=train_test_split(datalist,test_size=0.2,random_state=520)
    with open(os.path.join(targetpath,"train.list"),'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(targetpath,"test.list"),'w') as f:
        f.write('\n'.join(test_ids))
if __name__ == '__main__':
    split_data()