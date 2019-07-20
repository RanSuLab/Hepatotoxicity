import pandas as pd
import os
import math
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import csv
import keras

def upper_sample(data, label):
    ind1 = []
    ind0 = []
    for i in range(len(label)):
        if (label[i] == 0):
            ind0.append(i)
        if (label[i] == 1):
            ind1.append(i)
    ind1 = pd.DataFrame(ind1)
    ind0 = pd.DataFrame(ind0)

    if (len(ind1) > len(ind0)):
        pic_0 = ind0.sample(n=len(ind1), replace=True)
        pic_0 = pic_0.values
        pic_0 = pic_0[:, 0]
        ind1 = ind1.values
        ind1 = ind1[:, 0]
        data_0 = data[pic_0]
        label_0 = label[pic_0]
        data_1 = data[ind1]
        label_1 = label[ind1]
        return data_0, data_1, label_0, label_1
    if (len(ind1) < len(ind0)):
        pic_1 = ind1.sample(n=len(ind0), replace=True)
        pic_1 = pic_1.values
        pic_1 = pic_1[:, 0]
        ind0 = ind0.values
        ind0 = ind0[:, 0]
        data_1 = data[pic_1]
        label_1 = label[pic_1]
        data_0 = data[ind0]
        label_0 = label[ind0]
        return data_0, data_1, label_0, label_1
def trainTest_split(data_path,label_path):
    data=[]
    label=pd.read_csv(open(label_path))
    label=label.values
    label=label[:,0]
    i=0
    for doc in os.listdir(data_path):
        doc_path=data_path+doc
        doc=pd.read_csv(open(doc_path))
        doc=doc.values
        data.append(doc)
        i=i+1
        print(i,'\n')
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    os.makedirs(data_path+'train/data')
    os.makedirs(data_path+'train/label')
    train_data_path=data_path+'train/data/'
    train_label_path=data_path+'train/label/'
    a=1
    y_train=pd.DataFrame(y_train)
    labelPath=train_label_path+'label.csv'
    y_train.to_csv(labelPath)
    #fb=open(train_label_path+'label.csv','w')

    for i in X_train:
        #fb=open(train_data_path+a+'.csv','w')
        i=pd.DataFrame(i)
        i=i.to_csv(train_data_path+str(a)+'.csv')
        a=a+1
    os.makedirs(data_path+'test/data')
    os.makedirs(data_path+'test/label')
    test_data_path=data_path+'test/data/'
    test_label_path=data_path+'test/label/'
    #fb=open(test_label_path+'label.csv','w')
    y_test=pd.DataFrame(y_test)
    labelPath=test_label_path+'label.csv'
    y_test.to_csv(labelPath)
    b=1
    for i in X_test:
        #fb=open(test_data_path+b+'.csv','w')
        i=pd.DataFrame(i)
        i.to_csv(test_data_path+str(b)+'.csv')
        b=b+1
def trainTestVal_split(data_path,label_path,save_path):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut
    data=[]
    loo = LeaveOneOut()
    label=pd.read_csv(open(label_path))
    label=label.values
    label=label[:,1]
    i=0
    for doc in os.listdir(data_path):
        doc_path=data_path+'/'+doc
        doc=pd.read_csv(open(doc_path))
        doc=doc.values
        #doc=doc[1:len(doc[:,0]),1:len(doc[0,:])]
        data.append(doc)
        i=i+1
        print(i,'\n')
    data=np.array(data)
    #X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    for train, test in loo.split(data):
        a=test[0]
        lo_path=save_path+'/'+str(a)+'lo'
        os.makedirs(lo_path)
        x_test = data[test]
        y_test = label[test]
        x_train = data[train]
        y_train = label[train]
        x_test=pd.DataFrame(x_test[0])
        y_test=pd.DataFrame(y_test)
        test_path=lo_path+'/test'
        os.makedirs(test_path)
        os.makedirs(test_path+'/data')
        os.makedirs(test_path+'/label')
        x_test.to_csv(test_path+'/data/data.csv')
        y_test.to_csv(test_path+'/label/label.csv')
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        train_path=lo_path+'/train'
        val_path=lo_path+'/val'
        os.makedirs(train_path)
        os.makedirs(val_path)
        train_label_path=train_path+'/label'
        train_data_path=train_path+'/data'
        val_label_path=val_path+'/label'
        val_data_path=val_path+'/data'
        os.makedirs(train_data_path)
        os.makedirs(train_label_path)
        os.makedirs(val_label_path)
        os.makedirs(val_data_path)
        a, b, c, d = upper_sample(x_train_train, y_train_train)
        x_train_train = np.vstack((a, b))
        y_train_train = np.append(c, d)
        y_train_train=pd.DataFrame(y_train_train)
        y_train_test=pd.DataFrame(y_train_test)
        y_train_train.to_csv(train_label_path+'/label.csv')
        y_train_test.to_csv(val_label_path+'/label.csv')
        for i in range(len(x_train_train)):
            tmp=x_train_train[i]
            tmp=pd.DataFrame(tmp)
            tmp.to_csv(train_data_path+'/'+str(i)+'.csv')
        for i in range(len(x_train_test)):
            tmp=x_train_test[i]
            tmp=pd.DataFrame(tmp)
            tmp.to_csv(val_data_path+'/'+str(i)+'.csv')





def batchCsvToPicture(data_path,size):
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    for i in range(0,size):

        lo_path=data_path+'/'+str(i)+'lo'
        train_data_path=lo_path+'/'+'train/data/'
        val_data_path=lo_path+'/'+'val/data/'
        test_data_path=lo_path+'/'+'test/data/'
        train_pic_path=lo_path+'/'+'train/jpg'
        val_pic_path=lo_path+'/'+'val/jpg'
        test_pic_path=lo_path+'/'+'test/jpg'
        os.makedirs(train_pic_path)
        os.makedirs(val_pic_path)
        os.makedirs(test_pic_path)
        csvToPicture(train_data_path,train_pic_path+'/')
        csvToPicture(val_data_path,val_pic_path+'/')
        csvToPicture(test_data_path,test_pic_path+'/')
	print(i,'finished\n')
def batchSplit(data_path,size):
    for i in range(0,size):
        lo_path = data_path + '/' + str(i) + 'lo'
        train_data_path = lo_path + '/' + 'train/data/'
        val_data_path = lo_path + '/' + 'val/data/'
        test_data_path = lo_path + '/' + 'test/data/'
        train_pic_path = lo_path + '/' + 'train/jpg/'
        val_pic_path = lo_path + '/' + 'val/jpg/'
        test_pic_path = lo_path + '/' + 'test/jpg/'
        train_label_path=lo_path+'/'+'train/label/label.csv'
        test_label_path=lo_path+'/'+'test/label/label.csv'
        val_label_path=lo_path+'/'+'val/label/label.csv'
        split_by_labels(train_label_path,train_pic_path)
        split_by_labels(test_label_path,test_pic_path)
        split_by_labels(val_label_path,val_pic_path)
# csvToPicture
def csvToPicture(csv_path,new_path):
    i = 1
    for filename in os.listdir(csv_path):
        filepath = csv_path+filename
        csv_data = pd.read_csv(filepath)
        csv_data = np.array(csv_data)
        csv_data=csv_data[1:,:]
        image = Image.fromarray(csv_data)
        image = image.convert('L')
        image_name = str(i)+'.jpg'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        image_path = new_path+image_name
        image.save(image_path)
        i = i+1
def batchCrop(data_path,new_data_path,size,batch):
    for i in range(0,size):
        lo_path=data_path+'/'+str(i)+'lo/'
        new_lo_path=new_data_path+'/'+str(i)+'lo/'
        os.makedirs(new_data_path+'/'+str(i)+'lo')
        crop_image2(lo_path,new_lo_path,batch)


def split_by_labels(label_path,picture_path):

    label = pd.read_csv(label_path)

    for i,item in label.iterrows():
        img_name = str(i+1)+'.jpg'
        label = str(item[1])

        img_path = picture_path+img_name
        image = Image.open(img_path)
        new_path = picture_path+label

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_img_path = new_path+'/'+img_name
        image.save(new_img_path)

def crop_image(path,new_path):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name
        for second_path_name in os.listdir(first_path):
            second_path = first_path+'/'+second_path_name
            new_image_path_last = new_path+first_path_name+'/'+second_path_name
            if not os.path.exists(new_image_path_last):
                os.makedirs(new_image_path_last)
            for name in os.listdir(second_path):
                image_name = str(name.split('.')[0])
                image_path = second_path+'/'+name
                image = Image.open(image_path)
                for i in range(int(image.size[0]/327)):
                    img = image.crop((327*i,0,i*327+327,image.size[1]+1))
                    new_image_name = image_name+'_'+str(i+1)+'.jpg'
                    new_image_path = new_image_path_last+'/'+new_image_name
                    img.save(new_image_path)
def crop_image2(path,new_path,size):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name
        second_path_0=first_path+'/jpg/0'
        second_path_1=first_path+'/jpg/1'
        print(second_path_1)
            #second_path = first_path+'/'+second_path_name
        new_image_path_0 = new_path+first_path_name+'/0'
        new_image_path_1=new_path+first_path_name+'/1'
        if  os.path.exists(second_path_0):
            os.makedirs(new_image_path_0)
        if  os.path.exists(second_path_1):
            os.makedirs(new_image_path_1)
        if os.path.exists(second_path_0):
            for name in os.listdir(second_path_0):
                print(name)
                image_name = str(name.split('.')[0])
                image_path = second_path_0+'/'+name
                image = Image.open(image_path)
                for i in range(int(image.size[0]/size)):
                    img = image.crop((size*i,0,i*size+size,image.size[1]+1))
                    new_image_name = image_name+'_'+str(i+1)+'.jpg'
                    new_image_path = new_image_path_0+'/'+new_image_name
                    img.save(new_image_path)
        if os.path.exists(second_path_1):
            for name in os.listdir(second_path_1):
                image_name = str(name.split('.')[0])
                image_path = second_path_1+'/'+name
                image = Image.open(image_path)
                for i in range(int(image.size[0]/size)):
                    img = image.crop((size*i,0,i*size+size,image.size[1]+1))
                    new_image_name = image_name+'_'+str(i+1)+'.jpg'
                    new_image_path = new_image_path_1+'/'+new_image_name
                    img.save(new_image_path)
def train_random_crop_tes8t_cut(path, new_path):
    for first_path_name in os.listdir(path):
        first_path = path + first_path_name

        length = 585
        crop_size = 400
        if first_path_name == 'train':
            for second_path_name in os.listdir(first_path):
                second_path = first_path+'/'+second_path_name
                new_second_path = new_path + first_path_name + '/' + second_path_name
                if not os.path.exists(new_second_path):
                    os.makedirs(new_second_path)
                number = 50
                randint_value = length - crop_size
                for name in os.listdir(second_path):
                    image_name = name.split('.')[0]
                    image_path = second_path + '/' + name
                    image = Image.open(image_path)
                    for i in range(number):
                        a = random.randint(0, randint_value)
                        b = random.randint(0, randint_value)
                        img = image.crop((a, b, a + crop_size, b + crop_size))
                        img = img.resize((224, 224))
                        img_path = new_second_path + '/' + image_name + '_' + str(i + 1) + '.jpg'
                        img.save(img_path)
        else:
            for second_path_name in os.listdir(first_path):
                second_path = first_path+'/'+second_path_name
                new_second_path = new_path + first_path_name + '/' + second_path_name
                if not os.path.exists(new_second_path):
                    os.makedirs(new_second_path)
                for name in os.listdir(second_path):
                    image_name = str(name.split('.')[0])
                    image_path = second_path + '/' + name
                    image = Image.open(image_path)
                    for i in range(int(math.ceil(length / crop_size))):
                        a = i * crop_size
                        b = 0
                        for j in range(25):
                            c = a + crop_size
                            d = b + crop_size
                            if c >= length:
                                a = length - 1 - crop_size
                                c = a + crop_size
                            if d >= length:
                                b = length - 1 - crop_size
                                d = b + crop_size
                            # print(a, b, c, d)
                            img = image.crop((b, a, d, c))
                            img = img.resize((224, 224))
                            new_image_name = image_name + '_' + str(i + 1) + '_' + str(j + 1) + '.jpg'
                            new_image_path = new_second_path+'/' + new_image_name
                            img.save(new_image_path)
                            b += 7

def data_aug(path,new_path):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name

        if first_path_name == 'train':
            for second_path_name in os.listdir(first_path):
                second_path = first_path+'/'+second_path_name
                new_second_path = new_path + first_path_name + '/' + second_path_name
                if not os.path.exists(new_second_path):
                    os.makedirs(new_second_path)
                if second_path_name == '0':

                    for name in os.listdir(second_path):
                        image_name = name.split('.')[0]
                        image_path = second_path+'/'+name
                        image = Image.open(image_path)

                        new_image_path = new_second_path + '/' + image_name + '.jpg'
                        image.save(new_image_path)

                        # flip_lr
                        imageFlipLR = image.transpose(Image.FLIP_LEFT_RIGHT)
                        imageFlipLRPath = new_second_path + '/' + image_name + '_lr.jpg'
                        imageFlipLR.save(imageFlipLRPath)

                        # rotate 30
                        image_rotate_30 = image.rotate(30)
                        new_image_30_path = new_second_path + '/' + image_name + '_rotate30.jpg'
                        image_rotate_30.save(new_image_30_path)

                        # rotate 45
                        image_rotate_45 = image.rotate(45)
                        new_image_45_path = new_second_path + '/' + image_name + '_rotate45.jpg'
                        image_rotate_45.save(new_image_45_path)
                else:
                    for name in os.listdir(second_path):
                        image_name = name.split('.')[0]
                        image_path = second_path + '/' + name
                        image = Image.open(image_path)

                        new_image_path = new_second_path + '/' + image_name + '.jpg'
                        image.save(new_image_path)

                        # flip_lr
                        imageFlipLR = image.transpose(Image.FLIP_LEFT_RIGHT)
                        imageFlipLRPath = new_second_path + '/' + image_name + '_lr.jpg'
                        imageFlipLR.save(imageFlipLRPath)

                        # flip_tb
                        imageFlipTB = image.transpose(Image.FLIP_TOP_BOTTOM)
                        imageFlipTBPath = new_second_path + '/' + image_name + '_tb.jpg'
                        imageFlipTB.save(imageFlipTBPath)

                        # rotate 30
                        image_rotate_30 = image.rotate(30)
                        new_image_30_path = new_second_path + '/' + image_name + '_rotate30.jpg'
                        image_rotate_30.save(new_image_30_path)

                        # rotate 45
                        image_rotate_45 = image.rotate(45)
                        new_image_45_path = new_second_path + '/' + image_name + '_rotate45.jpg'
                        image_rotate_45.save(new_image_45_path)

                        # rotate 135
                        image_rotate_135 = image.rotate(135)
                        new_image_135_path = new_second_path + '/' + image_name + '_rotate135.jpg'
                        image_rotate_135.save(new_image_135_path)


        else:
            for second_path_name in os.listdir(first_path):
                second_path = first_path+'/'+second_path_name
                new_second_path = new_path + first_path_name + '/' + second_path_name
                if not os.path.exists(new_second_path):
                    os.makedirs(new_second_path)
                    for name in os.listdir(second_path):
                        image_name = name.split('.')[0]
                        image_path = second_path + '/' + name
                        image = Image.open(image_path)

                        new_image_path = new_second_path + '/' + image_name + '.jpg'
                        image.save(new_image_path)

def rotate(path,new_path):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name

        for second_path_name in os.listdir(first_path):
            second_path = first_path+'/'+second_path_name
            new_second_path = new_path + first_path_name + '/' + second_path_name
            if not os.path.exists(new_second_path):
                os.makedirs(new_second_path)
            if second_path_name == '0':

                for name in os.listdir(second_path):
                    image_name = name.split('.')[0]
                    image_path = second_path+'/'+name
                    image = Image.open(image_path)
                    image = image.resize((224,224))

                    new_image_path = new_second_path + '/' + image_name + '.jpg'
                    image.save(new_image_path)

                    # rotate 30
                    image_rotate_30 = image.rotate(30)
                    new_image_30_path = new_second_path + '/' + image_name + '_rotate30.jpg'
                    image_rotate_30.save(new_image_30_path)

                    # rotate 45
                    image_rotate_45 = image.rotate(45)
                    new_image_45_path = new_second_path + '/' + image_name + '_rotate45.jpg'
                    image_rotate_45.save(new_image_45_path)

                    # rotate 90
                    image_rotate_90 = image.rotate(90)
                    new_image_90_path = new_second_path + '/' + image_name + '_rotate90.jpg'
                    image_rotate_90.save(new_image_90_path)

                    # rotate 135
                    image_rotate_135 = image.rotate(135)
                    new_image_135_path = new_second_path + '/' + image_name + '_rotate135.jpg'
                    image_rotate_135.save(new_image_135_path)

                    # rotate 270
                    image_rotate_270 = image.rotate(270)
                    new_image_270_path = new_second_path + '/' + image_name + '_rotate270.jpg'
                    image_rotate_270.save(new_image_270_path)
            else:
                for name in os.listdir(second_path):
                    image_name = name.split('.')[0]
                    image_path = second_path + '/' + name
                    image = Image.open(image_path)
                    image = image.resize((224,224))

                    new_image_path = new_second_path + '/' + image_name + '.jpg'
                    image.save(new_image_path)

                    # rotate 15
                    image_rotate_15 = image.rotate(15)
                    new_image_15_path = new_second_path + '/' + image_name + '_rotate15.jpg'
                    image_rotate_15.save(new_image_15_path)

                    # rotate 30
                    image_rotate_30 = image.rotate(30)
                    new_image_30_path = new_second_path + '/' + image_name + '_rotate30.jpg'
                    image_rotate_30.save(new_image_30_path)

                    # rotate 45
                    image_rotate_45 = image.rotate(45)
                    new_image_45_path = new_second_path + '/' + image_name + '_rotate45.jpg'
                    image_rotate_45.save(new_image_45_path)

                    # rotate 90
                    image_rotate_90 = image.rotate(90)
                    new_image_90_path = new_second_path + '/' + image_name + '_rotate90.jpg'
                    image_rotate_90.save(new_image_90_path)

                    # rotate 135
                    image_rotate_135 = image.rotate(135)
                    new_image_135_path = new_second_path + '/' + image_name + '_rotate135.jpg'
                    image_rotate_135.save(new_image_135_path)

                    # rotate 180
                    image_rotate_180 = image.rotate(180)
                    new_image_180_path = new_second_path + '/' + image_name + '_rotate180.jpg'
                    image_rotate_180.save(new_image_180_path)

                    # rotate 235
                    image_rotate_235 = image.rotate(235)
                    new_image_235_path = new_second_path + '/' + image_name + '_rotate235.jpg'
                    image_rotate_235.save(new_image_235_path)

                    # rotate 270
                    image_rotate_270 = image.rotate(270)
                    new_image_270_path = new_second_path + '/' + image_name + '_rotate270.jpg'
                    image_rotate_270.save(new_image_270_path)

                    # rotate 315
                    image_rotate_315 = image.rotate(315)
                    new_image_315_path = new_second_path + '/' + image_name + '_rotate315.jpg'
                    image_rotate_315.save(new_image_315_path)

def flip(path,new_path):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name

        for second_path_name in os.listdir(first_path):
            second_path = first_path+'/'+second_path_name
            new_second_path = new_path + first_path_name + '/' + second_path_name
            if not os.path.exists(new_second_path):
                os.makedirs(new_second_path)
            for name in os.listdir(second_path):
                image_name = name.split('.')[0]
                image_path = second_path+'/'+name
                image = Image.open(image_path)

                new_image_path = new_second_path + '/' + image_name + '.jpg'
                image.save(new_image_path)

                # flip_lr
                imageFlipLR = image.transpose(Image.FLIP_LEFT_RIGHT)
                imageFlipLRPath = new_second_path + '/' + image_name + '_lr.jpg'
                imageFlipLR.save(imageFlipLRPath)

                # flip_tb
                imageFlipTB = image.transpose(Image.FLIP_TOP_BOTTOM)
                imageFlipTBPath = new_second_path + '/' + image_name + '_tb.jpg'
                imageFlipTB.save(imageFlipTBPath)

def resize_all(path,new_path,size):
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name
        new_first_path = new_path+first_path_name
        if not os.path.exists(new_first_path):
            os.makedirs(new_first_path)
        for image_name in os.listdir(first_path):
            image_path = first_path+'/'+image_name
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((size,size))

            new_image_path = new_first_path+'/'+image_name
            image.save(new_image_path)
def batchResize(data_path,new_data_path,pics,size):
    for i in range(0,pics):
        lo_path = data_path + '/' + str(i) + 'lo/'
        new_lo_path = new_data_path + '/' + str(i) + 'lo/'
        os.makedirs(new_data_path + '/' + str(i) + 'lo')
        for name in os.listdir(lo_path):
            new_tmp_path=new_lo_path+'/'+name
            os.makedirs(new_tmp_path)
            tmp=lo_path+'/'+name
            resize_all(tmp+'/',new_tmp_path+'/',size)








trainTestVal_split("./dm_invivo/pic","./label_dm_invivo.csv","./dm_invivo/cnn/save")
batchCsvToPicture("./dm_invivo/cnn/save",125)

batchCrop("./dm_invivo/cnn/save","./dm_invivo/cnn/crop",125,36)
batchResize("./dm_invivo/cnn/crop","./dm_invivo/cnn/resize/alexnet",125,227)
batchResize("./dm_invivo/rat_invivo/cnn/crop","./dm_invivo/rat_invivo/cnn/resize/resnet",125,224)
batchResize("./dm_invivo/cnn/crop","./dm_invivo/cnn/resize/googlenet",125,224)
batchResize("./dm_invivo/rat_invivo/cnn/crop",".dm_invivo/rat_invivo/cnn/resize/googlenet",125,28)

