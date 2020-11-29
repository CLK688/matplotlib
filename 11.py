from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import torch
import torch.nn as nn
from lenet import LeNet
from dataset import Dataset
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#导入模型类、数据集类
net = LeNet()
net.load_state_dict(torch.load("./model/model.pkl"))

# #实例化模型
# my_model = MyModel.to(device)
# #加载模型
# my_model.load_state_dict(torch.load("./result/model.pth"))
def predict():
    net.eval()
    y_true_list = []
    y_pred_list = []
    predict_dataset = Dataset(train=False,batch_size=batch_size)
    for i, (data,labels) in enumerate(predict_dataset):
        with torch.no_grad():
            data = data.to(device)
            y_true = labels.numpy()
            outputs = net(data)
            y_pred = outputs.max(-1)[-1]
            y_pred = y_pred.cpu().data.numpy()
            for i in range(batch_size):
                y_pred_list.append(y_pred[i])
                y_true_list.append(y_true[i])
    return  y_true_list, y_pred_list
# def get_data():
#     dataset = 
#     for i,(data, label) in enumerate(dataset):
#         data = data.to(device)
#         y_true = label.numpy().tolist()
#         with torch.no_grad():
#             output = my_model(data) 
#         output = output.max(dim = -1)[-1]     
#         y_pred = output.cpu().data.numpy().tolist()
#     return y_true, y_pred

def plot_confusion_matrix(cm, labels_name, title):
    cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Greens)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    

    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[second_index][first_index])

y_true, y_pred = predict()

# y_true = [2, 0, 2, 2, 0, 1,7,8,3,4,5,6]
# y_pred = [0, 0, 2, 2, 1, 2,5,6,7,8,3,4]
cm = confusion_matrix(y_true, y_pred)
labels_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(cm)
plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
plt.show()