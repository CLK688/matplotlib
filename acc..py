
import numpy as np
import pickle
import matplotlib.pyplot as plt

#测试集准确率
with open('C:/Users/11151/Desktop/123/data/testacclistFP.pkl','rb') as f:
    datatestacclistFP= pickle.load(f)
with open('C:/Users/11151/Desktop/123/data/testacclistFP2.pkl','rb') as f2:
    datatestacclistFP2= pickle.load(f2)
with open('C:/Users/11151/Desktop/123/data/testacclistFPre.pkl','rb') as f3:
    datatestacclistFPre= pickle.load(f3)
with open('C:/Users/11151/Desktop/123/data/testacclistFPre50.pkl','rb') as f4:
    datatestacclistFPre50= pickle.load(f4)

#训练集准确率
with open('C:/Users/11151/Desktop/123/data/trainacclistFP.pkl','rb') as f5:
    datatrainacclistFP= pickle.load(f5)
with open('C:/Users/11151/Desktop/123/data/trainacclistFP2.pkl','rb') as f6:
    datatrainacclistFP2= pickle.load(f6)
with open('C:/Users/11151/Desktop/123/data/trainacclistFPre.pkl','rb') as f7:
    datatrainacclistFPre= pickle.load(f7)
with open('C:/Users/11151/Desktop/123/data/trainacclistFPre50.pkl','rb') as f8:
    datatrainacclistFPre50= pickle.load(f8)

#训练集损失函数
with open('C:/Users/11151/Desktop/123/data/trainlosslistFP.pkl','rb') as f:
    datatrainlosslistFP= pickle.load(f)
with open('C:/Users/11151/Desktop/123/data/trainlosslistFP2.pkl','rb') as f2:
    datatrainlosslistFP2= pickle.load(f2)
with open('C:/Users/11151/Desktop/123/data/trainlosslistFPre.pkl','rb') as f3:
    datatrainlosslistFPre= pickle.load(f3)
with open('C:/Users/11151/Desktop/123/data/trainlosslistFPre50.pkl','rb') as f4:
    datatrainlosslistFPre50= pickle.load(f4)

#测试集损失函数
with open('C:/Users/11151/Desktop/123/data/testlosslistFP.pkl','rb') as f5:
    datatestlosslistFP= pickle.load(f5)
with open('C:/Users/11151/Desktop/123/data/testlosslistFP2.pkl','rb') as f6:
    datatestlosslistFP2= pickle.load(f6)
with open('C:/Users/11151/Desktop/123/data/testlosslistFPre.pkl','rb') as f7:
    datatestlosslistFPre= pickle.load(f7)
with open('C:/Users/11151/Desktop/123/data/testlosslistFPre50.pkl','rb') as f8:
    datatestlosslistFPre50= pickle.load(f8)


x1 = range(0,len(datatestacclistFP))
x2 = range(0,len(datatestacclistFP2))
x3 = range(0,len(datatestacclistFPre))
x4 = range(0,len(datatestacclistFPre50))

x5 = range(0,len(datatrainacclistFP))
x6 = range(0,len(datatrainacclistFP2))
x7 = range(0,len(datatrainacclistFPre))
x8 = range(0,len(datatrainacclistFPre50))

x9 = range(0,len(datatestlosslistFP))
x10 = range(0,len(datatestlosslistFP2))
x11 = range(0,len(datatestlosslistFPre))
x12 = range(0,len(datatestlosslistFPre50))

x13 = range(0,len(datatrainlosslistFP))
x14 = range(0,len(datatrainlosslistFP2))
x15 = range(0,len(datatrainlosslistFPre))
x16 = range(0,len(datatrainlosslistFPre50))

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

fig = plt.figure(figsize=(15,8))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# 绘制曲线test_acc
plt.subplot(2,2,1)
plt.plot(x1, datatestacclistFP, label = "SEI-ResNet",c="green")
plt.plot(x2, datatestacclistFP2, label = "Changed AlexNet",c="red")
plt.plot(x3, datatestacclistFPre, label = "ResNet34",c="blue")
plt.plot(x4, datatestacclistFPre50, label = "ResNet50",c="tomato")
plt.plot([x1[-1],x1[-1]],[datatestacclistFP[-1],0],ls="dotted",c="green")
plt.plot([x2[-1],x2[-1]],[datatestacclistFP2[-1],0],ls="dotted",c="red")
plt.plot([x3[-1],x3[-1]],[datatestacclistFPre[-1],0],ls="dotted",c="blue")
plt.plot([x4[-1],x4[-1]],[datatestacclistFPre50[-1],0],ls="dotted",c="tomato")
plt.title("Validataion accurary",fontsize = 13)
plt.xlabel("Epoch number")
plt.ylabel("accuracy")
plt.ylim(0,1)
plt.legend(bbox_to_anchor = (1.04,0),loc = 3, borderaxespad = 0,prop=font1)

#绘制曲线train_acc
plt.subplot(2,2,2)
plt.plot(x5, datatrainacclistFP, label = "SEI-ResNet",c="green")
plt.plot(x6, datatrainacclistFP2, label = "Changed AlexNet",c="red")
plt.plot(x7, datatrainacclistFPre, label = "ResNet34",c="blue")
plt.plot(x8, datatrainacclistFPre50, label = "ResNet50",c="tomato")
plt.plot([x5[-1],x5[-1]],[datatrainacclistFP[-1],0],ls="dotted",c="green")
plt.plot([x6[-1],x6[-1]],[datatrainacclistFP2[-1],0],ls="dotted",c="red")
plt.plot([x7[-1],x7[-1]],[datatrainacclistFPre[-1],0],ls="dotted",c="blue")
plt.plot([x8[-1],x8[-1]],[datatrainacclistFPre50[-1],0],ls="dotted",c="tomato")
plt.title("Train accurary",fontsize = 13)
plt.xlabel("Epoch number")
plt.ylabel("accuracy")
plt.ylim(0,1)
plt.legend(bbox_to_anchor = (1.04,0),loc = 3, borderaxespad = 0,prop=font1)

#绘制曲线test_loss
plt.subplot(2,2,3)
plt.plot(x9, datatestlosslistFP, label = "SEI-ResNet",c="green")
plt.plot(x10, datatestlosslistFP2, label = "Changed AlexNet",c="red")
plt.plot(x11, datatestlosslistFPre, label = "ResNet34",c="blue")
plt.plot(x12, datatestlosslistFPre50, label = "ResNet50",c="tomato")
# plt.plot([x9[-1],x9[-1]],[datatestlosslistFP[-1],0],ls="dotted",c="green")
# plt.plot([x10[-1],x10[-1]],[datatestlosslistFP2[-1],0],ls="dotted",c="red")
# plt.plot([x11[-1],x11[-1]],[datatestlosslistFPre[-1],0],ls="dotted",c="blue")
# plt.plot([x12[-1],x12[-1]],[datatestlosslistFPre50[-1],0],ls="dotted",c="tomato")
plt.title("Validataion loss",fontsize = 13)
plt.xlabel("Epoch number")
plt.ylabel("loss")
plt.ylim(0,1)
plt.legend(bbox_to_anchor = (1.04,0),loc = 3, borderaxespad = 0,prop=font1)

#绘制曲线train_loss
plt.subplot(2,2,4)
plt.plot(x13, datatrainlosslistFP, label = "SEI-ResNet",c="green")
plt.plot(x14, datatrainlosslistFP2, label = "Changed AlexNet",c="red")
plt.plot(x15, datatrainlosslistFPre, label = "ResNet34",c="blue")
plt.plot(x16, datatrainlosslistFPre50, label = "ResNet50",c="tomato")
# plt.plot([x13[-1],x13[-1]],[datatrainlosslistFP[-1],0],ls="dotted",c="green")
# plt.plot([x14[-1],x14[-1]],[datatrainlosslistFP2[-1],0],ls="dotted",c="red")
# plt.plot([x15[-1],x15[-1]],[datatrainlosslistFPre[-1],0],ls="dotted",c="blue")
# plt.plot([x16[-1],x16[-1]],[datatrainlosslistFPre50[-1],0],ls="dotted",c="tomato")
plt.title("Train loss",fontsize = 13)
plt.xlabel("Epoch number")
plt.ylabel("loss")
plt.ylim(0,1)
plt.legend(bbox_to_anchor = (1.04,0),loc = 3, borderaxespad = 0,prop=font1)
fig.tight_layout()

plt.savefig("train_loss.jpg",bbox_inches = 'tight')
plt.show()



