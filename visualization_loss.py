import os
import matplotlib.pyplot as plt

train_loss = []
train_precision = []
train_recall = []
val_loss = []
val_precision = []
val_recall = []

train_axis = []
val_axis = []


with open('logs\\tuned_SuperPoint\log.txt') as f:
    lines = f.readlines()
    for line in lines:
        if 'train - loss :' in line:
            train_loss.append(float(line[16:]))
        if 'val - loss :' in line:
            val_loss.append(float(line[14:]))
        if 'train - precision :' in line:
            train_precision.append(float(line[21:]))
        if 'val - precision :' in line:
            val_precision.append(float(line[19:]))
        if 'train - recall :' in line:
            train_recall.append(float(line[18:]))
        if 'val - recall :' in line:
            val_recall.append(float(line[16:]))

for i in range(0,len(train_loss)):
    train_axis.append(90000+400*i)
for i in range(0,len(val_loss)):
    val_axis.append(90000+286*i)

plt.subplot(311)
plt.plot(train_axis,train_loss,label="train_loss")
plt.plot(val_axis,val_loss,label="val_loss")
plt.ylabel('loss')
plt.grid()
leg0 = plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(train_axis,train_precision,label="train_precision")
plt.plot(val_axis,val_precision,label="val_precision")
plt.ylabel('precision')
plt.grid()
leg1 = plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(train_axis,train_recall,label="train_recall")
plt.plot(val_axis,val_recall,label="val_recall")
plt.ylabel('recall')
leg2 = plt.legend(loc='upper right')
plt.grid()
plt.show()