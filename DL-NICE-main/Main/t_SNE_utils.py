import os
import pickle
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir='../Data/1/121_Original/'

path_train_0 = dir+'C0_test.pkl'

with open(path_train_0,'rb') as f0:
        x_train_0 = pickle.load(f0)[0]
x_train_0 = x_train_0.reshape(300,3072)
y0=np.ones([300])*0
x_train_0 ,y0= (x_train_0,y0)

path_train_1 = dir+'C1_test.pkl'

with open(path_train_1,'rb') as f1:
        x_train_1 = pickle.load(f1)[0]
x_train_1 = x_train_1.reshape(300,3072)
y1=np.ones([300])*1

x_train_1, y1= (x_train_1, y1)

path_train_2 = dir+'C2_test.pkl'

with open(path_train_2,'rb') as f2:
        x_train_2 = pickle.load(f2)[0]
x_train_2 = x_train_2.reshape(300,3072)
y2=np.ones([300])*2
x_train_2, y2= (x_train_2, y2)


path_train_3 = dir+'C3_test.pkl'
with open(path_train_3,'rb') as f3:
        x_train_3= pickle.load(f3)[0]
x_train_3 = x_train_3.reshape(300,3072)
y3=np.ones([300])*3
x_train_3, y3= (x_train_3, y3)
# print(x_train_3, y3)


x_train = np.vstack([x_train_0, x_train_1, x_train_2, x_train_3])
y = np.hstack([y0, y1, y2, y3]).astype(np.int64)


print(x_train.shape, y.shape)

#visiualize
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sn.color_palette("hls", 7))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit. 我们为每个数字加上标签。
    txts = []
    for i in range(7):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
sn.set_style('whitegrid')
sn.set_palette('muted')
sn.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

digits_proj = TSNE().fit_transform(x_train)
scatter(digits_proj, y)
plt.show()