import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
# 101 avg log
x_thre = [0.8, 0.9, 1.0, 1.1]
x_sel = np.array([8000, 7906, 4005, 4]) / 8000
y_left_tau = [0.289,  0.275, 0.0338, 0]
y_right_acc = [0.905, 0.905756, 0.9127, 0.902]
# 101 log minmax
x_thre = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]
x_sel = np.array([7992  , 7925  , 6828 ,5060 , 3020 , 1505  ]) / 8000
y_left_tau = [0.287, 0.278,  0.19469, 0.0937, 0.025, 0.237]
y_right_acc = [0.905, 0.9056, 0.909, 0.9116, 0.9127, 0.912]
# 201 avg log cifar10
x_thre = [0.7, 0.8, 0.9, 1.0]
x_sel = np.array([7913 , 7819 , 7206 , 4999 ]) / 8000
y_left_tau = [0.58,  0.5709,  0.518, 0.367]
y_right_acc = [0.8425, 0.8484, 0.86129,  0.8774]
# 201 log minmax
x_thre = [0.1, 0.5, 0.7, 0.8, 0.9]
x_sel = np.array([7868, 7171, 4934 , 2861 , 2861 ]) / 8000
y_left_tau = [ 0.5759,   0.514,0.358, 0.2316, 0.146]
y_right_acc = [0.84676,  0.862, 0.8777, 0.885, 0.88986]


# 201 log mmx 100
x_thre = [0.2, 0.4, 0.7, 0.9, 0.95]
x_sel = np.array([7377 , 7206 , 4866 , 683 , 202 ]) / 8000
y_left_tau = [ 0.566,   0.5588,0.4051, 0.2049, 0.029]
y_right_acc = [0.638,  0.640667, 0.667288, 0.6886, 0.695]



# 101 sum  mmx
x_thre = [0.1, 0.7, 1.5, 2.5]
x_sel = np.array([8000 , 5738 , 1882 , 175 ]) / 8000
y_left_tau = [ 0.293,   0.235,0.346, 0.098]
y_right_acc = [0.905,  0.911,  0.910, 0.925]
# 201 sum mmx cifar10
x_thre = [0.1, 0.5, 0.7, 0.8, 0.9]
x_sel = np.array([7619 , 5135 , 1441  , 383  , 52  ]) / 8000
y_left_tau = [ 0.5448,    0.3349,0.286, 0.259, 0.573]
y_right_acc = [0.85,  0.86, 0.87, 0.87, 0.889]
# 201 sum mmx cifar100
x_thre = [0, 0.2, 0.5, 0.7, 0.9]
x_sel = np.array([7913, 6291, 1901, 553, 26]) / 8000
y_left_tau = [ 0.534,    0.44,0.265,  0.194, 0.702]
y_right_acc = [0.63,  0.64, 0.658, 0.672, 0.68]



# 101 sum log avg
x_thre = [0.4, 0.8, 1.2, 1.6]
x_sel = np.array([8000 , 5738 , 1882 , 175 ]) / 8000
y_left_tau = [ 0.293,   0.235,0.346, 0.098]
y_right_acc = [0.905,  0.911,  0.910, 0.925]
# 201 sum log mmx cifar10
x_thre = [0, 0.3, 0.6, 0.8, 0.9]
x_sel = np.array([7913  , 7334  , 3551   , 383  , 52  ]) / 8000
y_left_tau = [  0.57,     0.524,0.293, 0.259, 0.573]
y_right_acc = [0.84,  0.856, 0.881, 0.891, 0.893]
# sum 第一二张图数据填错了，注意
# 201 sum log  mmx cifar100
x_thre = [0, 0.2, 0.4, 0.8, 0.9]
x_sel = np.array([7913, 7515 , 5997 , 381 , 52 ]) / 8000
y_left_tau = [ 0.60,     0.569,0.4859,  0.227, 0.539]
y_right_acc = [0.62,  0.63, 0.654, 0.691, 0.697]



# sauc
# 101 10
x_thre = [0.58, 0.6, 0.62, 0.64, 0.66]
x_sel = np.array([8000 , 7879  , 6971  , 4468 , 1244  ]) / 8000
y_left_tau = [ -0.05,-0.05, -0.07, -0.089,  -0.128]
y_right_acc = [ 0.905,   0.905, 0.905,  0.905, 0.90]
# 201 10
x_thre = [0.58, 0.6, 0.62, 0.64, 0.66]
x_sel = np.array([7913  , 7905   , 7861   , 7333  , 3000   ]) / 8000
y_left_tau = [ -0.267,-0.267, -0.268, -0.244,  -0.073]
y_right_acc = [  84.25,   84.248, 84.23,  83.9, -0.073]
# 201 100
x_thre = [0.87, 0.88, 0.89, 0.9]
x_sel = np.array([7913 , 7901 , 4050 , 105 ]) / 8000
y_left_tau = [ 0.3044,0.304, 0.132, -0.0297]
y_right_acc = [62.01667,62.02,64.6776, 67.45]

thre_sel = []
for i in range(len(x_thre)):
    thre_sel.append(str(x_thre[i]) + '/' + str(x_sel[i]))
x = range(len(thre_sel))


plt.gcf().set_facecolor(np.ones(3)*240/255)
fig, ax1 = plb.subplots()
ax1.plot(x, y_left_tau, c='orangered', label='tau', linewidth=1)
plt.xticks(x, thre_sel)
plt.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(x, y_right_acc, c='blue', label='accuracy', linewidth=1)
plt.xticks(x, thre_sel)
plt.legend(loc=4)

plt.grid(True)
ax1.set_title("threshold, selectivity, tau, accuracy", size=18)
ax1.set_xlabel('threshold/selectivity', size=16)
ax1.set_ylabel('tau', size=16)
ax2.set_ylabel('accuracy', size=16)

plt.show()

