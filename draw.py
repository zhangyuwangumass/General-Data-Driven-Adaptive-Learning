import numpy as np
import matplotlib.pyplot as plt

standard = np.load('history/cifar10_1000_standard_history.npz')
reweight = np.load('history/cifar10_1000_80_reweight_history.npz')

s_train_loss = standard['train_loss']
s_valid_loss = standard['valid_loss']
s_test_loss = standard['test_loss']
s_train_acc = standard['train_acc']
s_valid_acc = standard['valid_acc']
s_test_acc = standard['test_acc']

s_f_train_loss = np.mean(s_train_loss[-11:-1])
s_f_valid_loss = np.mean(s_valid_loss[-11:-1])
s_f_test_loss = np.mean(s_test_loss[-11:-1])
s_f_train_acc = np.mean(s_train_acc[-11:-1])
s_f_valid_acc = np.mean(s_valid_acc[-11:-1])
s_f_test_acc = np.mean(s_test_acc[-11:-1])


r_train_loss = reweight['train_loss']
r_valid_loss = reweight['valid_loss']
r_test_loss = reweight['test_loss']
r_train_acc = reweight['train_acc']
r_valid_acc = reweight['valid_acc']
r_test_acc = reweight['test_acc']
weight = reweight['weight'].squeeze()

w_mean = np.mean(weight, axis=0)
print(w_mean)
w_std = np.std(weight, axis=0)
print(w_std)

r_f_train_loss = np.mean(r_train_loss[-11:-1])
r_f_valid_loss = np.mean(r_valid_loss[-11:-1])
r_f_test_loss = np.mean(r_test_loss[-11:-1])
r_f_train_acc = np.mean(r_train_acc[-11:-1])
r_f_valid_acc = np.mean(r_valid_acc[-11:-1])
r_f_test_acc = np.mean(r_test_acc[-11:-1])

print('standard final train loss:\t\t', s_f_train_loss)
print('standard final valid loss:\t\t', s_f_valid_loss)
print('standard final test loss:\t\t', s_f_test_loss)
print('standard final train acc:\t\t', s_f_train_acc)
print('standard final valid acc:\t\t', s_f_valid_acc)
print('standard final test acc:\t\t', s_f_test_acc)

print('reweighted final train loss:\t\t', r_f_train_loss)
print('reweighted final valid loss:\t\t', r_f_valid_loss)
print('reweighted final test loss:\t\t', r_f_test_loss)
print('reweighted final train acc:\t\t', r_f_train_acc)
print('reweighted final valid acc:\t\t', r_f_valid_acc)
print('reweighted final test acc:\t\t', r_f_test_acc)

'''
M = w_mean.shape[0]

x = np.arange(0, M, 1)

plt.plot(x, w_mean,color='y')
plt.plot(x, w_std, color='r')

plt.show()
'''

'''
# showing point weight

for i in range(20):
    sub_weight = weight[:, i]

    M = sub_weight.shape[0]

    plt.ylim = (0,2)

    x = np.arange(0, M, 1)

    plt.plot(x, sub_weight)

    plt.show()
'''


N = s_train_loss.shape[0]

x = np.arange(0, N, 1)

fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.plot(x, s_train_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_train_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_train_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_train_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Train Loss')

ax1 = fig.add_subplot(232)
ax1.plot(x, s_valid_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_valid_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_valid_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_valid_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Valid Loss')

ax1 = fig.add_subplot(233)
ax1.plot(x, s_test_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_test_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_test_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_test_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Test Loss')

ax1 = fig.add_subplot(234)
ax1.set_ylim([70,100])
ax1.plot(x, s_train_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_train_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_train_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_train_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Train Accuracy')

ax1 = fig.add_subplot(235)
ax1.set_ylim([70,100])
ax1.plot(x, s_valid_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_valid_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_valid_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_valid_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Valid Accuracy')

ax1 = fig.add_subplot(236)
ax1.set_ylim([70,100])
ax1.plot(x, s_test_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_test_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_test_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_test_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Test Accuracy')

plt.show()