import numpy as np
import matplotlib.pyplot as plt

standard = np.load('simple_history.npz')
reweight = np.load('reweight_history.npz')

s_loss = standard['train_loss']
s_acc = standard['train_acc']

r_loss = reweight['train_loss']
r_acc = reweight['train_acc']

N = len(s_loss)


train_idx = []
test_idx = []

for i in range(N):
    if i % (486) < 391:
        train_idx.append(i)
    else:
        test_idx.append(i)

print(len(train_idx))
print(len(test_idx))

s_train_loss = s_loss[train_idx]
s_test_loss = s_loss[test_idx]

r_train_loss = r_loss[train_idx]
r_test_loss = r_loss[test_idx]

s_train_acc = s_acc[train_idx]
s_test_acc = s_acc[test_idx]

r_train_acc = r_acc[train_idx]
r_test_acc = r_acc[test_idx]

print('best test accuracy for standard:\t\t', s_test_acc.max())
print('best test accuracy for reweight:\t\t', r_test_acc.max())

P = s_train_loss.shape[0]
Q = s_test_loss.shape[0]

x1 = np.arange(0, P, 1)
x2 = np.arange(0, Q, 1)

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(x1, s_train_loss, ls=':', c='b', label='Standard')
ax1.plot(x1, r_train_loss, ls=':', c='r', label='Reweighted')
ax1.legend()

ax1.set_title('Train Loss')

ax1 = fig.add_subplot(222)
ax1.plot(x2, s_test_loss, ls=':', c='b', label='Standard')
ax1.plot(x2, r_test_loss, ls=':', c='r', label='Reweighted')
ax1.legend()

ax1.set_title('Test Loss')

ax1 = fig.add_subplot(223)
ax1.set_ylim([60,100])
ax1.plot(x1, s_train_acc, ls=':', c='b', label='Standard')
ax1.plot(x1, r_train_acc, ls=':', c='r', label='Reweighted')
ax1.legend()

ax1.set_title('Train Accuracy')

ax1 = fig.add_subplot(224)
ax1.set_ylim([60,100])
ax1.plot(x2, s_test_acc, ls=':', c='b', label='Standard')
ax1.plot(x2, r_test_acc, ls=':', c='r', label='Reweighted')
ax1.legend()

ax1.set_title('Test Accuracy')

plt.show()