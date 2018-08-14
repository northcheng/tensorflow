Others(WIP)

"""
Dropout
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

# training data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
y = x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# test data
test_x = x.copy()
test_y = test_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# show data
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# tf placeholders
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# overfitting net
o1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# dropout net
d1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about plotting

for t in range(500):
    sess.run([o_train, d_train], {tf_x: x, tf_y: y, tf_is_training: True})  # train, set is_training=True

    if t % 10 == 0:
        # plotting
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [o_loss, d_loss, o_out, d_out], {tf_x: test_x, tf_y: test_y, tf_is_training: False} # test, set is_training=False
        )
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train'); plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x, o_out_, 'r-', lw=3, label='overfitting'); plt.plot(test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color':  'red'}); plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5)); plt.pause(0.1)

plt.ioff()
plt.show()



"""
Batch Normalization
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = tf.nn.tanh
B_INIT = tf.constant_initializer(-0.2)      # use a bad bias initialization

# training data
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
np.random.shuffle(x)
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise
train_data = np.hstack((x, y))

# test data
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

# plot input data
plt.scatter(x, y, c='#FF9359', s=50, alpha=0.5, label='train')
plt.legend(loc='upper left')

# tensorflow placeholder
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_train = tf.placeholder(tf.bool, None)     # flag for using BN on training or testing


class NN(object):
    def __init__(self, batch_normalization=False):
        self.is_bn = batch_normalization

        self.w_init = tf.random_normal_initializer(0., .1)  # weights initialization
        self.pre_activation = [tf_x]
        if self.is_bn:
            self.layer_input = [tf.layers.batch_normalization(tf_x, training=tf_is_train)]  # for input data
        else:
            self.layer_input = [tf_x]
        for i in range(N_HIDDEN):  # adding hidden layers
            self.layer_input.append(self.add_layer(self.layer_input[-1], 10, ac=ACTIVATION))
        self.out = tf.layers.dense(self.layer_input[-1], 1, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.loss = tf.losses.mean_squared_error(tf_y, self.out)

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_layer(self, x, out_size, ac=None):
        x = tf.layers.dense(x, out_size, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.pre_activation.append(x)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        if self.is_bn: x = tf.layers.batch_normalization(x, momentum=0.4, training=tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

nets = [NN(batch_normalization=False), NN(batch_normalization=True)]    # two nets, with and without BN

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# plot layer input distribution
f, axs = plt.subplots(4, N_HIDDEN+1, figsize=(10, 5))
plt.ion()   # something about plotting

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax,  ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0: p_range = (-7, 10); the_range = (-7, 10)
        else: p_range = (-4, 4); the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(()); a.set_xticks(())
        ax_pa_bn.set_xticks(p_range); ax_bn.set_xticks(the_range); axs[2, 0].set_ylabel('Act'); axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)

losses = [[], []]   # record test loss
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    np.random.shuffle(train_data)
    step = 0
    in_epoch = True
    while in_epoch:
        b_s, b_f = (step*BATCH_SIZE) % len(train_data), ((step+1)*BATCH_SIZE) % len(train_data) # batch index
        step += 1
        if b_f < b_s:
            b_f = len(train_data)
            in_epoch = False
        b_x, b_y = train_data[b_s: b_f, 0:1], train_data[b_s: b_f, 1:2]         # batch training data
        sess.run([nets[0].train, nets[1].train], {tf_x: b_x, tf_y: b_y, tf_is_train: True})     # train

        if step == 1:
            l0, l1, l_in, l_in_bn, pa, pa_bn = sess.run(
                [nets[0].loss, nets[1].loss, nets[0].layer_input, nets[1].layer_input,
                 nets[0].pre_activation, nets[1].pre_activation],
                {tf_x: test_x, tf_y: test_y, tf_is_train: False})
            [loss.append(l) for loss, l in zip(losses, [l0, l1])]   # recode test loss
            plot_histogram(l_in, l_in_bn, pa, pa_bn)     # plot histogram

plt.ioff()

# plot test loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
plt.ylabel('test loss'); plt.ylim((0, 2000)); plt.legend(loc='best')

# plot prediction line
pred, pred_bn = sess.run([nets[0].out, nets[1].out], {tf_x: test_x, tf_is_train: False})
plt.figure(3)
plt.plot(test_x, pred, c='#FF9359', lw=4, label='Original')
plt.plot(test_x, pred_bn, c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(x[:200], y[:200], c='r', s=50, alpha=0.2, label='train')
plt.legend(loc='best'); plt.show()



"""
Visualize Gradient Descent
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR = 0.1
REAL_PARAMS = [1.2, 2.5]
INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][2]

x = np.linspace(-1, 1, 200, dtype=np.float32)   # x data

# Test (1): Visualize a simple linear function with two parameters,
# you can change LR to 1 to see the different pattern in gradient descent.

# y_fun = lambda a, b: a * x + b
# tf_y_fun = lambda a, b: a * x + b


# Test (2): Using Tensorflow as a calibrating tool for empirical formula like following.

# y_fun = lambda a, b: a * x**3 + b * x**2
# tf_y_fun = lambda a, b: a * x**3 + b * x**2


# Test (3): Most simplest two parameters and two layers Neural Net, and their local & global minimum,
# you can try different INIT_PARAMS set to visualize the gradient descent.

y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise         # target

# tensorflow graph
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a, b)
mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_); b_list.append(b_); cost_list.append(mse_)    # record parameter changes
        result, _ = sess.run([pred, train_op])                          # training


# visualization codes:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')    # plot data
plt.plot(x, result, 'r-', lw=2)   # plot line fitting
# 3D cost figure
fig = plt.figure(2); ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
ax.set_xlabel('a'); ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
plt.show()



"""
Distributed Training
"""

import tensorflow as tf
import multiprocessing as mp
import numpy as np
import os, shutil


TRAINING = True

# training data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise


def work(job_name, task_index, step, lock):
    # set work's ip:port, parameter server and worker are the same steps
    cluster = tf.train.ClusterSpec({
        "ps": ['localhost:2221', ],
        "worker": ['localhost:2222', 'localhost:2223', 'localhost:2224',]
    })
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        # join parameter server
        print('Start Parameter Server: ', task_index)
        server.join()
    else:
        print('Start Worker: ', task_index, 'pid: ', mp.current_process().pid)
        # worker job
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            # build network
            tf_x = tf.placeholder(tf.float32, x.shape)
            tf_y = tf.placeholder(tf.float32, y.shape)
            l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
            output = tf.layers.dense(l1, 1)
            loss = tf.losses.mean_squared_error(tf_y, output)
            global_step = tf.train.get_or_create_global_step()
            train_op = tf.train.GradientDescentOptimizer(
                learning_rate=0.001).minimize(loss, global_step=global_step)

        # set training steps
        hooks = [tf.train.StopAtStepHook(last_step=100000)]

        # get session
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               checkpoint_dir='./tmp',
                                               hooks=hooks) as mon_sess:
            print("Start Worker Session: ", task_index)
            while not mon_sess.should_stop():
                # train
                _, loss_ = mon_sess.run([train_op, loss], {tf_x: x, tf_y: y})
                with lock:
                    step.value += 1
                if step.value % 500 == 0:
                    print("Task: ", task_index, "| Step: ", step.value, "| Loss: ", loss_)
        print('Worker Done: ', task_index)


def parallel_train():
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    # use multiprocessing to create a local cluster with 2 parameter servers and 4 workers
    jobs = [('ps', 0), ('worker', 0), ('worker', 1), ('worker', 2)]
    step = mp.Value('i', 0)
    lock = mp.Lock()
    ps = [mp.Process(target=work, args=(j, i, step, lock), ) for j, i in jobs]
    [p.start() for p in ps]
    [p.join() for p in ps]


def eval():
    tf_x = tf.placeholder(tf.float32, [None, 1])
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
    output = tf.layers.dense(l1, 1)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./tmp'))
    result = sess.run(output, {tf_x: x})
    # plot
    import matplotlib.pyplot as plt
    plt.scatter(x.ravel(), y, c='b')
    plt.plot(x.ravel(), result.ravel(), c='r')
    plt.show()


if __name__ == "__main__":
    if TRAINING:
        parallel_train()
    else:
        eval()