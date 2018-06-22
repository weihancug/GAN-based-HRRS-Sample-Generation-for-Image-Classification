#-*- coding:gb18030 -*-
#please  first use the dataset_split_preprocess.py to preprocess the dataset
#then check the path to match your computer ,
#finally, you can run the code
"""WGAN-GP ResNet for UC Merged 128x128 images"""
import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.plot
from UCM_preprocess import load
import numpy as np
import tensorflow as tf
import sklearn.datasets
import shutil
import time
import functools
import locale
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
locale.setlocale(locale.LC_ALL, '')
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATA_DIR = '/home/hpc-126/remote-host/UCM/train-128-20'
Log_DIR = '/home/hpc-126/remote-host/WGAN-Tensorflow-20/log'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')
nlabels=''
datatype='UCM' # there two datatype  UCM
N_GPUS = 2
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')
Imagesize = 128
BATCH_SIZE = 40 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 200000 # How many iterations to train for
DIM_G = 64 # Generator dimensionality
DIM_D = 64 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3*128*128 # Number of pixels in UCM (128*128*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score
GENERATETRAIN = False
CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss
gen_num=120 # generate samples num per class

#if MODEL_Restore != ‘’, then restore the model parameters
#MODEL_Restore = '/home/hpc-126/remote-host/WGAN-Tensorflow-20/mymodel128/mymodel128-100500'
MODEL_Restore = ''
MODEL_Save= '/home/hpc-126/remote-host/WGAN-Tensorflow-20/mymodel128/mymodel128'
validation = False
label_enum =[]
if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

if datatype =='NUPW':
    nlabels = 45
    for i in range(nlabels):
        label_enum.append(i)
        print label_enum
else:
    nlabels = 21
    label_enum = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


labels = {
  'golfcourse': 9,
  'overpass': 14,
  'freeway': 8,
  'denseresidential': 6,
  'mediumresidential': 12,
  'harbor': 10,
  'tenniscourt': 20,
  'mobilehomepark': 13,
  'parkinglot': 15,
  'agricultural': 0,
  'chaparral': 5,
  'airplane': 1,
  'river': 16,
  'baseballdiamond': 2,
  'intersection': 11,
  'beach': 3,
  'runway': 17,
  'forest': 7,
  'sparseresidential': 18,
  'buildings': 4,
  'storagetanks': 19
}


claName=[]

for k in label_enum:
    for o in labels:
        if labels[o] == int(k):
            claName.append(o)

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=nlabels)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=nlabels)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)


    return shortcut + output

# crease outputdim of conv2 *2
def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def remove_dir(path):
    try:
      shutil.rmtree(path)
    except OSError, e:
      if e.errno == 2:
        pass
      else:
        raise


def plot_confusion_matrix(cm,
                          classes,
                          label_order,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #    plt.title(title, fontsize=30)


    ylabels = []
    tick_marks = np.arange(len(label_order))
    for i in tick_marks:
         ylabels.append(classes[i].center(25) + str(label_order[i]).rjust(2))

    plt.xticks(tick_marks, label_order, rotation=90, fontsize=18)

    plt.yticks(tick_marks, ylabels, rotation=0, fontsize=18)

    plt.colorbar()
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if cm[i, j] > 0:
                plt.text(j, i, round(cm[i, j], 2),fontsize=18,
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "white")
        else:
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()




def plot_save_graph(y_test, y_pred, j,
                   claName,
                    label_order):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(20, 14))
    plot_confusion_matrix(cnf_matrix, claName,
                          label_order, normalize=True,
                          title='Confusion matrix, wioutnorimalization')
    plt.savefig("confusion_matrix/cnf_nor_gan" + str(j)+".png")





def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
   # output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 *8* DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G *8, 4, 4])
    # image is 4x4#
    output = ResidualBlock('Generator.1', DIM_G *8, DIM_G *4, 3, output, resample='up', labels=labels)
    # image is 8x8#
    output = ResidualBlock('Generator.2', DIM_G*4, DIM_G*4, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G*4, DIM_G*2, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.4', DIM_G * 2, DIM_G * 1, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.5', DIM_G * 1, DIM_G * 1, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, Imagesize, Imagesize])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D*2, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D*2, DIM_D*4, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D*4, DIM_D*4, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.5', DIM_D * 4, DIM_D * 8, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D *8, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D *8, nlabels, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
#config.gpu_options.allow_growth=True

with tf.Session(config=config) as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    #generate fake data
    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:len(DEVICES)/2]   # gpu0
    DEVICES_A = DEVICES[len(DEVICES)/2:]   #gpu1

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    # the core code  200 -300  compute the loss
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([         # all data containing fake and real data
                all_real_data_splits[i],
                all_real_data_splits[len(DEVICES_A)+i],
                fake_data_splits[i],
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
            disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))  # Wasserstein cost  of discriminator
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], labels=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(  # compute the classification accuracy of real samples
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1)),
                            real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(   #compute the classification of fake samples
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE/len(DEVICES_A):], dimension=1)),
                            real_and_fake_labels[BATCH_SIZE/len(DEVICES_A):]
                        ),
                        tf.float32
                    )
                ))


    for i, device in enumerate(DEVICES_B):    # in devices_b  compute the gradient_penalty loss
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES_A),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        validation_real=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
        validation_pred=tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1))
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
    #        fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            fake_labels = tf.cast(tf.random_uniform([n_samples]) * nlabels, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))

    #optimizer setting
    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(63, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]*3,dtype='int32'))
    fixed_noise_samples = Generator(63, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        #writer = tf.summary.FileWriter(Log_DIR, graph=tf.get_default_graph())
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((63, 3, Imagesize, Imagesize)), 'samples_{}.png'.format(frame))

#generate UCM train data
    train_labels = tf.constant(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype='int32'))
    generate_train = Generator(nlabels,train_labels)
    def generate_train_set():
        print ('start generate fake training')
        G_train = '/home/hpc-126/remote-host/UCM/generatetrain-train20'
        if os.path.exists(G_train):
            remove_dir(G_train)
        os.mkdir(G_train)
        for  i  in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            cate_path = os.path.join(G_train, str(i))
            os.mkdir(cate_path)

        for i in range(gen_num):
#            train_noise = tf.cast(np.random.normal(size=(gen_num, 128)).astype('float32'))
            train_samples = session.run(generate_train)
            train_samples = ((train_samples + 1.) * (255. / 2)).astype('int32')
            train_samples = train_samples.reshape(nlabels,3,Imagesize,Imagesize)
            # lib.save_images.save_images(train_samples,
            #                             os.path.join(cate_path, '1.jpg'))
            for count in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
                cate_path = os.path.join(G_train, str(count))
                single_sample = train_samples[count]
                single_sample = single_sample.reshape(1,3,Imagesize,Imagesize)
                name = 'sample_'+str(i)+'.jpg'
                lib.save_images.save_images(single_sample,
                                            os.path.join(cate_path,name))


    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)

    train_gen, dev_gen = load(DATA_DIR,BATCH_SIZE,Imagesize)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels

    def inf_test_gen():
        while True:
            for images,_labels in dev_gen():
                yield images,_labels

    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # restore the model parameter
    if MODEL_Restore !='':
        saver.restore(session,MODEL_Restore)
        print("Model restored : %s"%MODEL_Restore)
        print('Initialized')


    gen = inf_train_gen()



    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration})

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
#            lib.save_images.save_images(_data.reshape((64, 3, 32, 32)), 'samples_real.png')
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run([disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        # save the model
        if (iteration!=0) and (iteration % 500 ==0):
            savepath=saver.save(session, MODEL_Save, global_step=iteration)
            print ('save model to %s'%savepath)

        lib.plot.plot('cost', _disc_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan', _disc_acgan)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)

        #draw the graph of inception curve
        # if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
        #     inception_score = get_inception_score(50000)
        #     lib.plot.plot('inception_50k', inception_score[0])
        #     lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration==0 and GENERATETRAIN == True:
            generate_image(iteration, _data)
            generate_train_set()
            print ('generat train set is built')
	    exit(0)
        if validation==True:
            for i in range(1,2,1):
                dev_disc_costs = []
                dev_disc_acc = []
                total =0
                num=0
                pred=[]
                real=[]
                for images,_labels in dev_gen():
                    _dev_disc_cost,_disc_acgan_acc,_validation_real,_validation_pred = session.run([disc_cost,disc_acgan_acc,validation_real,validation_pred], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                    dev_disc_costs.append(_dev_disc_cost)
                    dev_disc_acc.append(_disc_acgan_acc)
                    _validation_pred=_validation_pred.tolist()
                    for ele in _validation_pred:
                    	pred.append(ele)
                    for ele in _validation_real:
                    	real.append(ele)
                num = len(dev_disc_acc)
                print ('validation  accuracy  is %.4f,  iteration num %d'%(np.mean(dev_disc_acc), num))
                #print ('validation_pred')
                #print (pred)
                #print ('validation_real')
                #print (real)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))
                plot_save_graph(real,pred,iteration,claName,label_enum)
        if iteration%500==0:
            generate_image(iteration, _data)


        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()


        # if iteration!=0 and iteration % 10000 == 0 and GENERATETRAIN == True:
        #     generate_train_set()
        #     print ('generat train set is built')
        lib.plot.tick()

