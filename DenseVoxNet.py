import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import numpy as np
import time
import test
import SimpleITK as ST
from dicom_read import read_dicoms

resolution = 64
batch_size = 2
lr_down = [0.0005,0.0001,0.00005]
ori_lr = 0.0005
power = 0.9
GPU0 = '0'
input_shape = [512,512,16]
output_shape = [512,512,16]
type_num = 0
already_trained=212

###############################################################
config={}
config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = './Data/'+name+'/train_25d/voxel_grids_64/'
    config['Y_train_'+name] = './Data/'+name+'/train_3d/voxel_grids_64/'

config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = './Data/'+name+'/test_25d/voxel_grids_64/'
    config['Y_test_'+name] = './Data/'+name+'/test_3d/voxel_grids_64/'

config['resolution'] = resolution
config['batch_size'] = batch_size
config['meta_path'] = '/opt/analyse_liver_data/data_meta_'+str(type_num)+'.pkl'
config['data_size'] = input_shape

################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './train_models/'
        self.train_sum_dir = './train_sum/'
        self.test_results_dir = './test_results/'
        self.test_sum_dir = './test_sum/'

        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
            print 'test_results_dir: deleted and then created!\n'
        os.makedirs(self.test_results_dir)

        if os.path.exists(self.train_models_dir):
            # shutil.rmtree(self.train_models_dir)
            print 'train_models_dir: existed! will be loaded! \n'
        else:
            os.makedirs(self.train_models_dir)

        if os.path.exists(self.train_sum_dir):
            # shutil.rmtree(self.train_sum_dir)
            print 'train_sum_dir: existed! \n'
        else:
            os.makedirs(self.train_sum_dir)

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            print 'test_sum_dir: deleted and then created!\n'
        os.makedirs(self.test_sum_dir)

    def my_ae_u_1(self,X,training):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
            ##### encode
            c_e = [1, 64, 128, 256, 512]
            s_e = [0, 1, 1, 1, 1]
            layers_e = []
            layers_e.append(X)
            for i in range(1, 5, 1):
                layer = tools.Ops.conv3d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i], name='e' + str(i))
                layer = tools.Ops.conv3d(layer,k=4,out_c=c_e[i], str=s_e[i], name='e_mid1' + str(i))
                layer = tools.Ops.conv3d(layer,k=4,out_c=c_e[i], str=s_e[i], name='e_mid2' + str(i))
                layer = tools.Ops.conv3d(layer, k=2, out_c=c_e[i], str=2, name='e_down' + str(i))
                # layer = tools.Ops.maxpool3d(tools.Ops.xxlu(layer,name='lrelu'), k=2, s=2, pad='SAME')
                layer = tools.Ops.batch_norm(layer,'bn_down'+str(i),training=training)
                layers_e.append(layer)

        #     ##### fc
        #     bat, d1, d2, d3, cc = [int(d) for d in layers_e[-1].get_shape()]
        #     lfc = tf.reshape(layers_e[-1], [bat, -1])
        #     lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=5000, name='fc1'), name='relu')
        #
        # with tf.device('/gpu:'+GPU0):
        #     lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),name='relu')
        #     lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

            ##### decode
            c_d = [0, 256, 128, 64, 1]
            s_d = [0, 2, 2, 2, 2, 2]
            layers_d = []
            layers_d.append(layers_e[-1])
            for j in range(1, 5, 1):
                u_net = True
                if u_net:
                    layer = tf.concat([layers_d[-1], layers_e[-j]], axis=4)
                    layer = tools.Ops.deconv3d(layer, k=4, out_c=c_d[j], str=s_d[j], name='d' + str(len(layers_d)))
                    layer = tools.Ops.conv3d(layer, k=4, out_c=c_d[j], str=1, name='d_mid_1' + str(len(layers_d)))
                    layer = tools.Ops.conv3d(layer, k=4, out_c=c_d[j], str=1, name='d_mid_2' + str(len(layers_d)))
                    layer = tools.Ops.conv3d(layer, k=4, out_c=c_d[j], str=1, name='d_mid_3' + str(len(layers_d)))
                else:
                    layer = tools.Ops.deconv3d(layers_d[-1], k=4, out_c=c_d[j], str=s_d[j], name='d' + str(len(layers_d)))

                if j != 4:
                    layer = tools.Ops.xxlu(layer,name='relu')
                    # batch normal layer
                    # layer = tools.Ops.batch_norm(layer, 'bn_up' + str(j), training=training)
                layers_d.append(layer)

            vox_no_sig = layers_d[-1]
            # vox_no_sig = tools.Ops.xxlu(vox_no_sig,name='relu')
            vox_sig = tf.sigmoid(layers_d[-1])
            vox_sig_modified = tf.maximum(vox_sig,0.01)
        return vox_sig, vox_sig_modified,vox_no_sig

    def ae_u(self,X,training,batch_size):
        original=16
        growth=20
        dense_layer_num=12
        # input layer
        X=tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
        # image reduce layer
        conv_input_1=tools.Ops.conv3d(X,k=3,out_c=2,str=2,name='conv_input_down')
        conv_input_normed=tools.Ops.batch_norm(conv_input_1, 'bn_dense_0_0', training=training)
        # network start
        conv_input=tools.Ops.conv3d(conv_input_normed,k=3,out_c=original,str=2,name='conv_input')
        with tf.device('/gpu:'+GPU0):
            ##### dense block 1
            c_e = []
            s_e = []
            layers_e=[]
            layers_e.append(conv_input)
            for i in range(dense_layer_num):
                c_e.append(original+growth*(i+1))
                s_e.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_e[-1], 'bn_dense_1_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_e[j],name='dense_1_'+str(j))
                next_input = tf.concat([layer,layers_e[-1]],axis=4)
                layers_e.append(next_input)
            # c_e = [1]
            # s_e = [0]
            # layers_e = []
            # layers_e.append(X)
            # for i in range(1, 5, 1):
            #     layer = tools.Ops.conv3d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i], name='e' + str(i))
            #     layer = tools.Ops.conv3d(layer,k=mid_k,out_c=c_e[i], str=s_e[i], name='e_mid1' + str(i))
            #     layer = tools.Ops.conv3d(layer,k=mid_k,out_c=c_e[i], str=s_e[i], name='e_mid2' + str(i))
            #     layer = tools.Ops.conv3d(layer, k=2, out_c=c_e[i], str=2, name='e_down' + str(i))
            #     # layer = tools.Ops.maxpool3d(tools.Ops.xxlu(layer,name='lrelu'), k=2, s=2, pad='SAME')
            #     layer = tools.Ops.batch_norm(layer,'bn_down'+str(i),training=training)
            #     layers_e.append(layer)

            ##### fc
            # bat, d1, d2, d3, cc = [int(d) for d in layers_e[-1].get_shape()]
            # lfc = tf.reshape(layers_e[-1], [bat, -1])
            # lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=5000, name='fc1'), name='relu')

        # middle down sample
            mid_layer = tools.Ops.batch_norm(layers_e[-1], 'bn_mid', training=training)
            mid_layer = tools.Ops.xxlu(mid_layer,name='relu')
            mid_layer = tools.Ops.conv3d(mid_layer,k=1,out_c=original+growth*dense_layer_num,str=1,name='mid_conv')
            mid_layer_down = tools.Ops.maxpool3d(mid_layer,k=2,s=2,pad='SAME')

        ##### dense block 
        with tf.device('/gpu:'+GPU0):
            # lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),name='relu')
            # lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

            c_d = []
            s_d = []
            layers_d = []
            layers_d.append(mid_layer_down)
            for i in range(dense_layer_num):
                c_d.append(original+growth*(dense_layer_num+i+1))
                s_d.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_d[-1],'bn_dense_2_'+str(j),training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_d[j],name='dense_2_'+str(j))
                next_input = tf.concat([layer,layers_d[-1]],axis=4)
                layers_d.append(next_input)

            ##### final up-sampling
            bn_1 = tools.Ops.batch_norm(layers_d[-1],'bn_after_dense',training=training)
            relu_1 = tools.Ops.xxlu(bn_1 ,name='relu')
            conv_27 = tools.Ops.conv3d(relu_1,k=1,out_c=original+growth*dense_layer_num*2,str=1,name='conv_up_sample_1')
            deconv_1 = tools.Ops.deconv3d(conv_27,k=2,out_c=128,str=2,name='deconv_up_sample_1')
            concat_up = tf.concat([deconv_1,mid_layer],axis=4)
            deconv_2 = tools.Ops.deconv3d(concat_up,k=2,out_c=64,str=2,name='deconv_up_sample_2')

            predict_map = tools.Ops.conv3d(deconv_2,k=1,out_c=2,str=1,name='predict_map')

            # zoom in layer
            predict_map_normed = tools.Ops.batch_norm(predict_map,'bn_after_dense_1',training=training)
            predict_map_zoomed = tools.Ops.deconv3d(predict_map_normed,k=2,out_c=1,str=2,name='deconv_zoom_3')

            vox_no_sig = predict_map_zoomed
            # vox_no_sig = tools.Ops.xxlu(vox_no_sig,name='relu')
            vox_sig = tf.sigmoid(predict_map_zoomed)
            vox_sig_modified = tf.maximum(vox_sig,0.01)
        return vox_sig, vox_sig_modified,vox_no_sig

    def dis(self, X, Y,training):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
            Y = tf.reshape(Y,[batch_size,output_shape[0],output_shape[1],output_shape[2],1])
            layer = tf.concat([X,Y],axis=4)
            c_d = [1,2,64,128,256,512]
            s_d = [0,2,2,2,2,2]
            layers_d =[]
            layers_d.append(layer)
            for i in range(1,6,1):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d_1'+str(i))
                if i!=5:
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                    # batch normal layer
                    layer = tools.Ops.batch_norm(layer, 'bn_up' + str(i), training=training)
                layers_d.append(layer)
            y = tf.reshape(layers_d[-1],[batch_size,-1])
            # for j in range(len(layers_d)-1):
            #     y = tf.concat([y,tf.reshape(layers_d[j],[batch_size,-1])],axis=1)
        return tf.nn.sigmoid(y)

    def train(self, data):
        best_acc = 0
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('ae'):
            Y_pred, Y_pred_modi,Y_pred_nosig = self.ae_u(X,training,batch_size)

        with tf.variable_scope('dis'):
            XY_real_pair = self.dis(X, Y,training)
        with tf.variable_scope('dis',reuse=True):
            XY_fake_pair = self.dis(X, Y_pred,training)

        with tf.device('/gpu:'+GPU0):
            ################################ ae loss
            Y_ = tf.reshape(Y,shape=[batch_size,-1])
            Y_pred_modi_ = tf.reshape(Y_pred_modi,shape=[batch_size,-1])
            w = 0.85
            ae_loss = tf.reduce_mean( -tf.reduce_mean(w*Y_*tf.log(Y_pred_modi_ + 1e-8),reduction_indices=[1]) -
                                      tf.reduce_mean((1-w)*(1-Y_)*tf.log(1-Y_pred_modi_ + 1e-8), reduction_indices=[1]) )
            sum_ae_loss = tf.summary.scalar('ae_loss', ae_loss)

            ################################ wgan loss
            gan_g_loss = -tf.reduce_mean(XY_fake_pair)
            gan_d_loss = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
            sum_gan_g_loss = tf.summary.scalar('gan_g_loss',gan_g_loss)
            sum_gan_d_loss = tf.summary.scalar('gan_d_loss',gan_d_loss)
            alpha = tf.random_uniform(shape=[batch_size,input_shape[0]*input_shape[1]*input_shape[2]],minval=0.0,maxval=1.0)

            Y_pred_ = tf.reshape(Y_pred,shape=[batch_size,-1])
            differences_ = Y_pred_ -Y_
            interpolates = Y_ + alpha*differences_
            with tf.variable_scope('dis',reuse=True):
                XY_fake_intep = self.dis(X, interpolates,training)
            gradients = tf.gradients(XY_fake_intep,[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
            gan_d_loss +=10*gradient_penalty

            #################################  ae + gan loss
            gan_g_w = 5
            ae_w = 100-gan_g_w
            ae_gan_g_loss = ae_w * ae_loss + gan_g_w * gan_g_loss

        with tf.device('/gpu:' + GPU0):
            ae_var = [var for var in tf.trainable_variables() if var.name.startswith('ae')]
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]
            ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(ae_gan_g_loss, var_list=ae_var)
            dis_optim = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(gan_d_loss,var_list=dis_var)

        print tools.Ops.variable_count()
        sum_merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        with tf.Session(config=config) as sess:
            # if os.path.exists(self.train_models_dir):
            #     try:
            #         saver.restore(sess,self.train_models_dir+'model.cptk')
            #     except Exception,e:
            #         saver.restore(sess,'./regular/'+'model.cptk')
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(1500):
                # data.shuffle_X_Y_files(label='train')
                #### full testing
                # ...
                if epoch % 2 == 0:
                    print '********************** FULL TESTING ********************************'
                    time_begin = time.time()
                    dicom_dir = "./3Dircadb1.2/PATIENT_DICOM"
                    mask_dir = "./3Dircadb1.2/MASKS_DICOM/liver"
                    test_batch_size = batch_size
                    # test_X = tf.placeholder(
                    #     shape=[test_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                    #     dtype=tf.float32)
                    # test_Y_pred, test_Y_pred_modi, test_Y_pred_nosig = self.ae_u(test_X, training,test_batch_size)
                    space, resized_array = test.get_organized_data(dicom_dir, input_shape)
                    test_mask = read_dicoms(mask_dir)
                    array_mask = ST.GetArrayFromImage(test_mask)
                    array_mask = np.transpose(array_mask, (2, 1, 0))
                    print "mask shape: ",np.shape(array_mask)
                    time1 = time.time()
                    block_num = 0
                    inputs = {}
                    results = {}
                    shape_resized = np.shape(resized_array)
                    print "input shape: ",shape_resized
                    for i in range(0, shape_resized[2], output_shape[2] / 2):
                        if i + output_shape[2] <= shape_resized[2]:
                            inputs[block_num] = resized_array[:, :, i:i + output_shape[2]]
                        else:
                            final_block = np.zeros([output_shape[0], output_shape[1], output_shape[2]], np.float32)
                            print i, shape_resized[2]
                            final_block[:, :, :shape_resized[2] - i] = resized_array[:, :, i:shape_resized[2]]
                            inputs[block_num] = final_block[:, :, :]
                        block_num = block_num + 1
                    numbers = inputs.keys()
                    # print numbers
                    for i in range(0, len(numbers), test_batch_size):
                        if i + test_batch_size < len(numbers):
                            temp_input = np.zeros(
                                [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                            for j in range(test_batch_size):
                                temp_input[j, :, :, :] = inputs[i + j][:, :, :]
                            Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
                                                                                   feed_dict={X: temp_input,
                                                                                              training: False})
                            for j in range(test_batch_size):
                                results[i + j] = Y_temp_modi[j, :, :, :, 0]
                        else:
                            temp_batch_size = len(numbers) - i
                            temp_input = np.zeros(
                                [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                            for j in range(temp_batch_size):
                                temp_input[j, :, :, :] = inputs[i + j][:, :, :]
                            X_temp = tf.placeholder(
                                shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                                dtype=tf.float32)
                            with tf.variable_scope('ae', reuse=True):
                                Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                             temp_batch_size)
                            Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                                [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                                feed_dict={X_temp: temp_input, training: False})
                            for j in range(temp_batch_size):
                                results[i + j] = Y_temp_modi[j, :, :, :, 0]
                    # print results.keys()
                    result_final = np.zeros([shape_resized[0], shape_resized[1],
                                             len(numbers) * (output_shape[2] / 2) + output_shape[2] / 2], np.float32)
                    for i in range(0, len(numbers)):
                        if i==0 or i==len(numbers):
                            result_final[:, :, i * output_shape[2]/2:i * output_shape[2]/2 + output_shape[2]] += 2*np.float32(
                            (results[i][:, :, :] - 0.01) > 0)
                        else:
                            result_final[:, :, i * output_shape[2]/2:i * output_shape[2]/2 + output_shape[2]] += np.float32(
                            (results[i][:, :, :] - 0.01) > 0)
                        # print i * output_shape[2]/2,i * output_shape[2]/2 + output_shape[2]
                        # print i
                    final_array = np.float32(result_final >= 2)
                    final_array = final_array[:,:,0:shape_resized[2]]
                    # print np.max(final_array)
                    print "result shape: ",np.shape(final_array)
                    final_img = ST.GetImageFromArray(np.transpose(final_array, [2, 1, 0]))
                    final_img.SetSpacing(space)
                    print "writing full testing result"
                    ST.WriteImage(final_img, './test_result/test_result' + str(epoch+already_trained) + '.vtk')
                    if epoch==0:
                        mask_img = ST.GetImageFromArray(np.transpose(array_mask, [2, 1, 0]))
                        mask_img.SetSpacing(space)
                        ST.WriteImage(mask_img, './test_result/test_mask.vtk')
                    test_IOU = 2*np.sum(final_array*array_mask)/(np.sum(final_array)+np.sum(array_mask))
                    print "IOU accuracy: ",test_IOU
                    time_end = time.time()
                    print '******************** time of full testing: '+str(time_end-time_begin)+'s ********************'
                data.shuffle_X_Y_pairs()
                total_train_batch_num = data.total_train_batch_num
                # train_files=data.X_train_files
                # test_files=data.X_test_files
                # total_train_batch_num = 500
                print "total_train_batch_num:", total_train_batch_num
                for i in range(total_train_batch_num):

                    #### training
                    X_train_batch, Y_train_batch = data.load_X_Y_voxel_train_next_batch()
                    # X_train_batch, Y_train_batch = data.load_X_Y_voxel_grids_train_next_batch()
                    # Y_train_batch=np.reshape(Y_train_batch,[batch_size, output_shape[0], output_shape[1], output_shape[2], 1])
                    gan_d_loss_c, = sess.run([gan_d_loss],feed_dict={X: X_train_batch, Y: Y_train_batch,training:False})
                    ae_loss_c, gan_g_loss_c, sum_train = sess.run([ae_loss, gan_g_loss, sum_merged],feed_dict={X: X_train_batch, Y: Y_train_batch,training:False})
                    learning_rate_g = ori_lr*(1-(epoch+already_trained)/1500)**power
                    sess.run([ae_g_optim],feed_dict={X: X_train_batch, Y: Y_train_batch, lr: learning_rate_g, training: True})
                    if epoch<=5:
                        sess.run([dis_optim], feed_dict={X: X_train_batch, Y: Y_train_batch, lr: lr_down[0],training:True})
                    elif epoch<=20:
                        sess.run([dis_optim], feed_dict={X: X_train_batch, Y: Y_train_batch, lr: lr_down[1],training:True})
                    else:
                        sess.run([dis_optim], feed_dict={X: X_train_batch, Y: Y_train_batch, lr: lr_down[2],training:True})

                    sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
                    if i%2==0:
                        print "epoch:", epoch, " i:", i, " train ae loss:", ae_loss_c," gan g loss:",gan_g_loss_c," gan d loss:",gan_d_loss_c



                    #### testing
                    if i  %10== 0 and epoch %1 ==0 :
                        X_test_batch, Y_test_batch = data.load_X_Y_voxel_test_next_batch(fix_sample=False)
                        # Y_test_batch = np.reshape(Y_test_batch,[batch_size, output_shape[0], output_shape[1], output_shape[2], 1])
                        ae_loss_t,gan_g_loss_t,gan_d_loss_t, Y_test_pred,Y_test_modi, Y_test_pred_nosig= \
                            sess.run([ae_loss, gan_g_loss,gan_d_loss, Y_pred,Y_pred_modi,Y_pred_nosig],feed_dict={X: X_test_batch, Y: Y_test_batch,training:False})
                        predict_result = np.float32(Y_test_modi>0.01)
                        predict_result = np.reshape(predict_result,[batch_size,input_shape[0], input_shape[1], input_shape[2]])
                        # Foreground
                        accuracy_for = np.sum(predict_result*Y_test_batch)/np.sum(Y_test_batch)
                        # Background
                        accuracy_bac = np.sum((1-predict_result)*(1-Y_test_batch))/(np.sum(1-Y_test_batch))
                        # IOU
                        predict_probablity = np.float32(Y_test_modi-0.01)
                        predict_probablity = np.reshape(predict_probablity,[batch_size,input_shape[0], input_shape[1], input_shape[2]])
                        accuracy = 2*np.sum(np.abs(predict_probablity*Y_test_batch))/np.sum(np.abs(predict_result)+np.abs(Y_test_batch))
                        if epoch%30==0 and epoch>0:
                            to_save = {'X_test': X_test_batch, 'Y_test_pred': Y_test_pred,'Y_test_true': Y_test_batch}
                            scipy.io.savemat(self.test_results_dir + 'X_Y_pred_' + str(epoch).zfill(2) + '_' + str(i).zfill(4) + '.mat', to_save, do_compression=True)
                        print "epoch:", epoch, " i:", i, "\nacc_for: ",accuracy_for ,"\nacc_bac: ",accuracy_bac,"\nIOU accuracy: ",accuracy,"\ntest ae loss:", ae_loss_t, " gan g loss:",gan_g_loss_t," gan d loss:", gan_d_loss_t
                        if accuracy>best_acc:
                            saver.save(sess, save_path=self.train_models_dir + 'model.cptk')
                            print "epoch:", epoch, " i:", i, "best model saved!"
                            best_acc = accuracy
                    if i %50== 0 and epoch%10==0 :
                        # data.plotFromVoxels(Y_test_modi[1,:,:,:,:]-0.01,Y_test_batch[1,:,:,:])
                        # print "original"
                        # print np.max(Y_test_pred_nosig)
                        # print np.min(Y_test_pred_nosig)
                        # print "sigmoided"
                        # print np.max(Y_test_pred)
                        # print np.min(Y_test_pred)
                        print "modified"
                        print np.max(Y_test_modi[0,:,:,:,:]-0.01)
                        print np.min(Y_test_modi[0,:,:,:,:]-0.01)


                    #### model saving
                    if i %30 == 0 and epoch%1==0:
                       # regular_train_dir = "./regular/"
                       # if not os.path.exists(regular_train_dir):
                       #     os.makedirs(regular_train_dir)
                       saver.save(sess, save_path=self.train_models_dir +'model.cptk')
                       print "epoch:", epoch, " i:", i, "regular model saved!"

    def test(self, dicom_dir):
       # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
       test_input_shape = input_shape
       test_input_shape[2] = 16
       test_batch_size = batch_size
       X = tf.placeholder(
           shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
           dtype=tf.float32)
       # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
       # Y = tf.placeholder(shape=[test_batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
       # lr = tf.placeholder(tf.float32)
       training = tf.placeholder(tf.bool)
       with tf.variable_scope('ae'):
           Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size)

       print tools.Ops.variable_count()
       # sum_merged = tf.summary.merge_all()

       saver = tf.train.Saver(max_to_keep=1)
       config = tf.ConfigProto(allow_soft_placement=True)
       config.gpu_options.visible_device_list = GPU0
       with tf.Session(config=config) as sess:
           if os.path.exists(self.train_models_dir):
               saver.restore(sess, self.train_models_dir + 'model.cptk')
           # sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
           # sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

           if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
               print "restoring saved model"
               saver.restore(sess, self.train_models_dir + 'model.cptk')
           else:
               sess.run(tf.global_variables_initializer())
           # test_X = tf.placeholder(
           #     shape=[test_batch_size, input_shape[0], input_shape[1], input_shape[2]],
           #     dtype=tf.float32)
           # test_Y_pred, test_Y_pred_modi, test_Y_pred_nosig = self.ae_u(test_X, training,test_batch_size)
           space, resized_array = test.get_organized_data(dicom_dir, test_input_shape)
           block_num = 0
           inputs = {}
           results = {}
           shape_resized = np.shape(resized_array)
           print "input shape: ", shape_resized
           for i in range(0, shape_resized[2], output_shape[2] / 2):
               if i + output_shape[2] <= shape_resized[2]:
                   inputs[block_num] = resized_array[:, :, i:i + output_shape[2]]
               else:
                   final_block = np.zeros([output_shape[0], output_shape[1], output_shape[2]],
                                          np.float32)
                   print i, shape_resized[2]
                   final_block[:, :, :shape_resized[2] - i] = resized_array[:, :,
                                                              i:shape_resized[2]]
                   inputs[block_num] = final_block[:, :, :]
               block_num = block_num + 1
           numbers = inputs.keys()
           # print numbers
           for i in range(0, len(numbers), test_batch_size):
               if i + test_batch_size < len(numbers):
                   temp_input = np.zeros(
                       [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                   for j in range(test_batch_size):
                       temp_input[j, :, :, :] = inputs[i + j][:, :, :]
                   Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                       [Y_pred, Y_pred_modi, Y_pred_nosig],
                       feed_dict={X: temp_input,
                                  training: False})
                   for j in range(test_batch_size):
                       results[i + j] = Y_temp_modi[j, :, :, :, 0]
               else:
                   temp_batch_size = len(numbers) - i
                   temp_input = np.zeros(
                       [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                   for j in range(temp_batch_size):
                       temp_input[j, :, :, :] = inputs[i + j][:, :, :]
                   X_temp = tf.placeholder(
                       shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                       dtype=tf.float32)
                   with tf.variable_scope('ae', reuse=True):
                       Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp,
                                                                                    training,
                                                                                    temp_batch_size)
                   Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                       [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                       feed_dict={X_temp: temp_input, training: False})
                   for j in range(temp_batch_size):
                       results[i + j] = Y_temp_modi[j, :, :, :, 0]
           # print results.keys()
           result_final = np.zeros([shape_resized[0], shape_resized[1],
                                    len(numbers) * (output_shape[2] / 2) + output_shape[2] / 2],
                                   np.float32)
           for i in range(0, len(numbers)):
               if i == 0 or i == len(numbers):
                   result_final[:, :,
                   i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[
                       2]] += 2 * np.float32(
                       (results[i][:, :, :] - 0.01) > 0)
               else:
                   result_final[:, :,
                   i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[2]] += np.float32(
                       (results[i][:, :, :] - 0.01) > 0)
                   # print i * output_shape[2]/2,i * output_shape[2]/2 + output_shape[2]
                   # print i
           final_array = np.float32(result_final >= 2)
           final_array = final_array[:, :, 0:shape_resized[2]]
           # print np.max(final_array)
           print "result shape: ", np.shape(final_array)
           final_img = ST.GetImageFromArray(np.transpose(final_array, [2, 1, 0]))
           final_img.SetSpacing(space)
           print "writing full testing result"
           return final_img, resized_array

    # def test(self,dicom_dir):
    #     # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
    #     test_input_shape = input_shape
    #     test_input_shape[2] = 16
    #     test_batch_size = batch_size
    #     X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
    #                        dtype=tf.float32)
    #     # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
    #     # Y = tf.placeholder(shape=[test_batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
    #     # lr = tf.placeholder(tf.float32)
    #     training = tf.placeholder(tf.bool)
    #     with tf.variable_scope('ae'):
    #         Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size)
    #
    #     print tools.Ops.variable_count()
    #     sum_merged = tf.summary.merge_all()
    #
    #     saver = tf.train.Saver(max_to_keep=1)
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.visible_device_list = GPU0
    #     with tf.Session(config=config) as sess:
    #         if os.path.exists(self.train_models_dir):
    #             saver.restore(sess, self.train_models_dir + 'model.cptk')
    #         sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
    #         sum_write_test = tf.summary.FileWriter(self.test_sum_dir)
    #
    #         if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
    #             print "restoring saved model"
    #             saver.restore(sess, self.train_models_dir + 'model.cptk')
    #         else:
    #             sess.run(tf.global_variables_initializer())
    #         # test_X = tf.placeholder(
    #         #     shape=[test_batch_size, input_shape[0], input_shape[1], input_shape[2]],
    #         #     dtype=tf.float32)
    #         # test_Y_pred, test_Y_pred_modi, test_Y_pred_nosig = self.ae_u(test_X, training,test_batch_size)
    #         space ,resized_array= test.get_organized_data(dicom_dir, test_input_shape)
    #         block_num = 0
    #         inputs = {}
    #         results = {}
    #         shape_resized = np.shape(resized_array)
    #         print "input shape: ", shape_resized
    #         for i in range(0, shape_resized[2], output_shape[2] / 2):
    #             if i + output_shape[2] <= shape_resized[2]:
    #                 inputs[block_num] = resized_array[:, :, i:i + output_shape[2]]
    #             else:
    #                 final_block = np.zeros([output_shape[0], output_shape[1], output_shape[2]], np.float32)
    #                 print i, shape_resized[2]
    #                 final_block[:, :, :shape_resized[2] - i] = resized_array[:, :, i:shape_resized[2]]
    #                 inputs[block_num] = final_block[:, :, :]
    #             block_num = block_num + 1
    #         numbers = inputs.keys()
    #         # print numbers
    #         for i in range(0, len(numbers), test_batch_size):
    #             if i + test_batch_size < len(numbers):
    #                 temp_input = np.zeros(
    #                     [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
    #                 for j in range(test_batch_size):
    #                     temp_input[j, :, :, :] = inputs[i + j][:, :, :]
    #                 Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
    #                                                                        feed_dict={X: temp_input,
    #                                                                                   training: False})
    #                 for j in range(test_batch_size):
    #                     results[i + j] = Y_temp_modi[j, :, :, :, 0]
    #             else:
    #                 temp_batch_size = len(numbers) - i
    #                 temp_input = np.zeros(
    #                     [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
    #                 for j in range(temp_batch_size):
    #                     temp_input[j, :, :, :] = inputs[i + j][:, :, :]
    #                 X_temp = tf.placeholder(
    #                     shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
    #                     dtype=tf.float32)
    #                 with tf.variable_scope('ae', reuse=True):
    #                     Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
    #                                                                                  temp_batch_size)
    #                 Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
    #                     [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
    #                     feed_dict={X_temp: temp_input, training: False})
    #                 for j in range(temp_batch_size):
    #                     results[i + j] = Y_temp_modi[j, :, :, :, 0]
    #         # print results.keys()
    #         result_final = np.zeros([shape_resized[0], shape_resized[1],
    #                                  len(numbers) * (output_shape[2] / 2) + output_shape[2] / 2], np.float32)
    #         for i in range(0, len(numbers)):
    #             if i == 0 or i == len(numbers):
    #                 result_final[:, :,
    #                 i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[2]] += 2 * np.float32(
    #                     (results[i][:, :, :] - 0.01) > 0)
    #             else:
    #                 result_final[:, :, i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[2]] += np.float32(
    #                     (results[i][:, :, :] - 0.01) > 0)
    #                 # print i * output_shape[2]/2,i * output_shape[2]/2 + output_shape[2]
    #                 # print i
    #         final_array = np.float32(result_final >= 2)
    #         final_array = final_array[:, :, 0:shape_resized[2]]
    #         # print np.max(final_array)
    #         print "result shape: ", np.shape(final_array)
    #         final_img = ST.GetImageFromArray(np.transpose(final_array, [2, 1, 0]))
    #         final_img.SetSpacing(space)
    #         print "writing full testing result"
    #         ST.WriteImage(final_img, './test_result/test_result.vtk')
    #         return final_img

if __name__ == "__main__":
    dicom_dir = "./3Dircadb1.2/PATIENT_DICOM"
    data = tools.Data(config)
    net = Network()
    net.train(data)
    final_img,img_array = net.test(dicom_dir)
    ST.WriteImage(final_img,'./final_result.vtk')
