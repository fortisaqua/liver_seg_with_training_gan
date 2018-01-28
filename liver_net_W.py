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
already_trained=181

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
        # os.makedirs(self.train_models_dir)

        if os.path.exists(self.train_sum_dir):
            # shutil.rmtree(self.train_sum_dir)
            print 'train_sum_dir: existed! \n'
        # os.makedirs(self.train_sum_dir)

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            print 'test_sum_dir: deleted and then created!\n'
        os.makedirs(self.test_sum_dir)


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

    def test(self,dicom_dir):
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        test_input_shape = input_shape
        test_input_shape[2] = 16
        test_batch_size = batch_size
        X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
                           dtype=tf.float32)
        # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        # Y = tf.placeholder(shape=[test_batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        # lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('ae'):
            Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size)

        print tools.Ops.variable_count()
        sum_merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        with tf.Session(config=config) as sess:
            if os.path.exists(self.train_models_dir):
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())
            # test_X = tf.placeholder(
            #     shape=[test_batch_size, input_shape[0], input_shape[1], input_shape[2]],
            #     dtype=tf.float32)
            # test_Y_pred, test_Y_pred_modi, test_Y_pred_nosig = self.ae_u(test_X, training,test_batch_size)
            space ,resized_array= test.get_organized_data(dicom_dir, test_input_shape)
            block_num = 0
            inputs = {}
            results = {}
            shape_resized = np.shape(resized_array)
            print "input shape: ", shape_resized
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
                if i == 0 or i == len(numbers):
                    result_final[:, :,
                    i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[2]] += 2 * np.float32(
                        (results[i][:, :, :] - 0.01) > 0)
                else:
                    result_final[:, :, i * output_shape[2] / 2:i * output_shape[2] / 2 + output_shape[2]] += np.float32(
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
            ST.WriteImage(final_img, './test_result/test_result.vtk')
            return final_img

def post_process(img):
    print img.GetSize()
    median_filter = ST.MedianImageFilter()
    median_filter.SetRadius(1)
    midian_img = median_filter.Execute(img)
    midian_array = ST.GetArrayFromImage(midian_img)
    midian_array = np.transpose(midian_array,[2,1,0])
    array_shape = np.shape(midian_array)
    seed = [0,0,0]
    max = 0
    for i in range(array_shape[0]):
        temp_max = np.sum(midian_array[i,:,:])
        if max < temp_max:
            max = temp_max
            seed[0]=i
    max = 0
    for i in range(array_shape[1]):
        temp_max = np.sum(midian_array[:,i,:])
        if max < temp_max:
            max = temp_max
            seed[1]=i
    max = 0
    for i in range(array_shape[2]):
        temp_max = np.sum(midian_array[:,:,i])
        if max < temp_max:
            max = temp_max
            seed[2]=i
    print seed
    growed_img = ST.NeighborhoodConnected(img, [seed], 0.9,1, [1, 1, 1], 1.0)

    return img,growed_img

def liver_segment(dicom_dir):
    net = Network()
    final_img = net.test(dicom_dir)
    final_img,growed_img = post_process(final_img)
    ST.WriteImage(final_img, './final_result.vtk')
    ST.WriteImage(growed_img, './growed_img.vtk')

if __name__ == "__main__":
    dicom_dir = "./3Dircadb1.2/PATIENT_DICOM"
    liver_segment(dicom_dir)
