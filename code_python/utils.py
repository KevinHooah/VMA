'''
Coded by Kevin Hu.
Contact: www.kevin-hu.com
'''
import tensorflow as tf
import numpy as np
import scipy.io as sio
import numpy as np

'''
These are for VMA paper code.
'''
def model_evaluate(model, acc_metric, tf_dataset, conf_mat):
    '''
    To evaluate the model and get the prediction confidence matrix, 
    You cannot directly call model.eval() since we dont't model.compile().
    
    Input:
        model: the trained model object
        acc_metric: the performance metric you defined
        tf_dataset: the tensorflow.data.Dataset object for the test data.
                    Remeber to set batch size = 1 for generate the confidence matrix.
        conf_mat: empty numpy matrix with a dimention of M x N. M is the number of data
                  sampels in the test data. N is the number of classes.
    Output:
        prediction performance (scalar), prediction confidence matrix (matrix)
    '''
    acc_metric.reset_states()
    layer = tf.keras.layers.Activation('softmax')
    for idx, (x_batch, y_batch) in enumerate(tf_dataset):
            logits = model(x_batch, training=False)
            output = layer(logits)
            pred_logit = output.numpy()
            conf_mat[idx] = pred_logit
            acc_metric.update_state(y_batch , logits)
    acc = acc_metric.result()
    return acc, conf_mat


def select_by_conf(conf_array, pred_array, label, threshold):
    '''
    To econduct the confidence-based pseudo-labeled data selection.
    
    Input:
        conf_array: 1-D numpy array with dimention of M, M is the number of pseudo-labeled data
        pred_array: 1-D numpu array of pseudo-labels
        label: the class you are conducting selection
        threshold: The threshold to keep the data. E.g., 0.8 means only top 20% will be kept.
                   Please be notified, it is not a very strict selection, please refer to 
                   official numpy.quantile() documentation for more information.
    Output:
        index of selected rows of the prediction array.
    '''
    idx1 = np.where(pred_array == label)[0]
    conf_of_label = np.squeeze(conf_array[idx1])
    rk_threshold = np.quantile(conf_of_label, threshold)
    idx2 = np.where(conf_of_label >= rk_threshold)[0]
    selected_rows = idx1[idx2]
    return selected_rows

'''
Followings are for future direction code.
'''

def read_mat(mat_dir):
    '''
    Read the .mat file into a numpy matrix.

    Input:
        mat_dir: directory to the .mat file
    Output:
        the numpy matrix of the corresponding .mat matrix.

    '''
    mat = sio.loadmat(mat_dir)
    np_matrix = mat['data']

    np_matrix[:,0] -= 1 #make the label start from 0

    return np_matrix

def prepare_tf_data(data_matrix, win_length, batch_size):
    '''
    Convert the data_matrix into tf.data.Dataset instance.
    Input:
        data_matrix: the numpy matrix of raw data.
        win_length: the sliding window length.
        batch_size: the batch size for training the model.
    Output:
        IMU Tensor and Vibration Tensor.
    '''

    imu_tensor = tf.keras.utils.timeseries_dataset_from_array(
    data_matrix[:,5:],
    data_matrix[:,0],
    sequence_length = win_length,
    sequence_stride = win_length/2,
    sampling_rate = 1,
    batch_size = batch_size,
    shuffle = False,
    seed = None,
    start_index = None,
    end_index = None)

    vib_tensor = tf.keras.utils.timeseries_dataset_from_array(
    data_matrix[:,1:5],
    data_matrix[:,0],
    sequence_length = win_length,
    sequence_stride = win_length/2,
    sampling_rate = 1,
    batch_size = batch_size,
    shuffle = False,
    seed = None,
    start_index = None,
    end_index = None)

    return imu_tensor, vib_tensor

def prepare_np_data(tf_data, window_size, sensor_size):
    '''
    Convert the tensorflow.data.Dataset object into numpy tensor for futher pseudo-labeled data selection.
    '''
    num_samples = tf_data.cardinality().numpy()
    np_iter= tf_data.as_numpy_iterator()
    X = np.empty([num_samples, window_size, sensor_size])
    y = np.empty(num_samples)
    for idx, batch in enumerate(np_iter):
        inputs, targets = batch
        X[idx] = inputs
        y[idx] = targets
    return (X, y)
