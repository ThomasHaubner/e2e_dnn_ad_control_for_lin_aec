#
# Thomas Haubner, LMS, 2022
#
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as Cb
import torch
import numpy as np
import libPython.class_frontend as class_frontend
import libPython.class_dataloader as class_dataloader
import os
from timeit import default_timer as timer
import libPython.class_callbacks as my_callbacks
import warnings

torch.manual_seed(42)

if __name__ == '__main__':

    print('Starting')                                               # logging data can be visualized by "tensorboard --logdir logs"
    start_time = timer()

    #######################
    # settings
    #######################
    # flags
    debugging_mode = False                                          # num_workers is set to 0 which simplifies debugging
    train_on_gpu = False                                            # process training data on GPU
    test_on_gpu = False                                             # process test data on GPU

    # testing
    test_data_path = os.path.join('data', 'test_data.h5')           # path to HDF5 test data
    test_data_save_path = os.path.join('data', 'proc_test_data')    # path which is used to save the processed test data
    eval_dic = {                                                    # evaluation settings
        'eval_erle': True,                                          # compute ERLE measure for testing data
        'eval_pesq_echo': False,                                    # compute PESQ measure (without noise) for testing data
        'save_model': True,                                         # save model to evaluation folder
        'save_wav': False                                           # save results as wav file
    }

    # data
    train_data_path = os.path.join('data', 'train_data.h5')         # path to HDF5 training data
    num_workers = int(os.cpu_count() * .5)                          # number of workers for data loader
    batch_size = 4                                                  # batch size (def.: "32" for broadband DNN and "4" for narrowband/hybrid DNN)
    train_val_ratio = .8                                            # split ratio of training-to-validation data
    check_val_every_n_epoch = 1                                     # run validation loop every "check_val_every_n_epoch" epoch
    pin_memory = True                                               # pinning memory
    shuffle_train_data = True                                       # shuffle training data
    shuffle_val_data = False                                        # shuffle evaluation data

    # stft settings
    stft_dic = {
        'nfft': 512,                                                 # DFT length
        'frameshift': 512//4,                                        # frame shift (note that quarter-frameshift shows much better performance relative to half-frameshift due to the reduced aliasing)
        'window_name': 'hamming'                                     # window name
    }

    # feature settings
    feat_dic = {
        'feat_list': ['abs_Uf', 'abs_Yf', 'abs_Ef', 'abs_Df_est', 'av_abs_Yf', 'av_abs_Ef', 'av_abs_Df_est'],     # select subset of ['abs_Uf', 'abs_Ef', 'abs_Df_est', 'abs_Yf', 'log_abs_Uf', 'log_abs_Ef', 'log_abs_Df_est', 'log_abs_Yf', 'Uf_re_im', 'Ef_re_im', 'Df_est_re_im', 'Yf_re_im', 'av_abs_Ef', 'av_abs_Yf', 'av_abs_Df_est']
        'floor_val': 10 ** (-12),                       # regularization parameter
        'nfft': stft_dic['nfft'],
    }

    # DNN settings
    dnn_dic = {
        'name': 'denseGruDenseNB',                                  # DNN inference architecture: 'denseGruDenseBB' (broadband), 'denseGruDenseNB' (narrowband/hybrid)
        'feat_dic': feat_dic,
        'nfft': stft_dic['nfft'],
    }

    if dnn_dic['name'] == 'denseGruDenseBB':
        dnn_dic['hidden_size_gru'] = 128                            # number of hidden states in GRU
        dnn_dic['num_layers_gru'] = 2                               # number of GRU layers

    elif dnn_dic['name'] == 'denseGruDenseNB':
        dnn_dic['hidden_size_gru'] = 64                             # number of hidden states in GRU
        dnn_dic['num_layers_gru'] = 2                               # number of GRU layers

    else:
        raise Exception('DNN architecture not known!')

    # AEC settings
    aec_dic = {
        'ctf_filt_len': 8,                          # CTF filter length for echo estimation
        'delta_reg_num': 1e-3,                      # regularizer in numerator of step-size (def.: 1e-3)
        'Psi_U_lambda': .9,                         # recursive averaging factor for loudspeaker power estimation (note: time constant depends on frameshift)
        'floor_val': 10 ** (-12),                   # minimum value for loudspeaker power initialization
        'est_mask_mu': 'dnnFreq',                   # estimation type for numerator mask mu of step-size: 'dnnFreq' (frequency-selective by DNN), 'dnnScal' (equivalent for all frequency bins by DNN)
        'est_mask_Ef': 'dnnFreq'                    # estimation type for denominator mask Ef of step-size: 'dnnFreq' (frequency-selective by DNN), 'dnnScal' (equivalent for all frequency bins by DNN), 'zero', 'one'
    }

    dnn_dic['aec_dic'] = aec_dic

    # training loss settings
    loss_dic = {
     'name': 'td_neg_log_echo_pow_supp_fac',         # possible choices: 'td_ms_pow_echo', 'fd_ms_pow_echo', 'td_neg_log_echo_pow_supp_fac', 'fd_neg_log_echo_pow_supp_fac'
     'floor_val': 10 ** (-12)                        # regularization parameter
    }

    # DNN optimizer settings
    max_epochs = 60                                             # maximum number of epochs
    early_stopping = True                                       # early stopping flag
    dnn_optim_dic = {
        'name': 'adam',                                         # optimizer name
        'learning_rate': 1e-3,                                  # learning rate
        'gradient_clip_val': 0.5,                               # gradient clipping
        'lr_scheduler': 'ReduceLROnPlateau',                    # learning rate scheduler
        'lr_scheduler_patience': 5,                             # number of epochs waited until learning rate is reduced
        'lr_scheduler_factor': .5                               # reduction factor for learning rate scheduler
    }

    # hardware settings
    if train_on_gpu:
        gpu_id = [0]
    else:
        gpu_id = None
    auto_select_gpus = True

    #######################
    # initialize modules
    #######################
    if debugging_mode:
        warnings.warn('Attention: Debugging mode => num_workers is set to 0 which drastically slows down training!')
        num_workers = 0

    # setting paths
    log_path = 'logs'

    # creating frontend dictionary
    frontend_dic = {
        'stft_dic': stft_dic,
        'dnn_dic': dnn_dic,
        'aec_dic': aec_dic,
        'loss_dic': loss_dic,
    }

    # initialize data loader
    data_module = class_dataloader.DataLoaderModule(train_data_path, test_data_path, num_workers, batch_size, train_val_ratio,
                                                    shuffle_train_data, shuffle_val_data, pin_memory, stft_dic['frameshift'])

    # create acoustic frontend model which contains the DNN-controlled AEC
    model = class_frontend.frontend(dnn_optim_dic, frontend_dic)

    # initialize logger
    tb_logger = pl_loggers.TensorBoardLogger(log_path)      # logging data can be evaluated by "tensorboard --logdir logs"

    # setting callbacks
    callback_list = []
    init_callback = my_callbacks.MyInitCallback()
    callback_list.append(init_callback)
    if early_stopping:
        early_stopping_cb = Cb.EarlyStopping(monitor='avg_val_loss_epoch_log', min_delta=.0, patience=20, mode='min')
        callback_list.append(early_stopping_cb)
    save_dirpath_dnn = os.path.join(tb_logger.log_dir, 'checkpoints')
    save_filename_dnn = '{epoch}-{avg_val_loss_epoch_log:.2f}-{avg_train_loss_epoch_log:.2f}'
    model_checkpoint_cb = Cb.ModelCheckpoint(save_top_k=1, save_last=True, monitor='avg_val_loss_epoch_log',
                                             dirpath=save_dirpath_dnn, filename=save_filename_dnn)
    callback_list.append(model_checkpoint_cb)
    lr_monitor_cb = Cb.LearningRateMonitor(logging_interval='epoch')
    callback_list.append((lr_monitor_cb))

    # initialize trainer
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=callback_list,
                         logger=tb_logger, precision=32, gradient_clip_val=dnn_optim_dic['gradient_clip_val'],
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gpus=gpu_id, auto_select_gpus=auto_select_gpus)

    #######################
    # training
    #######################
    trainer.fit(model, data_module)

    #######################
    # testing
    #######################
    # load model
    ckpt_path = trainer.checkpoint_callback.best_model_path
    model_best = class_frontend.frontend.load_from_checkpoint(checkpoint_path=ckpt_path)

    if test_on_gpu:
        model_best.to('cuda')
        model_best.to_device()
    else:
        model_best.to('cpu')
        model_best.to_device()

    # evaluate test data
    res_dic = model_best.eval_dataset(test_data_path, test_data_save_path, eval_dic)

    # print results to console
    if eval_dic['eval_erle'] or eval_dic['eval_pesq_echo']:
        print('\n\nResults:\t\tmean')
        if eval_dic['eval_erle']:
            print('\tERLE:\t\t' + str(np.round(res_dic['erle_mean'], 1)))
        if eval_dic['eval_pesq_echo']:
            print('\tPESQ(echo):\t' + str(np.round(res_dic['pesq_echo_mean'], 1)))

    runtime = timer() - start_time
    print('\nFinished:\tRuntime: {} min'.format(round(runtime / 60)))
