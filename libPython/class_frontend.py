#
# Thomas Haubner, LMS, 2022
#
import torch
import pytorch_lightning as pl
import numpy as np
import libPython.class_dnn_bb as class_dnn_bb
import libPython.class_dnn_nb as class_dnn_nb
import pesq
import os
import shutil
import scipy.io.wavfile as wavfile
import libPython.class_aec_ctf as class_aec_ctf
import h5py
import warnings


class frontend(pl.LightningModule):
    """Pytorch lightning module of DNN-controlled AEC with a CTF-based echo estimation model."""

    def __init__(self, dnn_optim_dic, frontend_dic):

        super(frontend, self).__init__()

        # saving dictionaries with settings
        self.dnn_optim_dic = dnn_optim_dic
        self.frontend_dic = frontend_dic

        # stft settings
        assert (frontend_dic['stft_dic']['nfft'] % 2) == 0                  # DFT length is assumed to be even
        self.frameshift = frontend_dic['stft_dic']['frameshift']            # frameshift
        self.nfft = frontend_dic['stft_dic']['nfft']                        # DFT length
        self.nnrb = self.nfft // 2 + 1                                      # number of non-redundant frequency bins due to spectral symmetry
        self.center_stft = True
        self.window_name = frontend_dic['stft_dic']['window_name']
        if frontend_dic['stft_dic']['window_name'] == 'hamming':
            self.window = torch.hamming_window(self.nfft, periodic=True, device=self.device, requires_grad=False)
        else:
            raise Exception('The chosen window is not implemented!')

        # initializing DNN
        if self.frontend_dic['dnn_dic']['name'] == 'denseGruDenseBB':
            self.the_dnn = class_dnn_bb.DNN_denseGruDense_BB(frontend_dic['dnn_dic'])

        elif self.frontend_dic['dnn_dic']['name'] == 'denseGruDenseNB':
            self.the_dnn = class_dnn_nb.DNN_denseGruDense_NB(frontend_dic['dnn_dic'])

        else:
            raise Exception('DNN type not known!')

        # feature settings
        self.required_signals = self.the_dnn.the_features.required_signals()

        # loss function settings
        self.loss_name = frontend_dic['loss_dic']['name']
        if self.loss_name in ['td_neg_log_echo_pow_supp_fac', 'fd_neg_log_echo_pow_supp_fac']:
            self.floor_val_loss = torch.tensor(frontend_dic['loss_dic']['floor_val'])
        if self.loss_name in ['fd_neg_log_echo_pow_supp_fac', 'fd_ms_pow_echo']:
            self.stft_sig_req_for_loss = True
        else:
            self.stft_sig_req_for_loss = False

        # initializing AEC
        frontend_dic['aec_dic']['nnrb'] = self.nnrb
        self.the_aec = class_aec_ctf.DNN_supported_nlms_ctf_aec(frontend_dic['aec_dic'])
        self.aec_ctf_filt_len = self.the_aec.ctf_filt_len

        # save hyperparameters for loading model from checkpoint
        self.save_hyperparameters()

        # flags
        self.val_mode = False

    # forward run of a single sequence
    def forward(self, u_td_sig, y_td_sig):
        """Forward run of a time-domain loudspeaker (u_td_sig) and microphone (y_td_sig) signal."""

        assert u_td_sig.shape[-1] == y_td_sig.shape[-1]

        numOnlineBlocks = int(np.floor((y_td_sig.shape[-1] / self.frameshift)))
        num_samples = numOnlineBlocks * self.frameshift

        u_td_batch = torch.unsqueeze(u_td_sig[:num_samples], 0).to(self.device)          # adding dummy batch dimension
        y_td_batch = torch.unsqueeze(y_td_sig[:, :num_samples], 0).to(self.device)       # adding dummy batch dimension

        num_batch = 1                                                   # dummy batch dimension
        gru_state_init = self.the_dnn.init_hidden(num_batch)            # initial values of hidden states
        sig_dic, syst_dic = self.forward_batch(u_td_batch, y_td_batch, gru_state_init)

        return sig_dic, syst_dic

    # forward run of a batch of sequences
    def forward_batch(self, u_td_sig, y_td_sig, gru_state_init):
        """Forward run of a batch of time-domain loudspeaker (u_td_sig) and microphone (y_td_sig) signals."""
        # loudspeaker tensor:           u_td_sig = (num_batch x num_samples)
        # microphone tensor:            y_td_sig = (num_batch x num_samples)
        # internal GRU state tensor:    gru_state_init = (num_layers_gru x batch_size x hidden_size_gru)

        # set values in submodules
        self.the_dnn.my_current_epoch = self.current_epoch

        # compute STFT of signals (batch_size  x nnrb x num_blocks)
        U_stft_sig = torch.stft(u_td_sig, self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, return_complex=True)
        Y_stft_sig = torch.stft(y_td_sig, self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, return_complex=True)

        # compute static local variables
        batch_size, nnrb, num_blocks = U_stft_sig.shape

        # create loudspeaker CTF tensor
        U_ctf_stft_sig = torch.zeros((batch_size, nnrb, num_blocks, self.aec_ctf_filt_len), device=U_stft_sig.device, dtype=U_stft_sig.dtype)   # (batch_size x nnrb x num_blocks x ctf_filt_len)
        U_ctf_stft_sig[:, :, :, 0] = U_stft_sig
        for filt_ind in range(1, self.aec_ctf_filt_len):
            U_ctf_stft_sig[:, :, filt_ind:, filt_ind] = U_stft_sig[:, :, :-filt_ind]

        # create STFT tensors for processed signals
        E_stft_sig = torch.zeros_like(Y_stft_sig, device=self.device)                     # (batch_size  x nnrb x num_blocks)
        D_est_stft_sig = torch.zeros_like(Y_stft_sig, device=self.device)                 # (batch_size  x nnrb x num_blocks)

        # DNN initialization
        gru_state = gru_state_init
        self.the_aec.init_parameters(batch_size)

        # sequential processing of frames
        for idx_block in range(num_blocks):
            # assign microphone and loudspeaker signal blocks
            Y_stft_block = Y_stft_sig[:, :, idx_block]              # (batch_size x nnrb)
            U_ctf_stft_block = U_ctf_stft_sig[:, :, idx_block, :]   # (batch_size x nnrb x ctf_filt_len)

            # compute prior error after AEC
            E_stft_block, D_est_stft_block = self.the_aec.comp_prior_error(U_ctf_stft_block, Y_stft_block)

            # assign error and estiamted echo signal blocks
            E_stft_sig[:, :, idx_block] = E_stft_block
            D_est_stft_sig[:, :, idx_block] = D_est_stft_block

            # DNN inference
            sig_block_dic = {}
            if self.required_signals['Uf'] is True:
                sig_block_dic['Uf'] = U_stft_sig[:, :, idx_block]
            if self.required_signals['Yf'] is True:
                sig_block_dic['Yf'] = Y_stft_sig[:, :, idx_block]
            if self.required_signals['Ef'] is True:
                sig_block_dic['Ef'] = E_stft_sig[:, :, idx_block].clone()
            if self.required_signals['Df_est'] is True:
                sig_block_dic['Df_est'] = D_est_stft_block.clone()

            dnn_masks_aec_dic, gru_state = self.the_dnn.forward_block(sig_block_dic, gru_state)

            # update AEC parameters
            self.the_aec.update_filter(dnn_masks_aec_dic, U_ctf_stft_block, E_stft_block)

        # inverse STFT
        e_td_sig = torch.istft(E_stft_sig, self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, length=y_td_sig.shape[1])
        d_est_td_sig = torch.istft(D_est_stft_sig, self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, length=y_td_sig.shape[1])

        # write signals to dictionary
        sig_dic = {
            'e_td_sig': e_td_sig,
            'd_est_td_sig': d_est_td_sig
        }

        # add STFT signal to dictionary if is needed for loss computation
        if self.stft_sig_req_for_loss:
            sig_dic['D_est_stft_sig'] = D_est_stft_sig

        return sig_dic

    # defining optimizer
    def configure_optimizers(self):
        """Optimizer settings."""

        if self.dnn_optim_dic['name'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.dnn_optim_dic['learning_rate'])

        elif self.dnn_optim_dic['name'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.dnn_optim_dic['learning_rate'])

        else:
            raise Exception('Optimizer is not known!')

        conf_opt_dic = {
            'optimizer': optimizer,
        }

        # learning rate scheduler settings
        if self.dnn_optim_dic['lr_scheduler'] is not None:
            if self.dnn_optim_dic['lr_scheduler'] == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       patience=self.dnn_optim_dic['lr_scheduler_patience'],
                                                                       factor=self.dnn_optim_dic['lr_scheduler_factor'])
            conf_opt_dic['lr_scheduler'] = {'scheduler': scheduler,
                                            'monitor': 'avg_val_loss_epoch_log',
                                            'name': 'dnn_learning_rate'
                                            }

        return conf_opt_dic

    # training
    def training_step(self, data_batch, batch_idx):
        """Training step for a single batch."""

        # get data (batch_id x block_id x freq_id)
        u_td_batch, y_td_batch, d_td_batch = data_batch

        # forward run
        gru_state_init = self.the_dnn.init_hidden(u_td_batch.size(0))
        sig_dic = self.forward_batch(u_td_batch, y_td_batch, gru_state_init)

        # compute loss
        sig_dic['d_td_sig'] = d_td_batch
        loss_res_dic = self.compute_loss(sig_dic)

        # logging
        self.logger.experiment.add_scalar('train_loss_mini_batch', loss_res_dic['loss'], self.global_step)      # use self.logger.experiment.add_scalar to set x-value of plot

        return loss_res_dic

    def training_epoch_end(self, outputs):
        """Runs at the end of every training epoch."""

        for loss_name in outputs[0].keys():
            # compute average loss of one epoch
            avg_loss_epoch = torch.stack([x[loss_name] for x in outputs]).mean()

            # logging
            self.logger.experiment.add_scalar('avg_train_' + loss_name + '_epoch', avg_loss_epoch, self.current_epoch)  # use self.logger.experiment.add_scalar to set x-value of plot
            self.log('avg_train_' + loss_name + '_epoch_log', avg_loss_epoch, logger=False)                             # use self.log as it is needed for early stopping

        return None

    # validation
    def validation_step(self, data_batch, batch_idx):
        """Validation step for a single batch."""

        # setting validation flag
        self.val_mode = True

        # get data (batch_id x block_id x freq_id)
        u_td_batch, y_td_batch, d_td_batch = data_batch

        # forward run
        gru_state_init = self.the_dnn.init_hidden(u_td_batch.size(0))
        sig_dic = self.forward_batch(u_td_batch, y_td_batch, gru_state_init)

        # compute loss
        sig_dic['d_td_sig'] = d_td_batch
        loss_res_dic = self.compute_loss(sig_dic)

        # setting validation flag
        self.val_mode = False

        return loss_res_dic

    def validation_epoch_end(self, outputs):
        """Runs at the end of every validation epoch."""

        for loss_name in outputs[0].keys():
            # compute average loss of one epoch
            avg_loss_epoch = torch.stack([x[loss_name] for x in outputs]).mean()

            # logging
            self.logger.experiment.add_scalar('avg_val_' + loss_name + '_epoch', avg_loss_epoch, self.current_epoch)  # use self.logger.experiment.add_scalar to set x-value of plot
            if loss_name == 'val_loss':
                self.log('avg_val_' + loss_name + '_epoch_log', avg_loss_epoch, prog_bar=True, logger=False)    # use self.log as it is needed for early stopping
            else:
                self.log('avg_val_' + loss_name + '_epoch_log', avg_loss_epoch, logger=False)                   # use self.log as it is needed for early stopping

        return None

    def test_step(self, data, batch_idx):
        raise Exception('Use "eval_dataset() or eval_sig() for evaluation of a complete data set or a single sequence!')

    # loss for optimizing DNN
    def compute_loss(self, sig_dic):
        """Loss computation."""

        loss_res_dic = {}

        # compute loss
        if self.loss_name == 'td_ms_pow_echo':
            loss_res_dic['loss'] = torch.mean((sig_dic['d_td_sig'] - sig_dic['d_est_td_sig'])**2)

        elif self.loss_name == 'fd_ms_pow_echo':
            D_stft_sig = torch.stft(sig_dic['d_td_sig'], self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, return_complex=True)
            loss_res_dic['loss'] = torch.mean(torch.abs(D_stft_sig - sig_dic['D_est_stft_sig']) ** 2)

        elif self.loss_name == 'td_neg_log_echo_pow_supp_fac':
            d_td_sig = sig_dic['d_td_sig']
            d_est_td_sig = sig_dic['d_est_td_sig']
            log_pow_rat = 10 * (torch.log10(torch.mean(d_td_sig ** 2, dim=-1) + self.floor_val_loss) - torch.log10(torch.mean((d_td_sig - d_est_td_sig) ** 2, dim=-1) + self.floor_val_loss))
            loss_res_dic['loss'] = -torch.mean(log_pow_rat)

        elif self.loss_name == 'fd_neg_log_echo_pow_supp_fac':
            D_stft_sig = torch.stft(sig_dic['d_td_sig'], self.nfft, hop_length=self.frameshift, win_length=self.nfft, window=self.window, center=self.center_stft, normalized=False, onesided=True, return_complex=True)
            D_est_stft_sig = sig_dic['D_est_stft_sig']

            erle_stft = 10 * (torch.log10(torch.mean(torch.abs(D_stft_sig) ** 2, dim=(1, 2)) + self.floor_val_loss) - torch.log10(torch.mean(torch.abs(D_stft_sig - D_est_stft_sig) ** 2, dim=(1, 2)) + self.floor_val_loss))
            loss_res_dic['loss'] = -torch.mean(erle_stft)

        else:
            raise Exception('Loss type: "' + self.loss_name + '" is not known!')

        return loss_res_dic

    # evaluation
    def eval_dataset(self, test_data_path, test_data_save_path, eval_dic):
        """Evaluation of an HDF5 test data set."""
        # Note that testing is done sequentially (no mini batches) for more accurate runtime measurement

        # create directory for saving processed test data
        test_data_save_path = os.path.join(test_data_save_path, self.get_name())
        if not os.path.isdir(test_data_save_path):
            os.makedirs(test_data_save_path)
        else:
            shutil.rmtree(test_data_save_path)
            os.makedirs(test_data_save_path)

        if eval_dic['save_wav']:
            save_wav_path = os.path.join(test_data_save_path, 'wav')
            if not os.path.isdir(save_wav_path):
                os.makedirs(save_wav_path)
            else:
                shutil.rmtree(save_wav_path)
                os.makedirs(save_wav_path)

        # load test data
        hf_file = h5py.File(test_data_path, 'r')
        fs = int(np.array(hf_file.get('fs')))
        u_td_tensor = torch.from_numpy(np.array(hf_file.get('u_td_tensor'))).float()
        y_td_tensor = torch.from_numpy(np.array(hf_file.get('y_td_tensor'))).float()
        assert u_td_tensor.shape == y_td_tensor.shape

        if 'd_td_tensor' in hf_file.keys():
            d_td_tensor = np.array(hf_file.get('d_td_tensor'))
            assert y_td_tensor.shape == d_td_tensor.shape

        else:
            # performance measures can only be computed if ground-truth echo image is available
            warnings.warn('Performance measures cannot be computed because ground-truth echo is not available in test data set!')
            d_td_tensor = None
            eval_dic['eval_erle'] = False
            eval_dic['eval_pesq_echo'] = False

        if 's_td_tensor' in hf_file.keys():
            s_td_tensor = np.array(hf_file.get('s_td_tensor'))
            assert y_td_tensor.shape == s_td_tensor.shape

        else:
            # PESQ can only be computed if ground-truth speech image is available
            warnings.warn('PESQ measure cannot be computed because near-end speech image is not available in test data set!')
            s_td_tensor = None
            eval_dic['eval_pesq_echo'] = False

        hf_file.close()
        num_files = y_td_tensor.shape[0]

        # process data sequentially and compute performance measures
        e_td_tensor = np.zeros((y_td_tensor.shape[0], y_td_tensor.shape[1]))
        d_est_td_tensor = np.zeros((y_td_tensor.shape[0], y_td_tensor.shape[1]))

        for file_id in range(num_files):
            # process data
            sig = {
                'u_td_sig': u_td_tensor[file_id, :],
                'y_td_sig': y_td_tensor[file_id, :],
            }

            sig = self.eval_sig(sig)

            e_td_tensor[file_id, :] = sig['e_td_sig']
            d_est_td_tensor[file_id, :] = sig['d_est_td_sig']

            # save processed data to wav files
            if eval_dic['save_wav']:
                # compute normalization value to avoid clipping
                normalizer_val = max([np.max(np.abs(sig['y_td_sig'])), np.max(np.abs(sig['e_td_sig']))])
                if d_td_tensor is not None:
                    sig['d_td_sig'] = d_td_tensor[file_id, :]
                    normalizer_val = max([normalizer_val, np.max(np.abs(sig['d_td_sig']))])
                if s_td_tensor is not None:
                    sig['s_td_sig'] = s_td_tensor[file_id, :]
                    normalizer_val = max([normalizer_val, np.max(np.abs(sig['s_td_sig']))])

                # save wav files
                wavfile.write(os.path.join(save_wav_path, 'file_' + str(file_id) + '_mic.wav'), fs, sig['y_td_sig'] / normalizer_val)
                wavfile.write(os.path.join(save_wav_path, 'file_' + str(file_id) + '_error.wav'), fs, sig['e_td_sig'] / normalizer_val)
                if 'd_td_sig' in sig.keys():
                    wavfile.write(os.path.join(save_wav_path, 'file_' + str(file_id) + '_echo.wav'), fs, sig['d_td_sig'] / normalizer_val)
                if 's_td_sig' in sig.keys():
                    wavfile.write(os.path.join(save_wav_path, 'file_' + str(file_id) + '_nearend_speech.wav'), fs, sig['s_td_sig'] / normalizer_val)

        y_td_tensor = y_td_tensor.cpu().numpy()

        # save processed data to HDF5 files
        hf_file = h5py.File(os.path.join(test_data_save_path, 'proc_test_data.h5'), 'w')
        hf_file.create_dataset('fs', data=fs)
        hf_file.create_dataset('e_td_tensor', data=e_td_tensor)
        hf_file.create_dataset('d_est_td_tensor', data=d_est_td_tensor)
        hf_file.create_dataset('y_td_tensor', data=y_td_tensor)
        hf_file.close()

        # compute performance measures
        res_dic = {}
        if eval_dic['eval_erle']:
            res_dic['erle_vec'] = np.zeros(num_files)
        if eval_dic['eval_pesq_echo']:
            res_dic['pesq_echo_vec'] = np.zeros(num_files)

        for file_id in range(num_files):
            d_td_sig = d_td_tensor[file_id, :]
            d_est_td_sig = d_est_td_tensor[file_id, :]
            if eval_dic['eval_erle']:
                res_dic['erle_vec'][file_id] = 10 * np.log10(np.sum(d_td_sig ** 2) / np.maximum(np.sum((d_td_sig - d_est_td_sig) ** 2), 1e-30))
            if eval_dic['eval_pesq_echo']:
                s_td_sig = s_td_tensor[file_id, :]
                res_dic['pesq_echo_vec'][file_id] = pesq.pesq(fs, s_td_sig, d_td_sig - d_est_td_sig + s_td_sig, mode='wb')

        # compute mean and standard deviation of performance measures
        if eval_dic['eval_erle']:
            res_dic['erle_mean'] = np.mean(res_dic['erle_vec'])
            res_dic['erle_std'] = np.std(res_dic['erle_vec'])

        if eval_dic['eval_pesq_echo']:
            res_dic['pesq_echo_mean'] = np.mean(res_dic['pesq_echo_vec'])
            res_dic['pesq_echo_std'] = np.std(res_dic['pesq_echo_vec'])

        # write mean and standard deviation of performance measures to text file
        with open(os.path.join(test_data_save_path, 'perf_meas.txt'), 'w+') as f:
            f.write('\t\tmean\tstd\n')
            if eval_dic['eval_erle']:
                f.write('ERLE:\t\t' + str(np.round(res_dic['erle_mean'], 1)) + '\t' + str(np.round(res_dic['erle_std'], 1)) + '\n')
            if eval_dic['eval_pesq_echo']:
                f.write('PESQ(echo):\t' + str(np.round(res_dic['pesq_echo_mean'], 1)) + '\t' + str(np.round(res_dic['pesq_echo_std'], 1)) + '\n')

        # saving model
        if eval_dic['save_model']:
            torch.save(self, os.path.join(test_data_save_path, 'saved_model.pth'))

        return res_dic

    def eval_sig(self, sig):
        """Evaluation of a single signal dictionary for testing."""

        self.eval()

        with torch.no_grad():
            assert sig['u_td_sig'].shape == sig['y_td_sig'].shape

            # adding dummy batch dimension at the beginning
            u_td_sig = torch.unsqueeze(sig['u_td_sig'], dim=0).to(self.device)
            y_td_sig = torch.unsqueeze(sig['y_td_sig'], dim=0).to(self.device)

            # inference
            gru_state_init = self.the_dnn.init_hidden(u_td_sig.shape[0])            # initial values of hidden GRU states
            sig_eval = self.forward_batch(u_td_sig, y_td_sig, gru_state_init)

            # save estimated signal to signal dictionary and transform to numpy array
            sig['e_td_sig'] = (torch.squeeze(sig_eval['e_td_sig'], dim=0)).cpu().numpy()
            sig['d_est_td_sig'] = (torch.squeeze(sig_eval['d_est_td_sig'], dim=0)).cpu().numpy()
            sig['y_td_sig'] = sig['y_td_sig'].cpu().numpy()

        return sig

    def get_name(self):
        """Returning name according to parameter settings."""

        name = 'dnn_lin_aec' + '_stft_(' + str(self.frontend_dic['stft_dic']['nfft']) + ',' + str(self.frontend_dic['stft_dic']['frameshift']) + ')_' + \
               self.the_dnn.name + '_feat_(' + '-'.join(self.frontend_dic['dnn_dic']['feat_dic']['feat_list']) + ')' + \
               '_aec_(' + 'filtLen_' + str(self.frontend_dic['aec_dic']['ctf_filt_len']) + \
               '_muMask_' + str(self.frontend_dic['aec_dic']['est_mask_mu']) + '_EfMask_' + str(self.frontend_dic['aec_dic']['est_mask_Ef']) + ')' +\
               '_loss_(' + self.frontend_dic['loss_dic']['name'] + ')'

        return name

    def to_device(self):
        """Moving data to cpu or gpu."""
        if hasattr(self, 'window'):
            self.window = self.window.to(self.device)
