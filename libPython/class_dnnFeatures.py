#
# Thomas Haubner, LMS, 2022
#
import torch
import pytorch_lightning as pl


class DnnFeature(pl.LightningModule):
    """Features class for DNN inference."""

    def __init__(self, feat_dic):

        super(DnnFeature, self).__init__()

        # settings
        self.nnrb = feat_dic['nfft'] // 2 + 1
        self.floor_val = torch.tensor(feat_dic['floor_val'])

        # feature list
        self.feat_list = feat_dic['feat_list']
        self.num_feat_maps = 0
        for feat_descr in self.feat_list:
            assert feat_descr in ['abs_Uf', 'abs_Ef', 'abs_Df_est', 'abs_Yf',
                                  'log_abs_Uf', 'log_abs_Ef', 'log_abs_Df_est', 'log_abs_Yf',
                                  'Uf_re_im', 'Ef_re_im', 'Df_est_re_im', 'Yf_re_im', 'av_abs_Ef', 'av_abs_Yf', 'av_abs_Df_est']    # available feature list
            if feat_descr[-5:] == 're_im':
                self.num_feat_maps += 2
            else:
                self.num_feat_maps += 1

        # compute feature size
        self.feat_len_freq = self.nnrb

        self.feat_size = self.num_feat_maps * self.feat_len_freq
        self.feat_size_nb = self.num_feat_maps
        assert self.feat_size > 0

        # feature normalization
        # mean and the standard deviation of the features are estimated online by recursive averaging during training
        self.my_current_epoch = 0
        # list of recursive averaging values per epoch
        self.rec_av_fac_feat_norm_epoch_list = [.999, .9999, .9999, .9999, .9999, .9999, .9999, .9999, .9999, .9999, .9999, .99999]
        self.rec_av_fac_feat_norm = self.rec_av_fac_feat_norm_epoch_list[0]

        # statistics are estimated per frequency bin and feature map
        self.register_buffer('feat_mean', torch.zeros((1, self.feat_len_freq, self.num_feat_maps), device=self.device, requires_grad=False))
        self.register_buffer('feat_pow', torch.ones((1, self.feat_len_freq, self.num_feat_maps), device=self.device, requires_grad=False))
        self.register_buffer('feat_std', torch.ones((1, self.feat_len_freq, self.num_feat_maps), device=self.device, requires_grad=False))

    def comp_feature_maps(self, sig_block_dic):
        """Feature computation."""

        feat_maps = torch.tensor([], device=self.device)
        # magnitude features
        if 'abs_Uf' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Uf'])            # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Uf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'abs_Ef' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Ef'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Ef'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'abs_Df_est' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Df_est'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Df_est'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'abs_Yf' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Yf'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Yf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        # logarithmic magnitude features
        if 'log_abs_Uf' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Uf'])            # (batch_size x freq_size)
            feat_tmp = torch.log10(feat_tmp + self.floor_val)
            assert feat_tmp.shape == (sig_block_dic['Uf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'log_abs_Ef' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Ef'])           # (batch_size x freq_size)
            feat_tmp = torch.log10(feat_tmp + self.floor_val)
            assert feat_tmp.shape == (sig_block_dic['Ef'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'log_abs_Df_est' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Df_est'])           # (batch_size x freq_size)
            feat_tmp = torch.log10(feat_tmp + self.floor_val)
            assert feat_tmp.shape == (sig_block_dic['Df_est'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'log_abs_Yf' in self.feat_list:
            feat_tmp = torch.abs(sig_block_dic['Yf'])            # (batch_size x freq_size)
            feat_tmp = torch.log10(feat_tmp + self.floor_val)
            assert feat_tmp.shape == (sig_block_dic['Yf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        # complex features by stacking real and imaginary components
        if 'Uf_re_im' in self.feat_list:
            feat_tmp = torch.real(sig_block_dic['Uf'])            # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Uf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

            feat_tmp = torch.imag(sig_block_dic['Uf'])            # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Uf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'Ef_re_im' in self.feat_list:
            feat_tmp = torch.real(sig_block_dic['Ef'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Ef'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

            feat_tmp = torch.imag(sig_block_dic['Ef'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Ef'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'Df_est_re_im' in self.feat_list:
            feat_tmp = torch.real(sig_block_dic['Df_est'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Df_est'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

            feat_tmp = torch.imag(sig_block_dic['Df_est'])           # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Df_est'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'Yf_re_im' in self.feat_list:
            feat_tmp = torch.real(sig_block_dic['Yf'])            # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Yf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

            feat_tmp = torch.imag(sig_block_dic['Yf'])            # (batch_size x freq_size x temp_inf_length)
            assert feat_tmp.shape == (sig_block_dic['Yf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        # average power features
        if 'av_abs_Ef' in self.feat_list:
            feat_tmp = torch.mean(torch.abs(sig_block_dic['Ef']), dim=1, keepdim=True) * torch.ones((1, sig_block_dic['Ef'].shape[1]), device=self.device)              # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Ef'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'av_abs_Df_est' in self.feat_list:
            feat_tmp = torch.mean(torch.abs(sig_block_dic['Df_est']), dim=1, keepdim=True) * torch.ones((1, sig_block_dic['Df_est'].shape[1]), device=self.device)      # (batch_size x freq_size)
            assert feat_tmp.shape == (sig_block_dic['Df_est'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        if 'av_abs_Yf' in self.feat_list:
            feat_tmp = torch.mean(torch.abs(sig_block_dic['Yf']), dim=1, keepdim=True) * torch.ones((1, sig_block_dic['Yf'].shape[1]), device=self.device) # (batch_size x freq_size x temp_inf_length)       # (batch_size x 1 x temp_inf_length)
            assert feat_tmp.shape == (sig_block_dic['Yf'].shape[0], self.nnrb)
            feat_tmp = torch.unsqueeze(feat_tmp, dim=-1)                         # adding singleton feature map dimension
            feat_maps = torch.cat((feat_maps, feat_tmp), dim=-1)

        assert feat_maps.shape == (feat_maps.shape[0], self.feat_len_freq, self.num_feat_maps)

        # computation of normalization factors
        if self.training:
            with torch.no_grad():
                if self.my_current_epoch < len(self.rec_av_fac_feat_norm_epoch_list):
                    self.rec_av_fac_feat_norm = self.rec_av_fac_feat_norm_epoch_list[self.my_current_epoch]

                self.feat_mean = self.rec_av_fac_feat_norm * self.feat_mean + (1 - self.rec_av_fac_feat_norm) * torch.mean(feat_maps, dim=0, keepdim=True)       # estimate statistics by averaging over batch dimensions
                self.feat_pow = self.rec_av_fac_feat_norm * self.feat_pow + (1 - self.rec_av_fac_feat_norm) * torch.mean(feat_maps**2, dim=0, keepdim=True)

                if torch.any(self.feat_pow < self.feat_mean ** 2):
                    raise Exception('The feature normalization estimator does not work properly!')

                self.feat_std = torch.sqrt(self.feat_pow - self.feat_mean ** 2)

        # normalizing feature batch
        feat_maps = (feat_maps - self.feat_mean) / self.feat_std

        return feat_maps

    def required_signals(self):
        """Creating list of signal names which are required for feature computation."""

        sig_list = {}

        if ('abs_Uf' in self.feat_list) or ('log_abs_Uf' in self.feat_list) or ('Uf_re_im' in self.feat_list):
            sig_list['Uf'] = True
        else:
            sig_list['Uf'] = False

        if ('abs_Yf' in self.feat_list) or ('log_abs_Yf' in self.feat_list) or ('Yf_re_im' in self.feat_list):
            sig_list['Yf'] = True
        else:
            sig_list['Yf'] = False

        if ('abs_Ef' in self.feat_list) or ('log_abs_Ef' in self.feat_list) or ('Ef_re_im' in self.feat_list):
            sig_list['Ef'] = True
        else:
            sig_list['Ef'] = False

        if ('abs_Df_est' in self.feat_list) or ('log_abs_Df_est' in self.feat_list) or ('Df_est_re_im' in self.feat_list):
            sig_list['Df_est'] = True
        else:
            sig_list['Df_est'] = False

        return sig_list
