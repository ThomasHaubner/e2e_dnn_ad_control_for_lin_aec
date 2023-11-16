#
# Thomas Haubner, LMS, 2022
#
import torch
import pytorch_lightning as pl
import libPython.class_dnnFeatures as class_dnnFeatures


class DNN_denseGruDense_BB(pl.LightningModule):
    """Broadband DNN for step-size inference."""

    def __init__(self, dnn_dic):

        super(DNN_denseGruDense_BB, self).__init__()

        # preparation
        self.name = dnn_dic['name'] + '_(' + str(dnn_dic['hidden_size_gru']) + '-' + str(dnn_dic['num_layers_gru']) + ')'
        self.dnn_dic = dnn_dic

        # computing mask dimensions
        assert dnn_dic['nfft'] % 2 == 0                         # DFT length must be even
        self.mask_freq_size = dnn_dic['nfft'] // 2 + 1          # number of non-redundant frequency bins due to spectral symmetry

        # feature class
        self.the_features = class_dnnFeatures.DnnFeature(dnn_dic['feat_dic'])
        self.feat_size = self.the_features.feat_size

        # estimation targets of the DNN
        self.est_mask_mu = self.dnn_dic['aec_dic']['est_mask_mu']
        assert self.est_mask_mu in ['dnnFreq', 'dnnScal']
        self.est_mask_Ef = self.dnn_dic['aec_dic']['est_mask_Ef']
        assert self.est_mask_Ef in ['dnnFreq', 'dnnScal', 'zero', 'one']

        # network definition
        self.hidden_size_gru = self.dnn_dic['hidden_size_gru']
        self.num_layers_gru = self.dnn_dic['num_layers_gru']

        # output activation
        self.sigmoid = torch.nn.Sigmoid()

        # encoder architecture
        self.enc_mod_list = torch.nn.ModuleList()
        self.enc_mod_list.append(torch.nn.Linear(self.feat_size, self.hidden_size_gru))
        self.enc_mod_list.append(torch.nn.LeakyReLU())
        self.gru_filt = torch.nn.GRU(self.hidden_size_gru, self.hidden_size_gru, self.num_layers_gru)

        # decoder architecture for estimation of mask mu
        if self.est_mask_mu == 'dnnFreq':
            # estimate only frequency-selective step-size
            self.dec_aec_stepSize_mod_list = torch.nn.ModuleList()
            self.dec_aec_stepSize_mod_list.append(torch.nn.Linear(self.hidden_size_gru, self.mask_freq_size))
            self.dec_aec_stepSize_mod_list.append(torch.nn.Sigmoid())

        elif self.est_mask_mu == 'dnnScal':
            # estimate only scalar step-size
            self.dec_aec_stepSize_mod_list = torch.nn.ModuleList()
            self.dec_aec_stepSize_mod_list.append(torch.nn.Linear(self.hidden_size_gru, 1))
            self.dec_aec_stepSize_mod_list.append(torch.nn.Sigmoid())

        else:
            raise Exception('This setting is not known')

        # decoder architecture for estimation of mask Ef
        if self.est_mask_Ef == 'dnnFreq':
            self.dec_aec_maskEfNum_mod_list = torch.nn.ModuleList()
            self.dec_aec_maskEfNum_mod_list.append(torch.nn.Linear(self.hidden_size_gru, self.mask_freq_size))
            self.dec_aec_maskEfNum_mod_list.append(torch.nn.Sigmoid())
        elif self.est_mask_Ef == 'dnnScal':
            self.dec_aec_maskEfNum_mod_list = torch.nn.ModuleList()
            self.dec_aec_maskEfNum_mod_list.append(torch.nn.Linear(self.hidden_size_gru, 1))
            self.dec_aec_maskEfNum_mod_list.append(torch.nn.Sigmoid())
        elif self.est_mask_Ef in ['zero', 'one']:
            pass
        else:
            raise Exception('This setting is not known')

    def forward_block(self, sig_block_dic, gru_state_in):
        """Forward run of a single STFT block."""

        # compute feature maps
        feat_maps = self.the_features.comp_feature_maps(sig_block_dic)
        feat_vec = torch.flatten(feat_maps, start_dim=1, end_dim=-1)

        # DNN inference
        x = feat_vec
        for enc_block in self.enc_mod_list:
            x = enc_block(x)

        # apply GRU
        # x_out <=> (num_batches, seq_len, num_hidden_states)
        # gru_state_out <=> (num_gru_layers, num_batches, num_hidden_states)
        x = torch.unsqueeze(x, dim=0)                           # adding dummy dimension for sequence index
        x, gru_state_out = self.gru_filt(x, gru_state_in)
        x_out_gru = torch.squeeze(x, dim=0)                     # deleting dummy sequence index

        # infer step-size control masks from GRU
        dnn_masks_aec_dic = self.forward_block_decoder(x_out_gru)

        return dnn_masks_aec_dic, gru_state_out

    def forward_block_decoder(self, x_out_gru):
        """Inference of step-size masks from GRU states."""

        # decoder inference of mu mask
        x = x_out_gru
        for dec_block in self.dec_aec_stepSize_mod_list:
            x = dec_block(x)
        mask_mu = torch.unsqueeze(x, dim=2)     # (num_batch x nnrb x 1)        (singleton dimension for filter dimension)

        # decoder inference of Ef mask
        if self.est_mask_Ef[:3] == 'dnn':
            x = x_out_gru
            for dec_block in self.dec_aec_maskEfNum_mod_list:
                x = dec_block(x)
            mask_Ef = x

        elif self.est_mask_Ef == 'zero':
            mask_Ef = torch.tensor([[.0]], device=self.device)

        elif self.est_mask_Ef == 'one':
            mask_Ef = torch.tensor([[1.0]], device=self.device)
        else:
            raise Exception('This setting is not known!')

        dnn_masks_aec_dic = {
            'mask_mu': mask_mu,
            'mask_Ef': mask_Ef
        }

        return dnn_masks_aec_dic

    # initialization of hidden DNN states
    def init_hidden(self, batch_size):
        """Initialization of internal GRU states."""

        init_state = torch.zeros((self.num_layers_gru, batch_size, self.hidden_size_gru), requires_grad=False, device=self.device)

        return init_state
