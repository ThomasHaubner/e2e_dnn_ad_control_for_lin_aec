#
# Thomas Haubner, LMS, 2022
#
import torch
import pytorch_lightning as pl


class DNN_supported_nlms_ctf_aec(pl.LightningModule):
    """Acoustic echo canceler class."""

    def __init__(self, aec_dic):

        super(DNN_supported_nlms_ctf_aec, self).__init__()

        # DNN training parameters
        self.batch_size = None

        # adaptive filter parameters
        self.ctf_filt_len = aec_dic['ctf_filt_len']
        self.nnrb = aec_dic['nnrb']

        # step-size parameters
        self.Psi_U_lambda = aec_dic['Psi_U_lambda']
        self.floor_val = torch.tensor(aec_dic['floor_val'], device=self.device)
        self.delta_reg_num = torch.tensor(aec_dic['delta_reg_num'], device=self.device)

        self.est_mask_mu = aec_dic['est_mask_mu']
        assert self.est_mask_mu in ['dnnFreq', 'dnnScal']
        self.est_mask_Ef = aec_dic['est_mask_Ef']
        assert self.est_mask_Ef in ['dnnFreq', 'dnnScal', 'zero', 'one']

        # internal state of the adaptive filter
        self.block_ind = None                       # STFT block (frame) index
        self.Psi_UfUf = None                        # loudspeaker power tensor
        self.Hf_est = None                          # estimated CTF coefficient tensor

    def comp_prior_error(self, Uf_ctf, Yf):
        """Compute prior AEC error."""

        assert Uf_ctf.shape == (self.batch_size, self.nnrb, self.ctf_filt_len)
        assert Yf.shape == (self.batch_size, self.nnrb)

        Y_est_stft_block = torch.sum(self.Hf_est * Uf_ctf, dim=-1)         # Estimated echo tensor (batch_size x nnrb)
        E_aec_stft_block = Yf - Y_est_stft_block                           # Error tensor

        return E_aec_stft_block, Y_est_stft_block

    def update_filter(self, dnn_masks_aec_dic, Uf_ctf, Ef_prior):
        """Update CTF filter coefficient vector by NLMS update."""
        # dnn_masks_aec_dic.keys() = [aec_mask_step_size, aec_mask_Ef]

        assert Uf_ctf.shape == (self.batch_size, self.nnrb, self.ctf_filt_len)
        assert Ef_prior.shape == (self.batch_size, self.nnrb)

        # loudspeaker power estimation
        if self.block_ind == 0:
            # initialize loudspeaker power
            self.Psi_UfUf = torch.sum(torch.max(torch.abs(Uf_ctf)**2, self.floor_val), dim=-1, keepdim=True)        # loudspeaker power tensor (batch_size x nnrb x 1)

        else:
            # update loudspeaker power by recursive averaging
            self.Psi_UfUf = self.Psi_U_lambda * self.Psi_UfUf + (1 - self.Psi_U_lambda) * torch.sum(torch.abs(Uf_ctf)**2, dim=-1, keepdim=True)             # (batch_size x nnrb x 1)

        # estimate the stochastic adaptive filter gradient
        dHf_inst = torch.conj(Uf_ctf) * torch.unsqueeze(Ef_prior, dim=-1)                   # LMS gradient tensor (batch_size x nnrb x ctf_filt_len)

        # compute frequency-selective step-size
        Psi_ZfZf = torch.unsqueeze(torch.abs(dnn_masks_aec_dic['mask_Ef'] * Ef_prior)**2, dim=-1)
        stepSize_adFilt = dnn_masks_aec_dic['mask_mu'] / (self.Psi_UfUf + Psi_ZfZf + self.delta_reg_num)    # step-size tensor

        # update filter
        self.Hf_est = self.Hf_est + stepSize_adFilt * dHf_inst                  # Updated CTF coefficient tensor

        # increase STFT block (frame) index
        self.block_ind = self.block_ind + 1

        if torch.any(torch.isnan(self.Hf_est)):
            print('NaN occured during AEC filter update!')


    def init_parameters(self, batch_size):
        """Initialize CTF filter coefficient vector."""
        self.batch_size = batch_size
        self.block_ind = 0
        self.Hf_est = torch.zeros((batch_size, self.nnrb, self.ctf_filt_len), device=self.device, dtype=torch.complex64)          # estimated CTF filter coefficient tensor (batch_size x nnrb x ctf_filt_len)
