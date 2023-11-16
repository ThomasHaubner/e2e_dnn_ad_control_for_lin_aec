#
# Thomas Haubner, LMS, 2020
#
from pytorch_lightning import callbacks as Cb


class MyInitCallback(Cb.Callback):
    """Custom pytorch lighting callbacks."""

    def on_fit_start(self, trainer, pl_module):
        """Called at the beginnign of the fit routine."""

        # moving data to cpu or gpu
        if hasattr(pl_module, 'window'):
            pl_module.window = pl_module.window.to(pl_module.device)

        if hasattr(pl_module, 'floor_val_loss'):
            pl_module.floor_val_loss = pl_module.floor_val_loss.to(pl_module.device)

        if hasattr(pl_module, 'the_dnn') and hasattr(pl_module.the_dnn, 'the_features') and hasattr(pl_module.the_dnn.the_features, 'floor_val'):
            pl_module.the_dnn.the_features.floor_val = pl_module.the_dnn.the_features.floor_val.to(pl_module.device)

    def on_test_start(self, trainer, pl_module):
        """Called at the beginning of the test routine."""

        # moving data to cpu or gpu
        if hasattr(pl_module, 'window'):
            pl_module.window = pl_module.window.to(pl_module.device)

        if hasattr(pl_module, 'floor_val_loss'):
            pl_module.floor_val_loss = pl_module.floor_val_loss.to(pl_module.device)

        if hasattr(pl_module, 'the_dnn') and hasattr(pl_module.the_dnn, 'the_features') and hasattr(pl_module.the_dnn.the_features, 'floor_val'):
            pl_module.the_dnn.the_features.floor_val = pl_module.the_dnn.the_features.floor_val.to(pl_module.device)
