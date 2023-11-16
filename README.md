# End-To-End Deep Learning-based Adaptation Control for Linear Acoustic Echo Cancellation

This repository contains an implementation of the deep learning-controlled acoustic echo cancellation algorithm that is described in the publication *End-to-End Deep Learning-Based Adaptation Control for Linear Acoustic Echo Cancellation* by T. Haubner, A. Brendel and W. Kellermann (see bibtex entry below for details). 

## Instructions for Code Usage

1) Create and activate a virtual anaconda environment according to the provided YAML file.

2) Create your customized training and testing data sets which are named `train_data.h5` and `test_data.h5`, respectively. Both datasets need to be HDF5 files with the entries
* `u_td_tensor`: Loudspeaker signal tensor
* `y_td_tensor`: Microphone signal tensor
* `d_td_tensor`: Ground-truth echo signal tensor
* `s_td_tensor`: Near-end speech signal tensor (is only required for `test_data.h5`)

of dimension `num_sequences x signal_length`, respectively, and located in the folder `./data`. Note that exemplary HDF5 files are provided in the folder `./data`. Yet, as the respective exemplary datasets contain only very limited amount of data, they are not suitable to train the DNN.

3) Choose your desired settings in `./main_train.py`.

4) Train and test the algorithm by activating the conda environment and running `python main_train.py`. The processed data, including the averaged performance measures, will be saved in the subfolder `./data/proc_test_data`.

## Reference

If you use ideas or code from this work, please cite our paper:

IEEE Publication:
```BibTex
@ARTICLE{e2eDnnLinAec_ieee,
  author={Haubner, Thomas and Brendel, Andreas and Kellermann, Walter},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={End-to-End Deep Learning-Based Adaptation Control for Linear Acoustic Echo Cancellation}, 
  year={2024},
  volume={32},
  pages={227-238},
  doi={10.1109/TASLP.2023.3325923}
  }
```

ArXiv Preprint:
```BibTex
@misc{e2eDnnLinAec_arxiv,
      title={End-To-End Deep Learning-based Adaptation Control for Linear Acoustic Echo Cancellation}, 
      author={Thomas Haubner and Andreas Brendel and Walter Kellermann},
      year={2023},
      eprint={2306.02450},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
