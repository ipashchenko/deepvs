import os
import numpy as np
from dmdt import LC
import tqdm


def get_data(var_fname_file, const_fname_file, data_dir, const_fraction=None,
             const_number=None, seed=1):
    np.random.seed(seed)
    with open(var_fname_file, 'r') as fo:
        var_fnames = fo.readlines()
    var_fnames = [var_fname.strip('\n') for var_fname in var_fnames]
    var_fnames = [os.path.join(data_dir, var_fname) for var_fname in
                   var_fnames]
    print("Loading variable stars LCs...")
    var_dmdt = [LC(fn).dmdt for fn in tqdm.tqdm(var_fnames)]
    shape = var_dmdt[0].shape
    with open(const_fname_file, 'r') as fo:
        const_fnames = fo.readlines()
    const_fnames = [const_fname.strip('\n') for const_fname in const_fnames]

    if const_number is not None:
        ind = np.random.choice(np.arange(len(const_fnames)), size=const_number,
                               replace=False)
    elif const_fraction is not None:
        ind = np.random.randint(0, len(const_fnames),
                                size=int(const_fraction*len(const_fnames)))
    else:
        ind = np.arange(len(const_fnames))
    const_fnames = [const_fnames[i] for i in ind]
    const_fnames = [os.path.join(data_dir, const_fname) for const_fname in
                     const_fnames]
    print("Loading constant stars LCs...")
    const_dmdt = [LC(fn).dmdt for fn in tqdm.tqdm(const_fnames)]

    images = np.array(var_dmdt+const_dmdt).reshape((len(var_dmdt)+len(const_dmdt),
                                                    shape[0],
                                                    shape[1],
                                                    1))
    labels = np.hstack((np.ones(len(var_dmdt)), np.zeros(len(const_dmdt))))
    return images, labels
