import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances


dateparser = lambda x: Time(np.float32(x), format='jd')


class LC(object):
    _column_names = ['mjd', 'mag', 'err']

    def __init__(self, path):
        self.dir, self.fname = os.path.split(path)
        self.data = pd.read_table(path, sep=" ", names=self._column_names,
                                  engine='python', usecols=[0, 1, 2])
                                  # parse_dates=['mjd'],
                                  # date_parser=dateparser)
        self.n = len(self.data)
        self.dm_edges = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1,
                         0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5, 8]
        self.dt_edges = [0, 1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5,
                         3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600,
                         960, 2000, 4000]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __len__(self):
        return len(self.data)

    def save(self, fname, sep=" "):
        self.data.to_csv(fname, sep=sep, header=False, index=False)

    def plot(self, fig=None, fmt='.k'):
        if not fig:
            fig = plt.figure()
        mjd = self.mjd
        mag = self.mag
        err = self.err
        plt.errorbar(mjd, mag, err, fmt=fmt)
        plt.gca().invert_yaxis()
        plt.show()
        return fig

    @property
    def mag(self):
        return self.data['mag'].values

    @mag.setter
    def mag(self, other):
        self.data['mag'] = other

    @property
    def mjd(self):
        return self.data['mjd'].values

    @property
    def datetime(self):
        """
        Return times as python datetime objects.
        """
        t = Time(self.data['mjd'], format='jd', scale='utc')
        return [t_.datetime for t_ in t]

    @property
    def err(self):
        return self.data['err'].values

    @err.setter
    def err(self, other):
        self.data['err'] = other

    @property
    def dmdt(self):
        """
        Return ``dmdt`` representation of the light curve following
        arXiv:1709.06257
        """
        metric = lambda x1, x2: x2-x1

        m = self.mag.reshape(-1, 1)
        t = self.mjd.reshape(-1, 1)
        m_dm = pairwise_distances(m, metric=metric)
        t_dm = pairwise_distances(t, metric=metric)
        ind = np.triu_indices(self.n, 1)
        dmdt = np.dstack((m_dm, t_dm))[ind[0], ind[1], ...]
        h, xedges, yedges = np.histogram2d(dmdt[:, 0], dmdt[:, 1],
                                           bins=(self.dm_edges, self.dt_edges))
        # Normalization to [0, 255] scale
        p = self.n*(self.n-1)/2
        h *= 255/p
        h += 0.99999
        h = np.asanyarray(h, dtype=int)
        return h


if __name__ == "__main__":
    lc = LC('/home/ilya/Dropbox/papers/ogle2/data/sc19/lmc_sc19_i_28995.dat')
    fig = lc.plot()
    dmdt = lc.dmdt
    plt.matshow(dmdt)
    plt.show()