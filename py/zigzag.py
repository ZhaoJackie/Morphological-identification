import pandas as pd
import numpy as np
from forex_db import Data


def zigzag(s, r1, r2):
    ut = 1 + r1
    dt = 1 - r2
    ld = s.index[0]
    lp = s.close[ld]
    tr = None
    zzd, zzp = [ld], [lp]
    for ix, ch, cl in zip(s.index, s.high, s.low):
        if tr is None:
            if ch / lp > ut:
                tr = 1
            elif cl / lp < dt:
                tr = -1
        elif tr == 1:
            if ch > lp:
                ld, lp = ix, ch
            elif cl / lp < dt:
                zzd.append(ld)
                zzp.append(lp)
                tr, ld, lp = -1, ix, cl
        else:
            if cl < lp:
                ld, lp = ix, cl
            elif ch / lp > ut:
                zzd.append(ld)
                zzp.append(lp)
                tr, ld, lp = 1, ix, ch
    if zzd[-1] != s.index[-1]:
        zzd.append(s.index[-1])
        if tr is None:
            zzp.append(s.close[zzd[-1]])
        elif tr == 1:
            zzp.append(s.high[zzd[-1]])
        else:
            zzp.append(s.low[zzd[-1]])
    return pd.Series(zzp, index=zzd)


def up(xts, r1=0.004, r2=0.004):
    r = zigzag(xts, r1, r2)
    # m = r
    r = r - r.shift(-1)
    r = r.dropna()
    p1 = np.where(r.index[np.where(r == max(r[r > 0]))[0][0]] == xts.index)[0][0]
    r_ = r[r != max(r)]
    p2 = np.where(r.index[np.where(r == max(r_[r_ > 0]))[0][-1]] == xts.index)[0][0]
    ret = [p1, p2]
    ret.sort()
    return {'up': ret}


def dn(xts, r1=0.004, r2=0.004):
    r = zigzag(xts, r1, r2)
    # m = r
    r = r - r.shift(1)
    r = r.dropna()
    p1 = np.where(r.index[np.where(r == min(r[r < 0]))[0][0]] == xts.index)[0][0]
    r_ = r[r != min(r)]
    p2 = np.where(r.index[np.where(r == min(r_[r_ < 0]))[0][-1]] == xts.index)[0][0]
    ret = [p1, p2]
    ret.sort()
    return {'dn': ret}


def rs(xts, ty, n):
    rate = [0.05, 0.05]
    r = zigzag(xts, rate[0], rate[1])
    ret = None
    if ty == 'up':
        r = r - r.shift(-1)
        r = r.dropna()
        r_ = r[r.index > xts.index[n]]
        p = np.where(r_.index[np.where(r_ == max(r_))[0][0]] == xts.index[n:])[0][-1] + n
        ret = xts.high[p]
    elif ty == 'dn':
        r = r - r.shift(1)
        r = r.dropna()
        r_ = r[r.index > xts.index[n]]
        p = np.where(r_.index[np.where(r_ == min(r_))[0][0]] == xts.index[n:])[0][-1] + n
        ret = xts.low[p]
    return ret


def test():
    ds = Data('15MIN').data()
    for tmp in ds:
        print tmp[0]
        d = tmp[1]
        # noinspection PyBroadException
        try:
            re = up(d, 0.004, 0.004)
        except:
            continue
        line = pd.Series([d.high[re['up'][0]], d.high[re['up'][1]]], index=[d.index[re['up'][0]], d.index[re['up'][1]]])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 9))
        d.high.plot()
        re['m'].plot()
        line.plot()
        # plt.show()
        plt.savefig('zig/' + tmp[0] + '60MIN' + '.png')
        plt.close()


if __name__ == '__main__':
    test()


# {'240MIN': 0.02, '15MIN': 0.005, '30MIN': 0.008, '60MIN': 0.01}
