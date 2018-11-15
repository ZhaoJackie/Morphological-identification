# -*- coding: UTF-8 -*-
import pandas
import numpy
import zigzag


class Triangle:
    def __init__(self, data, typ):
        assert type(data) == pandas.core.frame.DataFrame
        assert type(data.index) == pandas.core.indexes.datetimes.DatetimeIndex
        self.data = data
        self.typ = typ
        self.result = dict()
        self._ret()

    @staticmethod
    def _points(ts):
        le = len(ts)
        ind = pandas.Series(range(le), index=ts.index)
        point1 = None
        point2 = None
        # noinspection PyBroadException
        try:
            # 按照低高低高的顺序把关键点找出来
            x_low_f = ind[ts.low.isin([ts.low.min()])][0]
            x_high_f = ind[x_low_f:][ts.high[x_low_f:].isin([ts.high[x_low_f:].max()])][0]
            x_low_s = ind[x_high_f:][ts.low[x_high_f:].isin([ts.low[x_high_f:].min()])][0]
            x_high_s = ind[x_low_s:][ts.high[x_low_s:].isin([ts.high[x_low_s:].max()])][0]
            # x_low_t = ind[x_high_s:][ts.low[x_high_s:].isin([ts.low[x_high_s:].max()])][-1]
            point1 = {'up': [x_high_f, x_high_s], 'dn': [x_low_f, x_low_s]}
        except:
            # 如果最低点是最后一个点的话，那么找点会失败
            pass
        # noinspection PyBroadException
        try:
            # 按照高低高低的顺序来寻找关键点
            x_high_f = ind[ts.high.isin([ts.high.max()])][0]
            x_low_f = ind[x_high_f:][ts.low[x_high_f:].isin([ts.low[x_high_f:].min()])][0]
            x_high_s = ind[x_low_f:][ts.high[x_low_f:].isin([ts.high[x_low_f:].max()])][0]
            x_low_s = ind[x_high_s:][ts.low[x_high_s:].isin([ts.low[x_high_s:].min()])][0]
            # x_high_t = ind[x_low_s:][ts.high[x_low_s:].isin([ts.high[x_low_s:].min()])][-1]
            point2 = {'up': [x_high_f, x_high_s], 'dn': [x_low_f, x_low_s]}
        except:
            # 如果第一个高点是最后一个点的话，找关键点会失败
            pass
        return {'point1': point1, 'point2': point2}

    @staticmethod
    def _lr(x, y):
        # 线性回归，计算直线的截距和斜率
        assert x[0] != x[1]
        intercept = y[0] - (y[1] - y[0]) * x[0] / (x[1] - x[0])
        slop = (y[1] - y[0]) / (x[1] - x[0])
        return intercept, slop

    @staticmethod
    def _is_break(ts, up, dn, last_index, how):
        # 判断是否失败
        # return numpy.sum(ts.high > up) + numpy.sum(ts.low < dn) > 0
        #最后一个识别点之前不能够破位，之后是最后一个点破位，(最后一个识别点和最后一个点之间距离需要5跟bar???)
        if how == 'dn':
            return numpy.sum(ts.high > up) + numpy.sum(ts.low[:(last_index + 1)] < dn[:(last_index + 1)]) > 0 or \
                   not numpy.sum(ts.low[(last_index + 1):] < dn[(last_index + 1):]) == 1 or \
                   not ts.low[-1] < dn[-1]
        else:
            return numpy.sum(ts.high[:(last_index + 1)] > up[:(last_index + 1)]) + numpy.sum(ts.low < dn) > 0 or \
                  not numpy.sum(ts.high[(last_index + 1):] > up[(last_index + 1):]) == 1 or \
                not ts.high[-1] > up[-1]

    def _ret(self):
        le = len(self.data)
        st = set()
        for ind in xrange(le):
            xts = self.data.iloc[ind:, :]
            le = len(xts)
            # noinspection PyBroadException
            points = self._points(xts)
            for point in points.values():
                if point:
                    up_x = point['up']
                    # 第二个高点与第一个高点之间要多于5个bar
                    # 第二个高点不能是最后一个bar
                    if up_x[1] - up_x[0] < 5 or up_x[1] >= le - 1:
                        continue
                    up_y = [xts.high[up_x[0]], xts.high[up_x[1]]]
                    dn_x = point['dn']
                    # 第二个低点与第一个低点之间要多于5个bar
                    # 第二个低点不能是最后一个bar
                    if dn_x[1] - dn_x[0] < 5 or dn_x[1] >= le - 1:
                        continue
                    tm = up_x + dn_x
                    # 将高低点list按照大小排序
                    tm.sort()
                    # if tm[-1] - tm[0] < 20 or tm[2] - tm[0] > 3.5 * (tm[3] - tm[1]) or tm[1] - tm[0] < 3 \
                    #         or tm[2] - tm[1] < 3 or tm[3] - tm[2] < 3:
                    #     continue
                    # 最开始的点与最后的点之间要多于20个bar
                    # 第四个点与第一个点之间的bar的数目要多于4倍的第四个bar与第二个bar的数目
                    # 点与点不能重复
                    if tm[-1] - tm[0] < 20 or tm[2] - tm[0] > 4 * (tm[3] - tm[1]) or tm[0] == tm[1] or tm[1] == tm[2] \
                            or tm[2] == tm[3]:
                        continue
                    # 每两个点之间差距要多于5个bar并且最左边的高低点不能是是第一根bar需要留一点空间（5根bar）
                    if tm[0] < 5 or tm[1] - tm[0] < 5 or tm[2] - tm[1] < 5 or tm[3] - tm[2] < 5:
                        continue
                    dn_y = [xts.low[dn_x[0]], xts.low[dn_x[1]]]
                    # 计算上轨和下轨的截距和斜率
                    up_intercept, up_slope = self._lr(up_x, up_y)
                    dn_intercept, dn_slope = self._lr(dn_x, dn_y)
                    # if abs(up_slope - dn_slope) < 0.02:
                    #     continue
                    # 计算上轨和下轨具体序列
                    up = pandas.Series(index=xts.index)
                    up[up_x[0]:] = up_intercept + up_slope * numpy.array(range(up_x[0], le))
                    dn = pandas.Series(index=xts.index)
                    dn[dn_x[0]:] = dn_intercept + dn_slope * numpy.array(range(dn_x[0], le))
                    # 判断上下轨是否破位
                    if up_x[1] > dn_x[1]:
                        last_index = up_x[1]
                        how = 'dn'
                    else:
                        last_index = dn_x[1]
                        how = 'up'
                    pw = self._is_break(xts, up, dn, last_index, how)
                    if pw:
                        continue
                    up_tps = [xts.index[up_x[0]], xts.index[up_x[1]]]
                    dn_tps = [xts.index[dn_x[0]], xts.index[dn_x[1]]]
                    # 计算支撑位阻力位
                    resi = pandas.Series(index=xts.index)
                    resi[:] = up_y[1]
                    supp = pandas.Series(index=xts.index)
                    supp[:] = dn_y[1]
                    # 返回最终结果，用来画图，传向后台
                    t = {'up': up, 'dn': dn, 'xts': xts, 'resi': resi, 'supp': supp,
                         'pattern': 'triangle', 'typ': self.typ, 'starttime': str(xts.index[tm[0]]),
                         'moldingtime': str(xts.index[tm[-1]]), 'startprice': xts.close[tm[0]],
                         'moldingprice': xts.close[tm[-1]]}
                    if len(set(up_tps + dn_tps) & st) < 3:
                        st = set(up_tps + dn_tps)
                        self.result['t' + str(ind)] = t


class Wedge:
    def __init__(self, data, typ):
        assert type(data) == pandas.core.frame.DataFrame
        assert type(data.index) == pandas.core.indexes.datetimes.DatetimeIndex
        self.data = data.head(len(data)-1)
        self.original_data = data
        self.typ = typ
        self.result = dict()
        self._ret()

    @staticmethod
    def _points(ts):
        le = len(ts)
        ind = pandas.Series(range(le), index=ts.index)
        point1 = None
        point2 = None
        # noinspection PyBroadException
        try:
            # 按照低高低高的顺序找到四个关键点
            x_low_f = ind[ts.low.isin([ts.low.min()])][0]
            x_high_s = ind[x_low_f:][ts.high[x_low_f:].isin([ts.high[x_low_f:].max()])][0]
            x_low_s = ind[x_high_s:][ts.low[x_high_s:].isin([ts.low[x_high_s:].min()])][0]
            x_high_f = ind[:x_low_f][ts.high[:x_low_f].isin([ts.high[:x_low_f:].max()])][0]
            point1 = {'up': [x_high_f, x_high_s], 'dn': [x_low_f, x_low_s]}
        except:
            # 像三角形一样，有可能寻找失败
            pass
        # noinspection PyBroadException
        try:
            # 按照高低高低的顺序寻找四个关键点
            x_high_f = ind[ts.high.isin([ts.high.max()])][0]
            x_low_s = ind[x_high_f:][ts.low[x_high_f:].isin([ts.low[x_high_f:].min()])][0]
            x_high_s = ind[x_low_s:][ts.high[x_low_s:].isin([ts.high[x_low_s:].max()])][0]
            x_low_f = ind[:x_high_f][ts.low[:x_high_f].isin([ts.low[:x_high_f].min()])][0]
            point2 = {'up': [x_high_f, x_high_s], 'dn': [x_low_f, x_low_s]}
        except:
            # 像三角形一样，有寻找失败的可能
            pass
        return {'point1': point1, 'point2': point2}

    @staticmethod
    def _lr(x, y):
        # 线性回归计算直线截距和斜率
        assert x[0] != x[1]
        intercept = y[0] - numpy.round((y[1] - y[0]) * x[0] / (x[1] - x[0]), 6)
        slop = numpy.round((y[1] - y[0]) / (x[1] - x[0]), 6)
        return intercept, slop

    @staticmethod
    def _is_break(ts, up, dn):
        return numpy.sum(ts.high > up) + numpy.sum(ts.low < dn) > 0

    @staticmethod
    def _is_shut(slop1, slop2, intercept1, intercept2, le):
        # 楔形不能张口，要闭口，就是说两条线的交叉点在图像的右边
        return (intercept1 - intercept2)/(slop2 - slop1) > le and slop1 * slop2 > 0  # and \
        # numpy.abs((slop1 - slop2)/(slop1 + slop2)) >= 0.085

    def _ret(self):
        le = len(self.data)
        st = set()
        for ind in xrange(le):
            xts = self.data.iloc[ind:, :]
            le = len(xts)
            # noinspection PyBroadException
            points = self._points(xts)
            for point in points.values():
                if point:
                    up_x = point['up']
                    # 两个高点和低点之间的距离不能小于5
                    # 第二个高点不能是最后一个点
                    if up_x[1] - up_x[0] < 5 or up_x[1] >= le - 1:
                        continue
                    up_y = [xts.high[up_x[0]], xts.high[up_x[1]]]
                    dn_x = point['dn']
                    # 两个低点之间的距离不能小于5个bar
                    # 第二个低点不能是最后一个点
                    if dn_x[1] - dn_x[0] < 5 or dn_x[1] >= le - 1:
                        continue
                    tm = up_x + dn_x
                    # 将四个点合并起来并排序
                    tm.sort()
                    # if tm[-1] - tm[0] < 20 or tm[2] - tm[0] > 3.5 * (tm[3] - tm[1]) or tm[1] - tm[0] < 3 \
                    #         or tm[2] - tm[1] < 3 or tm[3] - tm[2] < 3:
                    #     continue
                    # 最开始的点与最后的点之间要多于20个bar
                    # 第四个点与第一个点之间的bar的数目要多于4倍的第四个bar与第二个bar的数目
                    # 点与点不能重复
                    if tm[-1] - tm[0] < 20 or tm[2] - tm[0] > 4 * (tm[3] - tm[1]) or tm[0] == tm[1] or tm[1] == tm[2] or tm[2] == tm[3]:
                        continue
                    # 每两个点之间差距要多于5个bar并且最左边的高低点不能是是第一根bar需要留一点空间（5根bar）
                    if tm[0] < 5 or tm[1] - tm[0] < 5 or tm[2] - tm[1] < 5 or tm[3] - tm[2] < 5:
                        continue
                    dn_y = [xts.low[dn_x[0]], xts.low[dn_x[1]]]
                    # 计算上轨和下轨的斜率截距
                    up_intercept, up_slope = self._lr(up_x, up_y)
                    dn_intercept, dn_slope = self._lr(dn_x, dn_y)
                    # if abs(up_slope - dn_slope) < 0.02:
                    #     continue
                    # 计算上下轨
                    if up_x[1] > dn_x[1]:
                        direction = -1
                    else:
                        direction = 1
                    # else:
                    #     print '************************'
                    # if abs(up_slope - dn_slope) < 0.02:
                    #     continue
                    # up = pandas.Series(index=xts.index)
                    # print len(pandas.Series(index=xts.index))
                    up = pandas.Series(index=self.original_data.iloc[ind:, :].index)

                    up[up_x[0]:] = up_intercept + up_slope * numpy.array(range(up_x[0], le + 1))
                    # print len(up)
                    # print len(up.head(len(up)-1))
                    # print len(self.original_data.iloc[ind:, :])
                    # print len(xts)
                    # dn = pandas.Series(index=xts.index)
                    dn = pandas.Series(index=self.original_data.iloc[ind:, :].index)
                    dn[dn_x[0]:] = dn_intercept + dn_slope * numpy.array(range(dn_x[0], le + 1))
                    pw = self._is_break(xts, up.head(len(up)-1), dn.head(len(dn)-1))
                    if pw:
                        continue
                    # 判断是否为闭口的，这是区别三角形和楔形的关键
                    shut = self._is_shut(up_slope, dn_slope, up_intercept, dn_intercept, le)
                    if not shut:
                        continue
                    # 计算支撑位和阻力位
                    up_tps = [xts.index[up_x[0]], xts.index[up_x[1]]]
                    dn_tps = [xts.index[dn_x[0]], xts.index[dn_x[1]]]
                    resi = pandas.Series(index=xts.index)
                    resi[:] = up_y[1]
                    supp = pandas.Series(index=xts.index)
                    supp[:] = dn_y[1]
                    t = {"dir": direction, 'up': up, 'dn': dn, 'xts': self.original_data.iloc[ind:, :], 'resi': resi, 'supp': supp,
                         'pattern': 'wedge', 'typ': self.typ, 'starttime': str(xts.index[tm[0]]),
                         'moldingtime': str(xts.index[tm[-1]]), 'startprice': xts.close[tm[0]],
                         'moldingprice': xts.close[tm[-1]]}
                    if len(set(up_tps + dn_tps) & st) < 3:
                        st = set(up_tps + dn_tps)
                        self.result['t' + str(ind)] = t


class Channel:
    def __init__(self, data, typ=None):
        assert type(data) == pandas.core.frame.DataFrame
        assert type(data.index) == pandas.core.indexes.datetimes.DatetimeIndex
        self.data = data
        self.typ = typ
        self.result = dict()
        self._ret()

    @staticmethod
    def _points(s, typ):
        # zigzag工具所需要的参数
        if typ == '15MIN':
            rate = [0.004, 0.004]
        elif typ == '30MIN':
            rate = [0.008, 0.008]
        elif typ == '60MIN':
            rate = [0.01, 0.01]
        elif typ == '240MIN':
            rate = [0.02, 0.02]
        else:
            rate = [0.03, 0.03]
        point1 = None
        point2 = None
        # noinspection PyBroadException
        try:
            # 用zigzag识别出来的两个高点，如果参数过大，可能一个高点也找不到
            point1 = zigzag.up(s, r1=rate[0], r2=rate[1])
        except:
            pass
        # noinspection PyBroadException
        try:
            # 用zigzag识别出来两个低点，如果参数过大，可能一个低点也找不到
            point2 = zigzag.dn(s, r1=rate[0], r2=rate[1])
        except:
            pass
        return {'point1': point1, 'point2': point2}

    @staticmethod
    def _lr(x, y):
        # 线性回归，计算直线斜率和截距
        assert x[0] != x[1]
        intercept = y[0] - numpy.round((y[1] - y[0]) * x[0] / (x[1] - x[0]), 10)
        slop = numpy.round((y[1] - y[0]) / (x[1] - x[0]), 10)
        return intercept, slop

    def _ret(self):
        le = len(self.data)
        st = set()
        for ind in xrange(le):
            xts = self.data.iloc[ind:, :]
            le = len(xts)
            points = self._points(xts, self.typ)
            for point in points.values():
                if point:
                    if point.keys()[0] == 'up':
                        up_x = point['up']
                        # 两个高点之间的距离不能小于20个bar
                        # 第二个高点不能是最后一个点
                        if up_x[1] - up_x[0] < 20 or up_x[1] >= le - 1:
                            continue
                        # 计算上轨道的斜率
                        up_y = [xts.high[up_x[0]], xts.high[up_x[1]]]
                        up_intercept, up_slope = self._lr(up_x, up_y)
                        # 如果上轨道，那么不能斜率向上，因为向上的通道必须先画下轨道
                        if up_slope > 0:
                            continue
                        up = pandas.Series(index=xts.index)
                        # 计算上轨道序列
                        up[up_x[0]:] = up_intercept + up_slope * numpy.array(range(up_x[0], le))
                        # 找开盘价收盘价中序列最小值（一个序列）
                        min_stand = numpy.minimum(xts.open, xts.close)
                        # 找开盘价收盘价中序列最大值（一个序列）
                        max_stand = numpy.maximum(xts.open, xts.close)
                        # 计算
                        # 计算平移截距
                        stance = max(abs(up_slope * numpy.array(range(up_x[0], le)) + up_intercept -
                                         min_stand.values[up_x[0]:le]) / numpy.sqrt(1 + up_slope ** 2))
                        # 有截距了，有斜率了，可以计算平移曲线
                        dn_intercept, dn_slope = up_intercept - stance * numpy.sqrt(1 + up_slope ** 2), up_slope
                        dn = pandas.Series(index=xts.index)
                        # 计算平移后的曲线，即是下轨道
                        dn[up_x[0]:] = dn_intercept + dn_slope * numpy.array(range(up_x[0], le))
                        # 观察上下轨道破位与否
                        if numpy.sum(up < max_stand) > 0:
                            continue
                        if numpy.sum(dn > xts.low) < 1:
                            continue
                        # 下轨道破位的地方点距离不小于上轨道距离的0.3倍
                        inx = pandas.Series(range(le), index=xts.index)
                        tp = inx[dn > xts.low]
                        if tp.max() - tp.min() < 0.3 * (up_x[1] - up_x[0]):
                            continue

                        if up_x[1] > tp.max():
                            direction = -1
                        else:
                            direction = 1

                        up_tps = [xts.index[up_x[0]], xts.index[up_x[1]]]
                        resi = pandas.Series(index=xts.index)
                        resi[:] = up_y[1]
                        supp = pandas.Series(index=xts.index)
                        supp[:] = zigzag.rs(xts, 'dn', up_x[1])
                        t = {'dir': direction, 'up': up, 'dn': dn, 'xts': xts, 'resi': resi, 'supp': supp,
                             'pattern': 'channel', 'typ': self.typ, 'starttime': str(up_tps[0]),
                             'moldingtime': str(up_tps[1]), 'startprice': xts.close[up_x[0]],
                             'moldingprice': xts.close[up_x[1]]}
                        if len(set(up_tps) & st) < 1:
                            st = set(up_tps)
                            self.result['t' + str(ind)] = t
                    if point.keys()[0] == 'dn':
                        dn_x = point['dn']
                        # 两个低点之间的距离不能小于20个bar
                        # 第二个低点不能是最后一个点
                        if dn_x[1] - dn_x[0] < 20 or dn_x[1] >= le - 1:
                            continue
                        # 计算下轨道的斜率
                        dn_y = [xts.low[dn_x[0]], xts.low[dn_x[1]]]
                        dn_intercept, dn_slope = self._lr(dn_x, dn_y)
                        # 如果下轨道，那么不能斜率向下，因为向下的通道必须先画上轨道
                        if dn_slope < 0:
                            continue
                        # 计算下轨道序列
                        dn = pandas.Series(index=xts.index)
                        dn[dn_x[0]:] = dn_intercept + dn_slope * numpy.array(range(dn_x[0], le))
                        # 计算开盘和收盘最小值序列
                        # 计算开盘和收盘最大值序列
                        min_stand = numpy.minimum(xts.open, xts.close)
                        max_stand = numpy.maximum(xts.open, xts.close)
                        # 计算平移幅度
                        stance = max(abs(dn_slope * numpy.array(range(dn_x[0], le)) + dn_intercept -
                                         max_stand.values[dn_x[0]:le]) / numpy.sqrt(1 + dn_slope ** 2))
                        # 计算上轨的斜率和截距
                        up_intercept, up_slope = dn_intercept + stance * numpy.sqrt(1 + dn_slope ** 2), dn_slope
                        # 有斜率和截距了可以计算上轨道了
                        up = pandas.Series(index=xts.index)
                        up[dn_x[0]:] = up_intercept + up_slope * numpy.array(range(dn_x[0], le))
                        # 判断两条线的破位情况
                        if numpy.sum(dn > min_stand) > 0:
                            continue
                        if numpy.sum(up < xts.high) < 1:
                            continue
                        # 破位的两点之间bars不能小于下轨道bar数目的0.3倍
                        inx = pandas.Series(range(le), index=xts.index)
                        tp = inx[up < xts.high]
                        if tp.max() - tp.min() < 0.3 * (dn_x[1] - dn_x[0]):
                            continue

                        if dn_x[1] > tp.max():
                            direction = 1
                        else:
                            direction = -1

                        dn_tps = [xts.index[dn_x[0]], xts.index[dn_x[1]]]
                        resi = pandas.Series(index=xts.index)
                        # resi[:] = zigzag.rs(xts, 'up', dn_x[1])
                        supp = pandas.Series(index=xts.index)
                        supp[:] = dn_y[1]

                        t = {'dir': direction, 'up': up, 'dn': dn, 'xts': xts, 'resi': resi, 'supp': supp,
                             'pattern': 'channel', 'typ': self.typ, 'starttime': str(dn_tps[0]),
                             'moldingtime': str(dn_tps[1]), 'startprice': xts.close[dn_x[0]],
                             'moldingprice': xts.close[dn_x[1]]}
                        if len(set(dn_tps) & st) < 1:
                            st = set(dn_tps)
                            self.result['t' + str(ind)] = t
