"""
Created on Tue Jul 21 23:12:55 2020

@author: Muhammad Waqar
"""

# Importing Libraries
import pandas
from multiprocessing import Pool
from datetime import datetime, timedelta
from scipy.stats import *
import quandl
import math
import random
from numpy.random import normal as nl
import numpy as np
import matplotlib.pyplot as plt

# config
pandas.options.mode.chained_assignment = None

# Initializing Quandl
quandl.ApiConfig.api_key = 't8h-CrL1R7xapQB6TbhB'


# Difference in consecutive prices
def hist(FC: pandas.DataFrame):
    FC['hist'] = 0.0
    for indr in range(FC.shape[0]):
        try:
            FC['hist'][indr] = math.log(FC['Price'][indr]) - math.log(FC['Price'][indr - 1])
        except KeyError:
            FC['hist'][indr] = math.log(FC['Price'][indr])
    return FC


# Difference in mean and standard deviation of prices
def mov_mean(data: pandas.DataFrame):
    data['avg'] = 0.0
    data['std'] = 0.0
    data['weeks'] = ''
    for indr in reversed(range(data.shape[0])):
        data['weeks'][indr] = str(data.shape[0] - indr)
        data['avg'][indr] = data['Price'][indr - data.shape[0]:].mean()
        data['std'][indr] = data['Price'][indr - data.shape[0]:].std()
    return data


# User input
def get_user_input():
    while 1:
        try:
            model = input('Enter GBM or GOU or Both: ')
            return model.upper()
        except KeyError:
            print('Invalid input!')
            continue


class Demo:
    def __init__(self, mode_: str, data: pandas.DataFrame, dr1: datetime, dr2: datetime,  st1: datetime, st2: datetime, fc1: datetime, fc2: datetime):
        self.forecast_data = pandas.DataFrame()
        self.mode = mode_
        self.data = data
        self.oil = data
        self.date = st1
        self.dr1 = dr1
        self.dr2 = dr2
        self.st1 = st1
        self.st2 = st2
        self.fc1 = fc1
        self.fc2 = fc2
        self.pre_process()
        # Difference of forecast date and calibration date
        self.diff = int((fc_2.year - self.date.year) + (fc_2.month - self.date.month) / 12 + (fc_2.day - self.date.day) / 365) * 52
        self.delta_t = 1 / 52

    def pre_process(self):
        self.data = self.data.reset_index()
        self.data['Price'] = self.data['Value']
        self.data = self.data.drop(['Value'], axis=1)
        self.data = self.data.loc[self.data['Date'] >= self.st1].reset_index().drop(['index'], axis=1)
        self.data = self.data.loc[self.data['Date'] <= self.st2]
        self.data['Week'] = self.data['Date'].index
        self.data['Years'] = self.data['Week'] / 52
        self.date = self.data['Date'][-1:].reset_index()['Date'][0] + timedelta(weeks=-1)

    def variables(self):
        """
        Function used for calibration of the model

        Parameters (All parameters in this function are float type)
        -----------------------------------------------------------
        mean : Mean
        std : Standard Deviation
        pv : Percentage Volatility
        pd : Percentage Drift
        lt_mean : Long-term Mean
        :return:
        """
        if self.mode == 'GBM':
            self.data['xt'] = 0.0
            for indr in range(self.data.shape[0]):
                try:
                    self.data['xt'][indr] = math.log(self.data['Price'][indr] / self.data['Price'][indr - 1])
                except KeyError:
                    continue
            mean = sum(self.data['xt']) / len(self.data['xt'])
            std = tstd(self.data['xt'])
            pv = std / math.sqrt(self.delta_t)
            pd = mean / self.delta_t + pv ** 2 / 2
            return mean, std, pv, pd
        if self.mode == 'GOU':
            self.data['x'] = 0.0
            self.data['y'] = 0.0
            num = 0
            for indr in range(self.data.shape[0]):
                num += 1
                self.data['x'][indr] = math.log(self.data['Price'][indr])
                try:
                    self.data['y'][indr] = math.log(self.data['Price'][indr + 1])
                except KeyError:
                    continue
            fitted_line = np.polyfit(self.data['x'][1:self.data.shape[0] - 1], self.data['y'][1:self.data.shape[0] - 1], deg=1)
            self.data['y_reg'] = 0.0
            self.data['residual'] = 0.0
            for indr in range(self.data.shape[0]):
                self.data['y_reg'][indr] = fitted_line[0] * self.data['y'][indr] + fitted_line[1]
                try:
                    self.data['residual'][indr] = self.data['y'][indr + 1] - self.data['y_reg'][indr]
                except KeyError:
                    continue
            sd_residual = tstd(self.data['residual'][1:self.data.shape[0] - 2])
            lt_mean = fitted_line[1] / (1 - fitted_line[0])
            speed = -math.log(fitted_line[0]) / self.delta_t
            volatility = sd_residual * math.sqrt(-2.0 * math.log(fitted_line[0]) / self.delta_t / (1 - fitted_line[0] ** 2))
            return lt_mean, speed, volatility

    def mean_sd(self, pd: float = 0.0, pv: float = 0.0, volatility: float = 0.0, speed: float = 0.0, lt_mean: float = 0.0):
        if self.mode == 'GBM':
            self.data['mean'] = 0.0
            self.data['sd'] = 0.0
            for indr in range(self.data.shape[0]):
                self.data['mean'][indr] = self.data['Price'][0] * math.exp(pd * self.data['Years'][indr])
                self.data['sd'][indr] = math.sqrt(self.data['Price'][0] ** 2 * math.exp(2 * pd * self.data['Years'][indr]) * (math.exp(pv ** 2 * self.data['Years'][indr]) - 1))
        elif self.mode == 'GOU':
            self.data['mean_lnP'] = 0.0
            self.data['sd_lnP'] = 0.0
            self.data['mean'] = 0.0
            self.data['sd'] = 0.0
            for indr in range(self.data.shape[0]):
                self.data['mean_lnP'][indr] = math.log(self.data['Price'][0]) * math.exp(-speed * self.data['Years'][indr]) + lt_mean * (1 - math.exp(-speed * self.data['Years'][indr]))
                self.data['sd_lnP'][indr] = math.sqrt(volatility ** 2 / 2 / speed * (1 - math.exp(-2 * speed * self.data['Years'][indr])))
                self.data['mean'][indr] = math.exp(self.data['mean_lnP'][indr] + (self.data['sd_lnP'][indr] ** 2) / 2.0)
                self.data['sd'][indr] = math.sqrt(math.exp(2 * self.data['mean_lnP'][indr] + self.data['sd_lnP'][indr] ** 2) * (math.exp(self.data['sd_lnP'][indr] ** 2) - 1))

    def forecast(self, pd: float = 0, pv: float = 0, volatility: float = 0, speed: float = 0, lt_mean: float = 0):
        """
         Function used for forecast of the model

        Parameters (All parameters in this function are float type)
        -----------------------------------------------------------
        years_fc : forecast years
        years_fs : forecast years from start date
        mean_f : mean of Forecast price
        sd_f : standard deviation of forecast price
        mean_lnPt : mean of the natural log of oil price
        sd_lnPt : standard deviation of the natural log of oil price
        P10, P90 : P90 and P10 are low and high estimates respectively
        dt : time increment, most commonly a week
        wt, w : wiener process,  a normal distribution with zero mean and standard deviation
        Pt : oil price at Year t
        lnP : logarithm of Forecasted oil price

        :return:
        """
        date = self.date
        if self.mode == 'GBM':
            Price = self.data['Price'][self.data.shape[0] - 1]
            Year = self.data['Years'][self.data.shape[0] - 1]
            cols = ['years_fc', 'years_fs', 'mean_f', 'sd_f', 'mean_lnPt', 'sd_lnPt', 'P10', 'P90', 'dt', 'wt', 'Pt']
            FC = pandas.DataFrame(columns=cols)
            num = -1
            FC['Date_fc'] = date
            for indr in range(self.diff):
                FC = FC.append(pandas.Series([0.0] * len(cols), index=cols), ignore_index=True)
                date += timedelta(weeks=1)
                FC['Date_fc'][indr] = date
                rnd = random.random()
                FC['years_fc'][indr] = (num + 1) / 52
                FC['years_fs'][indr] = FC['years_fc'][indr] + Year
                FC['mean_f'][indr] = Price * math.exp(pd * FC['years_fc'][indr])
                FC['sd_f'][indr] = math.sqrt(Price ** 2 * math.exp(2 * pd * FC['years_fc'][indr]) * (math.exp(pv ** 2 * FC['years_fc'][indr]) - 1))
                FC['mean_lnPt'][indr] = math.log(Price) + (pd - pv ** 2 / 2) * FC['years_fc'][indr]
                FC['sd_lnPt'][indr] = pv * math.sqrt(FC['years_fc'][indr])
                FC['P10'][indr] = math.exp(FC['mean_lnPt'][indr] - 1.28 * FC['sd_lnPt'][indr])
                FC['P90'][indr] = math.exp(FC['mean_lnPt'][indr] + 1.28 * FC['sd_lnPt'][indr])
                if indr != 0:
                    FC['dt'][indr] = FC['years_fc'][indr] - FC['years_fc'][indr - 1]
                else:
                    FC['dt'][indr] = 0.0
                if indr != 0:
                    FC['wt'][indr] = norm.ppf(rnd, 0.0, math.sqrt(FC['dt'][indr]))
                else:
                    FC['wt'][indr] = 0.0
                if indr != 0:
                    FC['Pt'][indr] = FC['Pt'][indr - 1] * math.exp((pd - pv ** 2 / 2) * FC['dt'][indr] + pv * FC['wt'][indr])
                else:
                    FC['Pt'][indr] = Price
                num += 1
            self.forecast_data = FC.copy()
        if self.mode == 'GOU':
            Price = self.data['Price'][self.data.shape[0] - 1]
            Year = self.data['Years'][self.data.shape[0] - 1]
            cols = ['years_fc', 'years_fs', 'mean_f', 'sd_f', 'mean_lnPt',
                    'sd_lnPt', 'P10', 'P90', 'dt', 'w', 'lnP', 'Pt']
            FC = pandas.DataFrame(columns=cols)
            num = -1
            FC['Date_fc'] = date
            for indr in range(self.diff):
                FC = FC.append(pandas.Series([0.0] * len(cols), index=cols), ignore_index=True)
                date += timedelta(weeks=1)
                FC['Date_fc'][indr] = date
                rnd = random.random()
                FC['years_fc'][indr] = (num + 1) / 52
                FC['years_fs'][indr] = FC['years_fc'][indr] + Year
                FC['mean_lnPt'][indr] = math.log(Price) * math.exp(-speed * FC['years_fc'][indr]) + lt_mean * (1 - math.exp(-speed * FC['years_fc'][indr]))
                FC['sd_lnPt'][indr] = math.sqrt((volatility ** 2 / (2 * speed)) * (1 - math.exp(-speed * FC['years_fc'][indr])))
                FC['mean_f'][indr] = math.exp(FC['mean_lnPt'][indr] + FC['sd_lnPt'][indr] ** 2 / 2)
                FC['sd_f'][indr] = math.sqrt(math.exp(2 * FC['mean_lnPt'][indr] + FC['sd_lnPt'][indr] ** 2) * (math.exp(FC['sd_lnPt'][indr] ** 2) - 1))
                FC['P10'][indr] = math.exp(FC['mean_lnPt'][indr] - 1.28 * FC['sd_lnPt'][indr])
                FC['P90'][indr] = math.exp(FC['mean_lnPt'][indr] + 1.28 * FC['sd_lnPt'][indr])
                if indr != 0:
                    FC['dt'][indr] = FC['years_fc'][indr] - FC['years_fc'][indr - 1]
                else:
                    FC['dt'][indr] = 0.0
                if indr != 0:
                    FC['w'][indr] = norm.ppf(rnd, 0.0, math.sqrt(math.exp(2 * speed * FC['dt'][indr]) - 1))
                else:
                    FC['w'][indr] = 0.0
                if indr != 0:
                    FC['lnP'][indr] = FC['lnP'][indr - 1] * math.exp(-speed * FC['dt'][indr]) + lt_mean * (1 - math.exp(-speed * FC['dt'][indr])) + volatility / math.sqrt(2 * speed) * math.exp(
                        -speed * FC['dt'][indr]) * FC['w'][indr]
                else:
                    FC['lnP'][indr] = math.log(Price)
                FC['Pt'][indr] = math.exp(FC['lnP'][indr])
                num += 1
            self.forecast_data = FC.copy()

    def auto_proc(self):
        if self.mode == 'GBM':
            mean, std, pv, pd = self.variables()
            self.mean_sd(pd=pd, pv=pv)
            self.forecast(pd=pd, pv=pv)
        if self.mode == 'GOU':
            lt_mean, speed, volatility = self.variables()
            self.mean_sd(volatility=volatility, speed=speed, lt_mean=lt_mean)
            self.forecast(volatility=volatility, speed=speed, lt_mean=lt_mean)
        self.data = hist(self.data)
        self.data = mov_mean(self.data)
        return self.data, self.forecast_data


def plots(data: pandas.DataFrame, forecast_data: pandas.DataFrame):
    plot1 = plt.figure(1)
    plt.plot(data['Date'], data['Price'], label='DATA')
    plt.plot(data['Date'], data['mean'], '--', label='Mean')
    plt.plot(data['Date'], data['sd'], label='STD')
    plt.title('Data range to be calibrated')
    plt.xlabel('Years')
    plt.ylabel('Price ($/bbl)')
    plt.grid()
    plt.legend()

    plot2 = plt.figure(2)
    plt.plot(data['Date'], data['Price'], label='Calibration data')
    plt.plot(forecast_data['Date_fc'], forecast_data['mean_f'], label='Mean')
    plt.plot(forecast_data['Date_fc'], forecast_data['P10'], '--', label='P10')
    plt.plot(forecast_data['Date_fc'], forecast_data['P90'], '--', label='P90')
    plt.plot(forecast_data['Date_fc'], forecast_data['Pt'], label='1 realization')
    plt.xlabel('Years')
    plt.ylabel('Price ($/bbl)')
    plt.grid()
    plt.legend()

    plot3 = plt.figure(3)
    plt.hist(nl(data['hist']), density=True, bins=30)
    plt.xlabel('Ln(P(t)) - Ln(P(t-1))')
    plt.title('Histogram of Logarithm of Price Changes')
    plt.ylabel('Number')
    plt.grid()

    data.plot(x='weeks', y=['avg', 'std'], kind='line')
    plt.title('Change in mean and standard deviation of price changes as a function of number of weeks')
    plt.xlabel('number of weeks of data used')
    plt.ylabel('Mean and SD of Changes')
    plt.grid()
    plt.legend()


def both_plots(gbm: Demo, gou: Demo):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(gbm.data['Date'], gbm.data['Price'])
    ax2.plot(gou.data['Date'], gou.data['Price'])


def mp_worker(obj):
    a, b = obj.auto_proc()
    return a, b


if __name__ == "__main__":
    method = get_user_input()
    dr = input("Enter the historical data range: ").replace(' ', '')
    dr_1 = datetime.strptime(dr.split('-')[0], '%d/%m/%Y')
    dr_2 = datetime.strptime(dr.split('-')[1], '%d/%m/%Y')
    st = input("Enter the range for calibration: ").replace(' ', '')
    st_1 = datetime.strptime(st.split('-')[0], '%d/%m/%Y')
    st_2 = datetime.strptime(st.split('-')[1], '%d/%m/%Y')
    fc = input("Enter the range for forecast data: ").replace(' ', '')
    fc_1 = datetime.strptime(fc.split('-')[0], '%d/%m/%Y')
    fc_2 = datetime.strptime(fc.split('-')[1], '%d/%m/%Y')
    oil = pandas.DataFrame(quandl.get("FRED/DCOILWTICO", start_date=dr_1, end_date=dr_2, collapse="weekly"))
    obj1 = Demo('GBM', oil, dr_1, dr_2, st_1, st_2, fc_1, fc_2)
    obj2 = Demo('GOU', oil, dr_1, dr_2, st_1, st_2, fc_1, fc_2)
    objs = [(obj1,), (obj2,)]

    if method == 'BOTH':
        pool = Pool(processes=2)
        results = pool.starmap(mp_worker, objs)
        pool.close()

        obj1.data = results[0][0]
        obj1.forecast_data = results[0][1]
        obj2.data = results[1][0]
        obj2.forecast_data = results[1][1]

        plot1 = plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(obj1.data['Date'], obj1.data['Price'], label='DATA')
        plt.plot(obj1.data['Date'], obj1.data['mean'], label='GBM Mean')
        plt.plot(obj1.data['Date'], obj1.data['sd'], label='GBM SD')
        plt.ylabel('Price ($/bbl)')
        plt.grid()
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(obj2.data['Date'], obj2.data['Price'], label='DATA')
        plt.plot(obj2.data['Date'], obj2.data['mean'], label='GOU Mean')
        plt.plot(obj2.data['Date'], obj2.data['sd'], label='GOU SD')
        plt.xlabel('Years')
        plt.ylabel('Price ($/bbl)')
        plt.grid()
        plt.legend()

        plot2 = plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(obj1.data['Date'], obj1.data['Price'], label='DATA')
        plt.plot(obj1.forecast_data['Date_fc'], obj1.forecast_data['mean_f'], label='GBM Mean')
        plt.plot(obj1.forecast_data['Date_fc'], obj1.forecast_data['P10'], '--', label='P10')
        plt.plot(obj1.forecast_data['Date_fc'], obj1.forecast_data['P90'], '--', label='P90')
        plt.plot(obj1.forecast_data['Date_fc'], obj1.forecast_data['Pt'], label='1 realization')
        plt.ylabel('Price ($/bbl)')
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(obj2.data['Date'], obj2.data['Price'], label='DATA')
        plt.plot(obj2.forecast_data['Date_fc'], obj2.forecast_data['mean_f'], label='GOU Mean')
        plt.plot(obj2.forecast_data['Date_fc'], obj2.forecast_data['P10'], '--', label='P10')
        plt.plot(obj2.forecast_data['Date_fc'], obj2.forecast_data['P90'], '--', label='P90')
        plt.plot(obj2.forecast_data['Date_fc'], obj2.forecast_data['Pt'], label='1 realization')
        plt.xlabel('Years')
        plt.ylabel('Price ($/bbl)')
        plt.grid()
        plt.legend()

        plot3 = plt.figure(3)
        plt.plot(obj1.data.Date, obj1.data['Price'], 'g', label='DATA')
        plt.plot(obj1.forecast_data.Date_fc, obj1.forecast_data['mean_f'], 'r', label='GBM Mean')
        plt.plot(obj1.forecast_data.Date_fc, obj1.forecast_data['P10'], '--r', label='P10')
        plt.plot(obj1.forecast_data.Date_fc, obj1.forecast_data['P90'], '--r', label='P90')
        plt.plot(obj2.forecast_data.Date_fc, obj2.forecast_data['mean_f'], 'b', label='GOU Mean')
        plt.plot(obj2.forecast_data.Date_fc, obj2.forecast_data['P10'], '--b', label='P10')
        plt.plot(obj2.forecast_data.Date_fc, obj2.forecast_data['P90'], '--b', label='P90')
        plt.xlabel('Years')
        plt.ylabel('Price ($/bbl)')
        plt.grid()
        plt.legend()

    if method == 'GBM':
        obj1.auto_proc()
        plots(obj1.data, obj1.forecast_data)
    if method == 'GOU':
        obj2.auto_proc()
        plots(obj2.data, obj2.forecast_data)

    plot5 = plt.figure(5)
    oil = oil.reset_index()
    plt.plot(oil.Date,oil['Value'])
    plt.xlabel('Years')
    plt.ylabel('Price ($/bbl)')
    plt.title('Historical Data Range')
    plt.grid()
    plt.legend()
    plt.show()