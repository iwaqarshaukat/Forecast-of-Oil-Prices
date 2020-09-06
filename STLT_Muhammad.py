"""
Created on Tue Aug 18 19:05:55 2020

@author: Muhammad Waqar
"""

# Importing Libraries
import pandas as pd
import math
import numpy
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# config
pd.options.mode.chained_assignment = None


# User Input
def get_user_input():
    while 1:
        try:
            model = input('Enter STLT or GBM: ')
            return model.upper()
        except KeyError:
            print('Invalid input!')
            continue


class STLT:
    def __init__(self, mode_: str, data1: pd.DataFrame, data2: pd.DataFrame, x_o, sigma_x, kappa, lambda_x, xi_o, sigma_xi, mu_xi, lambda_xi, rho_xi):
        self.mode = mode_
        self.data = data1
        self.vol = data1
        self.future = data2
        self.x_o = x_o
        self.sigma_x = sigma_x
        self.kappa = kappa
        self.lambda_x = lambda_x
        self.xi_o = xi_o
        self.sigma_xi = sigma_xi
        self.mu_xi = mu_xi
        self.lambda_xi = lambda_xi
        self.rho_xi = rho_xi

    def min_ls(self, obs_v, dis_rate, row):
        """
        Parameters (All parameters in this function are float type)
        -----------------------------------------------------------
        fto : Observed Futures
        k : Strike Price
        obs_v : Observed Volatility
        calc_call : Calculated Call Option Price
        dis_rate : Discount rate
        mismatch: difference between observed and calculated call option price
        :return:
        """
        self.data['d'][row] = math.log(self.data['fto'][row] / self.data['k'][row]) / obs_v + 0.5 * obs_v
        self.data['calc_call'][row] = math.exp(-dis_rate * self.data['dt'][row]) * (self.data['fto'][row] * norm.cdf(self.data['d'][row], 0, 1) - self.data['k'][row] * norm.cdf(self.data['d'][row] - obs_v, 0, 1))
        self.data['mismatch'][row] = (self.data['calc_call'][row] - self.data['obs_call'][row]) ** 2
        return self.data['mismatch'][row]

    # Optimization Function
    def do_optim(self, dis_rate=0.03):
        self.data['d'] = 0.0
        self.data['calc_call'] = 0.0
        self.data['mismatch'] = 0.0
        self.data['obs_v'] = 0.1
        for row in range(self.data.shape[0]):
            # Minimization
            res = minimize(self.min_ls, self.data['obs_v'][row], args=(dis_rate, row))
            self.data['obs_v'][row] = res.x
        return self.data

    def stlt_cal(self):
        """
        Function used for forecast of the models
        Parameters (All parameters in this function are float type)
        -----------------------------------------------------------
        ln_futures : Logarithm of observed futures
        futures: Modeled logarithm futures
        m_futures : Modeled futures
        sqrd_error, sqrd_err : Difference between modeled and observed value
        obs_vo : Observed annualized Volatility
        mod_v : Modeled Volatility
        mod_vol : Modeled annualized Volume
        :return:
        """
        global params
        self.future['ln_futures'] = 0.1
        self.future['futures'] = 0.1
        self.future['m_futures'] = 0.1
        self.future['sqrd_error'] = 0.1
        self.vol['obs_vo'] = 0.1
        self.vol['mod_v'] = 0.1
        self.vol['mod_vol'] = 0.1
        self.vol['sqrd_err'] = 0.1
        if self.mode == 'STLT':
            params = [self.x_o, self.sigma_x, self.kappa, self.lambda_x, self.xi_o, self.sigma_xi, self.mu_xi, self.lambda_xi, self.rho_xi]
            res = minimize(self.stlt_f, numpy.array(params), method='BFGS')
            params = res.x
        if self.mode == 'GBM':
            params = [self.sigma_xi, self.mu_xi, self.lambda_xi]
            res = minimize(self.stlt_f, numpy.array(params), method='BFGS')
            params = res.x
        return params, self.future, self.vol

    def stlt_f(self, params: list):
        for indr in range(self.future.shape[0]):
            self.future['ln_futures'][indr] = math.log(self.future['fto'][indr])
            if self.mode == 'STLT':
                try:
                    self.future['futures'][indr] = math.exp(-params[2] * self.future['dt'][indr]) * params[0] - (1 - math.exp(-params[2] * self.future['dt'][indr])) * params[3] / params[2] + params[4] + (params[6] - params[7]) * self.future['dt'][indr] + (1 - math.exp(-2 * params[2] * self.future['dt'][indr])) * (params[1] ** 2) / 4.0 / params[2] + (params[5] ** 2) / 2 * self.future['dt'][indr] + (1 - math.exp(-params[2] * self.future['dt'][indr])) * params[8] * params[1] * params[5] / params[2]
                except KeyError:
                    print("error")
            if self.mode == 'GBM':
                try:
                    self.future['futures'][indr] = params[0] + (params[2] - params[3]) * self.future['dt'][indr] + (params[1] ** 2) / 2.0 * self.future['dt'][indr]
                except KeyError:
                    print("error")
            self.future['m_futures'][indr] = math.exp(self.future['futures'][indr])
            self.future['sqrd_error'][indr] = (self.future['futures'][indr] - self.future['ln_futures'][indr]) ** 2
        for ind in range(self.data.shape[0]):
            self.vol['obs_vo'][ind] = self.data['obs_v'][ind] / math.sqrt(self.data['dt'][ind])
            if self.mode == 'STLT':
                try:
                    self.vol['mod_v'][ind] = math.sqrt((1 - math.exp(-2.0 * params[2] * self.data['dt'][ind])) * (params[1] ** 2) / 2 / params[2] + (params[5] ** 2) * self.data['dt'][ind] + 2 * (1 - math.exp(-params[2] * self.data['dt'][ind])) * params[8] * params[1] * params[5] / params[2])
                except KeyError:
                    print("error")
            if self.mode == 'GBM':
                try:
                    self.vol['mod_v'][ind] = math.sqrt((params[0] ** 2) * self.data['dt'][ind])
                except KeyError:
                    print("error")
            self.vol['mod_vol'][ind] = self.vol['mod_v'][ind] / math.sqrt(self.data['dt'][ind])
            self.vol['sqrd_err'][ind] = (self.vol['mod_v'][ind] - self.data['obs_v'][ind]) ** 2
        ln_f = sum(self.future['sqrd_error'])
        i_v = sum(self.vol['sqrd_err'])
        wi_v = i_v * 1.0
        total_err = ln_f + wi_v
        return total_err

    def auto_pro(self):
        self.do_optim()
        self.stlt_cal()
        return self.data, self.future, self.vol


def plots(data: pd.DataFrame, future: pd.DataFrame, vol: pd.DataFrame):
    plot1 = plt.figure(1)
    plt.plot(future['dt'], future['fto'], '--', label='Observed')
    plt.plot(future['dt'], future['m_futures'], label='Modeled')
    plt.title('Futures')
    plt.xlabel('dt(years)')
    plt.ylabel('Futures Price (USD/bbl)')
    plt.grid()
    plt.legend()

    plot2 = plt.figure(2)
    plt.plot(data['dt'], vol['obs_vo'], '--', label='Observed')
    plt.plot(data['dt'], vol['mod_vol'], label='Modeled')
    plt.title('Volatility')
    plt.xlabel('dt(years)')
    plt.ylabel('Annualized Volatility')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    method = get_user_input()
    data1 = pd.read_excel("STLT Calibration Using Implied Volatility.xlsx", sheet_name='Modeled Volatility')
    data2 = pd.read_excel("STLT Calibration Using Implied Volatility.xlsx", sheet_name='Future Prices')
    obj = STLT(method, data1.copy(), data2.copy(), x_o=-0.2, sigma_x=0.4, kappa=0.3, lambda_x=-0.02, xi_o=4, sigma_xi=0.02, mu_xi=-0.02, lambda_xi=0.003, rho_xi=0.3)
    obj.auto_pro()
    plots(obj.data, obj.future, obj.vol)
