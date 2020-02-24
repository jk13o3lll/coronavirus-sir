import numpy as np
import pandas as pd
import math
from datetime import datetime
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib import rc
rc('mathtext', default='regular')

data = pd.read_csv('Updates_NC.csv')

# preprocess1
wuhan = data[data['城市'] == '武汉市']
wuhanLivingPopulation = 11081000 # https://www.hongheiku.com/shijirenkou/801.html
# wuhanInfection = wuhan.groupby('报道时间')['新增确诊'].sum()
# wuhanRecovered = wuhan.groupby('报道时间')['新增出院'].sum()
# wuhanDead = wuhan.groupby('报道时间')['新增死亡'].sum()
wuhanInfection = wuhan.groupby('报道时间')['新增确诊'].max()
wuhanRecovered = wuhan.groupby('报道时间')['新增出院'].max()
wuhanDead = wuhan.groupby('报道时间')['新增死亡'].max()
date = [datetime.strptime(datestr, '%m月%d日') for datestr in wuhanInfection.index]
date2 = [d.replace(year=2020) for d in date]
wuhan = {
    'datetime': date2,
    '报道时间': wuhanInfection.index,
    '新增确诊': wuhanInfection.values,
    '新增出院': wuhanRecovered.values,
    '新增死亡': wuhanDead.values
}
wuhan = pd.DataFrame(wuhan, index = [i for i in range(wuhanInfection.shape[0])])
wuhan = wuhan[wuhan['报道时间'] >= '1月18日'] # remove noise data
wuhan.sort_values(by=['datetime'], inplace=True)
print(wuhan)
# print((wuhan['datetime'].values[1:] - wuhan['datetime'].values[:-1]) / 86400e9)

# preprocess2
accInfection = np.cumsum(wuhan['新增确诊'].values) # 累計確診
accRecovered = np.cumsum(wuhan['新增出院'].values) # 累計出院
accDead = np.cumsum(wuhan['新增死亡'].values)      # 累計死亡
N = wuhanLivingPopulation
R = accRecovered + accDead
I = accInfection - R
S = N - I - R

# plot
dates = matplotlib.dates.date2num(wuhan['datetime'].values)
t = np.insert(np.cumsum(dates[1:] - dates[:-1]), 0, 0)
fig, ax = plt.subplots()
lR = ax.plot_date(dates, R, 'r.', label='R')
lI = ax.plot_date(dates, I, 'r+', label='I')
ax2 = ax.twinx()
lS = ax2.plot_date(dates, S, 'bx', label='S')
min1, max1 = min(R.min(), I.min()), max(R.max(), I.max())
min2, max2 = S.min(), S.max()
ax.set_ylim([1.1*min1-0.1*max1, 1.3*max1-0.3*min1])
ax2.set_ylim([1.1*min2-0.1*max2, 1.3*max2-0.3*min2])
ax.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')
ax.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
fig.autofmt_xdate()
fig.tight_layout()
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

# RK45 to simluate
tf = int(wuhan['datetime'].values[-1] - wuhan['datetime'].values[0]) / 86400e9
y0 = np.array([S[0], I[0], R[0]], dtype=np.float64)
t_span = (0.0, tf) # 100 days
t_eval = np.arange(t_span[0], t_span[1], 0.01)
ax = plt.subplot(111)
ax2 = ax.twinx()
ax.plot(t, R, '-.', color=(1, 0.8, 0.8), label='R0')
ax.plot(t, I, '-', color=(1, 0.8, 0.8), label='I0')
ax2.plot(t, S, '-', color=(0.8, 0.8, 1), label='S0')
lR, = ax.plot(t_eval, np.zeros(t_eval.shape[0], dtype=np.float64), 'r-.', label='R')
lI, = ax.plot(t_eval, np.zeros(t_eval.shape[0], dtype=np.float64), 'r-', label='I')
lS, = ax2.plot(t_eval, np.zeros(t_eval.shape[0], dtype=np.float64), 'b-', label='S')
min10, max10 = min(R.min(), I.min()), max(R.max(), I.max())
min20, max20 = S.min(), S.max()
ax.set_ylim([1.1*min10-0.1*max10, 1.3*max10-0.3*min10])
ax2.set_ylim([1.1*min20-0.1*max20, 1.3*max20-0.3*min20])
# ax.set_ylim([0, N])
# ax2.set_ylim([0, N])
ax.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')
ax.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
for beta in np.arange(0.3, 0.9, 0.01): # 1000
    # for gamma in 1 / np.arange(5, 20, 0.1): # 1000
    for gamma in np.arange(0.3, 0.9, 0.01): # 1000
        # define model
        def sir(t, y):
            dydt = np.empty(3, dtype=np.float64)
            dydt[0] = -beta / N * y[1] * y[0]; # dS/dt
            dydt[2] = gamma * y[1] # dR/dt
            dydt[1] = -dydt[0] - dydt[2] # dI/dt
            return dydt
        # simluate
        sol = solve_ivp(sir, t_span, y0, method='RK45', t_eval=t_eval)
        # sol = solve_ivp(sir, t_span, y0, method='Radau', t_eval=t_eval)
        # sol = solve_ivp(sir, t_span, y0, method='BDF', t_eval=t_eval)
        # sol = solve_ivp(sir, t_span, y0, method='LSODA', t_eval=t_eval)
        # plot
        lS.set_ydata(sol.y[0])
        lI.set_ydata(sol.y[1])
        lR.set_ydata(sol.y[2])
        # min1, max1 = min(sol.y[2].min(), sol.y[1].min()), max(sol.y[2].max(), sol.y[1].max())
        # min2, max2 = sol.y[0].min(), sol.y[0].max()
        # min1, max1 = min(min10, min1), max(max10, max1)
        # min2, max2 = min(min20, min2), max(max20, max2)
        # ax.set_ylim([1.1*min1-0.1*max1, 1.3*max1-0.3*min1])
        # ax2.set_ylim([1.1*min2-0.1*max2, 1.3*max2-0.3*min2])
        plt.draw()
        plt.pause(0.001)
        print('%.3f\t%.3f' % (beta, gamma))

# start to drop until almost all infected