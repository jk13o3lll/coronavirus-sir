# coronavirus-sir
Small project for using coronavirus epidemic data and SIR model to simluate.

## SIR model

Wiki:
[https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)

SIR model is simulated by ode45.
We test several beta and gamma within a specific range.

This code mainly based on NumPy, SciPy, Pandas, and Matplotlib.

## Dataset

疫情數據由澎派新聞美數課整理提供:
[https://github.com/839Studio/Novel-Coronavirus-Updates](https://github.com/839Studio/Novel-Coronavirus-Updates)

## Results

The infected numbers start to drop until mosts of population are infected.

![preprocessed_data](figures/preprocessed_data.png)
![preprocessed_data_plot](figures/preprocessed_data_plot.png)
![beta_0_3_gamma_0_3](figures/beta_0_3_gamma_0_3.png)
![beta_0_7_gamma_0_4](figures/beta_0_7_gamma_0_4.png)
![beta_0_8_gamma_0_3](figures/beta_0_8_gamma_0_3.png)
![beta_0_8_gamma_0_3_fullscale](figures/beta_0_8_gamma_0_3_fullscale.png)

## Contact

ccwang.jack@gmail.com
