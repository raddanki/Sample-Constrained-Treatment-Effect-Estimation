# Sample-Constrained-Treatment-Effect-Estimation
Code accompanying our publication on Sample Constrained Treatment Effect Estimation that appeared in the Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS 2022). 


Instructions to run the code:

To generate the plots and run the code, run main.py in src/ folder.

The only parameters to change are ntrials, which indicates the number of times to run the estimators and we plot the mean values.

For Individual Treatment Effect, the relevant files are run_ITE.py and ITE.py.

ITE.py contains all the estimators for ITE estimation and run_ITE has functions to aggregate the estimators with appropriate number of trials. 

For Average Treatment Effect, the relevant files are run_ATE.py and ATE.py.

In ATE.py we use the implementation of Gram-Schmidt-Walk as given by Harshaw et al (2019). This requires installation of Julia. You can comment out this part in main.py file if it fails to run. More instructions are available here: https://github.com/crharshaw/GSWDesign.jl

The data generating mechanism is implemented in data.py. We have collected the datasets in the folder -- datasets/

Plotting functions are available in custom_plot.py. All the plots generated will appear in the folder -- plots/
