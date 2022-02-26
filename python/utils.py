import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_posterior(data, figsize=(30, 15)):
    
    n_samples = len(data)
    B = len(list(data.values())[0])
    v_samples = list(data.keys())
    
    params = list(data[v_samples[0]][0].samples.keys())
    params = [x for x in params if x not in  ['mu_pred', 'y_pred', 'lp__']]
    n_params = len(params)
    
    mcmc_samples = len(data[v_samples[0]][0].samples[params[0]])
    
    # compute mean posterior across repetitions (B) within each sample size
    parms_avg = {x:{y:None for y in params} for x in v_samples}
    parms_true = {x:{y:None for y in params} for x in v_samples}
    parm_ranges_t = {y:[] for y in params}
    parm_ranges = {y:None for y in params}
    
    for i in range(n_samples):
        for k in range(n_params):
            res = np.zeros(mcmc_samples)
            res_true = 0.
            for j in range(B):
                res += data[v_samples[i]][j].samples[params[k]].ravel()
                res_true += data[v_samples[i]][j].__dict__[params[k]]
            parms_avg[v_samples[i]][params[k]] = res/B
            parms_true[v_samples[i]][params[k]] = res_true/B
            
    min_max = lambda x: (min(x), max(x))
    
    v = list(parms_avg.values())

    for j in params:
        for i in range(len(v)):
            min_param = min(v[i][j])
            max_param = max(v[i][j])
            parm_ranges_t[j].append(min_param)
            parm_ranges_t[j].append(max_param)  
            parm_ranges_t[j].append(getattr(data[v_samples[0]][0], j))
        parm_ranges[j]=(min_max(parm_ranges_t[j]))
    
    if n_params > 3:
        title = 'Heckman'
        fig, axs = plt.subplots(len(data),6, figsize=figsize, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.5)
        fig.suptitle(title, fontsize=16)
       
    else:
        title = 'OLS'
        fig, axs = plt.subplots(len(data),3, figsize=figsize, facecolor='w', edgecolor='k')
        #fig.subplots_adjust(hspace = .5, wspace=.5)
        fig.suptitle(title, fontsize=20)

    axs = axs.ravel()
    
    axs = axs.ravel()
        
    p = 0
    for i in v_samples:
        for j in params:
            sns.histplot(ax = axs[p], data=parms_avg[i][j], stat="density") 
            axs[p].set_xlim(parm_ranges[j])
            axs[p].set(title=j +'; N='+ i)
            axs[p].axvline(np.quantile(parms_avg[i][j], .05), color='g', linestyle='--', label='90% intervals')
            axs[p].axvline(np.quantile(parms_avg[i][j], .95), color='g', linestyle='--')
            axs[p].axvline(parms_true[i][j], color='r', label='Actual')
            axs[p].legend(loc="upper right", fontsize='large')
            p+=1
        

def plot_mu_posterior(heckman_fit, ols_fit):
    
    n_samples = len(heckman_fit)
    B = len(list(heckman_fit.values())[0])
    v_samples = list(heckman_fit.keys())
        
    params = ['mu_pred', 'y_pred']
    
    mcmc_samples = heckman_fit[v_samples[0]][0].samples[params[0]].shape[0]
    
    post_avg_heckman = {x:{y:None for y in params} for x in v_samples}
    post_avg_ols = {x:{y:None for y in params} for x in v_samples}
    
    for i in range(n_samples):
        for k in range(len(params)):
            res = np.zeros(shape=(mcmc_samples, 
                          heckman_fit[v_samples[0]][0].x_new.shape[0]))
            for j in range(B):
                res += heckman_fit[v_samples[i]][j].samples[params[k]]
            post_avg_heckman[v_samples[i]][params[k]] = res/B
            
    for i in range(n_samples):
        for k in range(len(params)):
            res = np.zeros(shape=(mcmc_samples, 
                           ols_fit[v_samples[0]][0].x_new.shape[0]))
            for j in range(B):
                res += ols_fit[v_samples[i]][j].samples[params[k]]
            post_avg_ols[v_samples[i]][params[k]] = res/B
    
    
    for v in v_samples:
        mu_pred0 = post_avg_heckman[v]['mu_pred']
        mu_pred_m0 = np.mean(mu_pred0, axis=0)
        mu_pred_ci0 = np.quantile(mu_pred0, [0.05, 0.95], axis=0).T
        x0 = heckman_fit[v_samples[0]][0].x_new.ravel()
    
        mu_pred1 = post_avg_ols[v]['mu_pred']
        mu_pred_m1 = np.mean(mu_pred1, axis=0)
        mu_pred_ci1 = np.quantile(mu_pred1, [0.05, 0.95], axis=0).T
        x1 = ols_fit[v_samples[0]][0].x_new.ravel()
    
    
        fig, axs = plt.subplots(1,2, figsize=(20, 8), facecolor='w', edgecolor='k')
        fig.suptitle("N="+str(v), fontsize=25)
    
        alpha = heckman_fit[v_samples[0]][0].alpha
        beta = heckman_fit[v_samples[0]][0].beta
    
        axs[0].plot(x0, mu_pred_m0, color='blue', marker='', label='Fitted')
        axs[0].fill_between(x0, mu_pred_ci0[:,0], mu_pred_ci0[:,1], facecolor='blue', alpha=0.2, label ='Fitted (90% Intervals)')
        axs[0].plot(x0, (alpha + beta * x0).ravel(), color='red', marker='', label = 'True')
        axs[0].legend(loc="upper right", fontsize='x-large')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('Heckman')
                    
        axs[1].plot(x1, mu_pred_m1, color='blue', marker='', label='Fitted')
        axs[1].fill_between(x1, mu_pred_ci1[:,0], mu_pred_ci1[:,1], facecolor='blue', alpha=0.2, label ='Fitted (90% Intervals)')
        axs[1].plot(x1, (alpha + beta * x1).ravel(), color='red', marker='', label = 'True')
        axs[1].legend(loc="upper right", fontsize='x-large')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].set_title('OLS')


                 

# def plot_posterior(data):
    
#     params = list(data.samples.keys())
#     params = [x for x in params if x not in  ['y_rep', 'lp__']]
#     n_param = len(params)
    
#     if n_param > 3:
#         title = 'Heckman'
#         fig, axs = plt.subplots(2,3, figsize=(18, 8), facecolor='w', edgecolor='k')
#         fig.subplots_adjust(hspace = .5, wspace=.5)
#         fig.suptitle(title+'; N='+ str(data.N), fontsize=25)
       
#     else:
#         title = 'OLS'
#         fig, axs = plt.subplots(1,3, figsize=(10, 3), facecolor='w', edgecolor='k')
#         #fig.subplots_adjust(hspace = .5, wspace=.5)
#         fig.suptitle(title+'; N='+ str(data.N), fontsize=16)

#     axs = axs.ravel()

#     for i in range(n_param):
#         sns.histplot(ax = axs[i], data=data.samples[params[i]], stat="density") 
#         axs[i].set(title=params[i])
#         axs[i].axvline(np.quantile(data.samples[params[i]], .05), color='g', linestyle='--', label='90% intervals')
#         axs[i].axvline(np.quantile(data.samples[params[i]], .95), color='g', linestyle='--')
#         axs[i].axvline(data.__dict__[params[i]], color='r', label='Actual')
#         axs[i].legend(loc="upper right", fontsize='small')
        
        


