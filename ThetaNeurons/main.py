from allFunctions import *

if __name__ == '__main__':
    #default_theta_neuron(50, 0.5, 0.8, 1.5, 0.04, 5, 750, 2000)
    #visualize_change('tau', 'noNoise/results_tau_negative.csv', 57, 5000, 5000, 0.001)
    #visualize_data('SuspectResults/suspect_kappa.csv', 10, 20000, 0)
    suspect_results('suspect_v2.csv')
    if click.confirm(f'[__main__] Run stretching?', default=False):
        print(f'[Multiprocessing] {multiprocessing.cpu_count()} cores available')
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result_eta = pool.starmap_async(stretching_eta, [(-0.01, 0, 2000, -1, 'Noise/results_eta_negative.csv', 0.001,),
                                                         (0.01, 10, 2000, -1, 'Noise/results_eta_positive.csv', 0.001,),
                                                         (-0.01, 0, 2000, -1, 'noNoise/results_eta_negative.csv', 0,),
                                                         (0.01, 10, 2000, -1, 'noNoise/results_eta_positive.csv', 0,)
                                                         ])
        result_kappa = pool.starmap_async(stretching_kappa, [(-0.01, 0, 2000, -1, 'Noise/results_kappa_negative.csv', 0.001,),
                                                             (0.01, 10, 2000, -1, 'Noise/results_kappa_positive.csv', 0.001,),
                                                             (-0.01, 0, 2000, -1, 'noNoise/results_kappa_negative.csv',0,),
                                                             (0.01, 10, 2000, -1, 'noNoise/results_kappa_positive.csv', 0,)
                                                             ])
        result_tau = pool.starmap_async(stretching_tau, [(-0.01, 0, 2000, -1, 'Noise/results_tau_negative.csv', 0.001,),
                                                     (0.01, 10, 2000, -1, 'Noise/results_tau_positive.csv', 0.001,),
                                                     (-0.01, 0, 2000, -1, 'noNoise/results_tau_negative.csv', 0,),
                                                     (0.01, 10, 2000, -1, 'noNoise/results_tau_positive.csv', 0,)])
        result_eta.get()
        result_kappa.get()
        result_tau.get()
