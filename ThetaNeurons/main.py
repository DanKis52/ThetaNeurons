from allFunctions import *

if __name__ == '__main__':
    #default_theta_neuron(5, 10, 0.8, 0.2, 0.04, 5, 1000, 20000)
    visualize_change('eps', 'Noise/results_eps_positive.csv', 249, 2000, 100000, 0.001)
    #visualize_data('Noise/results_eps_positive.csv', 245, 80000, 0.001)
    #suspect_results('suspect.csv')
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
        result_eps = pool.starmap_async(stretching_eps, [(-0.001, 0, 2000, -1, 'Noise/results_eps_negative.csv', 0.001,),
                                                     (0.001, 10, 2000, -1, 'Noise/results_eps_positive.csv', 0.001,),
                                                     (-0.001, 0, 2000, -1, 'noNoise/results_eps_negative.csv', 0,),
                                                     (0.001, 10, 2000, -1, 'noNoise/results_eps_positive.csv', 0,)])
        #result_eta.get()
        #result_kappa.get()
        #result_tau.get()
        result_eps.get()
