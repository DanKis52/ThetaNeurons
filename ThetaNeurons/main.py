import odesExplicitSolvers as odesES
import odesSystemDescription as odesSD
import csv
import numpy as np
import timeit
import matplotlib.pyplot as plt
import multiprocessing

odesMethod = ['rk4', '0']


#  исходный код из MainScript.py
def default_theta_neuron(m, eta, tau, kappa, epsilon, N, times_zero_eps, times_w_eps, filename='default.csv'):
    numElements = 2 * N + 1
    mass = tau * np.abs(kappa) * m / (1 + m) * np.sqrt(1 + ((1 + eta) * tau) ** 2)
    alpha = np.arccos(- (1 + eta) * tau / np.sqrt(1 + ((1 + eta) * tau) ** 2))
    theta0 = np.arccos(- 1 / (numElements - 1))
    maxR2 = np.abs((1 + N * np.exp(+ 1j * theta0) + N * np.exp(1j * theta0)) / numElements)
    initialPoint = [0]
    initialPoint = np.append(initialPoint, np.array([+ theta0] * N))
    initialPoint = np.append(initialPoint, np.array([- theta0] * N))
    try:
        initialPoint[1:] += np.array(
            [-0.83105796, 0.14132231, -0.16670416, 0.4830021, 0.2342222, -0.40335194, 0.0925514, -0.84020301, 0.5538672, -0.5652341])
    except ValueError:
        return print('[MainScript] Incorrect initial points array')
    initialPoint = np.append(initialPoint, 0)
    print(f'[MainScript] m = {m}, eta = {eta}, tau = {tau}, kappa = {kappa}, epsilon = {epsilon}, numElements = {numElements}')
    print(f'[MainScript] mass = {mass}, alpha = {alpha}, theta_0 = {theta0}, maxR2 = {maxR2}')
    # eps = 0
    odesSpan = [0., times_zero_eps]
    odesNumSteps = [10 * np.int_(odesSpan[1]), 10]
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, 0)
    odesModelDescription = [odesModel, 'rhsFunction']
    qpi, t = odesES.ivpSolution(odesMethod, odesSpan, odesNumSteps, odesModelDescription, initialPoint)
    qpi1, t1 = np.transpose(qpi), t
    # eps != 0
    odesModel.eps = epsilon
    odesSpan = [0., times_w_eps]
    odesNumSteps = [10 * np.int_(odesSpan[1]), 10]
    initialPoint = qpi1[:, -1]
    qpi, t = odesES.ivpSolution(odesMethod, odesSpan, odesNumSteps, odesModelDescription, initialPoint)
    qpi2, t2 = np.transpose(qpi), t + t1[-1]
    qpi, t = np.concatenate((qpi1, qpi2), axis=1), np.concatenate((t1, t2), axis=0)

    for n in range(0, 2 * N + 1):
        v = np.angle(np.exp(1j * (qpi[n] - qpi[0])))
        itspan, itstride = np.int_(np.where(t == 30)[0][0]), 5
        vmean = [np.trapz(v[it:it + itspan], t[it:it + itspan]) / (t[it + itspan] - t[it]) for it in
                 range(0, len(t) - itspan, itstride)]
        plt.plot(t[0:-itspan:itstride], vmean)
    plt.grid(color=[.5, .5, .5], linestyle=':', linewidth=1)
    plt.xlabel("$t$")
    plt.ylabel("$\phi$")
    plt.axvline(x=(t1[-1]), color='red')
    plt.title(f'$\eta$ = {odesModel.eta}\n')
    plt.show()

    r1 = []
    r2 = []
    for j in range(len(qpi[0])):
        points = qpi[:, j]
        sum_1 = 0
        sum_2 = 0
        for k in range(0, numElements):
            sum_1 += np.exp(1j * points[k])
            sum_2 += np.exp(2j * points[k])
        r1.append(np.abs(sum_1 / len(initialPoint - 1)))
        r2.append(np.abs(sum_2 / len(initialPoint - 1)))

    data = [m, eta, tau, kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
    data = np.append(data, qpi[:, -1])

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['m', 'eta', 'tau', 'kappa', 'epsilon', 'mean r1', 'mean r2', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6',
             'th7', 'th8', 'th9', 'th10', 'th11', 's'])
        writer.writerow(data)
        f.close()
    print(f'[MainScript] {filename} updated')


#  string_index - номер строки в соответствии с нумерацией в csv файле
def visualize_data(filename, string_index, times):
    m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, string_index-1)
    numElements = len(initialPoint) - 1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Visualization] Calculating theta for eta = {odesModel.eta}...')
    theta_sol, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint)
    qpi = np.transpose(theta_sol)
    print(f'Final points: {qpi[:,-1]}')

    for n in range(0, numElements):
        v = np.angle(np.exp(1j * (qpi[n] - qpi[0])))
        itspan, itstride = np.int_(np.where(t == 30)[0][0]), 5
        vmean = [np.trapz(v[it:it + itspan], t[it:it + itspan]) / (t[it + itspan] - t[it]) for it in
                 range(0, len(t) - itspan, itstride)]
        plt.plot(t[0:-itspan:itstride], vmean)
    plt.grid(color=[.5, .5, .5], linestyle=':', linewidth=1)
    plt.xlabel("$t$")
    plt.ylabel("$\phi$")
    plt.title(f'$\eta$ = {odesModel.eta}\n')
    plt.show()


def is_cyclope(points, eta):
    if len(np.unique(np.round(points, 1))) == 4:
        return True
    else:
        with open('suspect_eta.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eta])
            f.close()
        return True


#  график перехода eta
#  string_index - номер строки в соответствии с нумерацией в csv файле
#  times - t для каждой eta
def visualize_eta_change(filename, string_index, times1, times2):
    if string_index < 2 and string_index != -1:
        return print('[Visualisation error] String index incorrect')
    data_first = []
    data_second = []
    with open(filename, 'r', newline='') as f:
        file_reader = csv.reader(f, delimiter=",")
        if string_index == -1:
            data_str = list(file_reader)[string_index-1:]
        else:
            data_str = list(file_reader)[string_index - 1:string_index+1]
        f.close()
    if not data_str:
        return print('[Visualisation error] String index incorrect')
    data_first = np.append(data_first, [float(x) for x in data_str[0]])
    data_second = np.append(data_second, [float(x) for x in data_str[1]])
    m = data_first[0]
    eta1, eta2 = data_first[1], data_second[1]
    tau = data_first[2]
    kappa = data_first[3]
    epsilon = data_first[4]
    odesS = [0., times1]
    odesNS = [10 * np.int_(odesS[1]), 10]
    initialPoint = data_first[7:]
    numElements = len(initialPoint) - 1
    odesModel = odesSD.thetaNeurons(tau, eta1, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    print(f'[Visualization] Calculating theta for eta = {eta1} & eta = {eta2}')
    qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint)
    qpi1, t1 = np.transpose(qpi), t
    odesModel.eta = eta2
    initialPoint = qpi1[:, -1]
    odesS = [0., times2]
    odesNS = [10 * np.int_(odesS[1]), 10]
    qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint)
    qpi2, t2 = np.transpose(qpi), t + t1[-1]
    qpi, t = np.concatenate((qpi1, qpi2), axis=1), np.concatenate((t1, t2), axis=0)

    for n in range(0, numElements):
        v = np.angle(np.exp(1j * (qpi[n] - qpi[0])))
        itspan, itstride = np.int_(np.where(t == 30)[0][0]), 5
        vmean = [np.trapz(v[it:it + itspan], t[it:it + itspan]) / (t[it + itspan] - t[it]) for it in
                 range(0, len(t) - itspan, itstride)]
        plt.plot(t[0:-itspan:itstride], vmean)
    plt.grid(color=[.5, .5, .5], linestyle=':', linewidth=1)
    plt.xlabel("$t$")
    plt.ylabel("$\phi$")
    plt.title(f'$\eta$ = {eta1} to $\eta$ = {eta2}\n')
    plt.axvline(x=(t1[-1]), color='red')
    plt.show()


def get_string_data(filename, string_index):
    data = []
    with open(filename, 'r', newline='') as f:
        file_reader = csv.reader(f, delimiter=",")
        data_str = list(file_reader)[string_index]
        f.close()
    data = np.append(data, [float(x) for x in data_str])
    return data[0], data[1], data[2], data[3], data[4], data[7:]


def stretching_eta(delta_eta, eta_final, times, start_string_index, filename):
    m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, start_string_index)
    if delta_eta < 0 and eta_final > eta:
        return print(f'[finalEta = {eta_final}] unreachable: [eta = {eta}, deltaEta = {delta_eta}]')
    numElements = len(initialPoint)-1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Stretching eta] m = {m}, eta = {eta}, tau = {tau}, kappa = {kappa}, epsilon = {epsilon}, numElements = {numElements}, deltaEta = {delta_eta}, finalEta = {eta_final}, odesSpan = {odesS}')
    if delta_eta > 0:
        while round(odesModel.eta+delta_eta, 3) <= eta_final:
            elapsedTime = timeit.default_timer()
            odesModel.eta = round(odesModel.eta + delta_eta, 3)
            theta_sol, t_sol = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint)
            qpi = np.transpose(theta_sol)
            r1 = []
            r2 = []
            for j in range(len(qpi[0])):
                points = qpi[:, j]
                sum_1 = 0
                sum_2 = 0
                for k in range(0, numElements):
                    sum_1 += np.exp(1j * points[k])
                    sum_2 += np.exp(2j * points[k])
                r1.append(np.abs(sum_1 / numElements))
                r2.append(np.abs(sum_2 / numElements))
            data = [m, odesModel.eta, tau, kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(f'[{round(elapsedTime,2)}s.] eta = {odesModel.eta} in {filename} saved, {round((eta_final-odesModel.eta)/delta_eta)} iterations left')
            if not is_cyclope(qpi[:, -1], odesModel.eta):
                break
        else:
            print(f'[eta = {odesModel.eta}] Process completed!')
    if delta_eta < 0:
        while round(-odesModel.eta/delta_eta, 3) > 0:
            elapsedTime = timeit.default_timer()
            odesModel.eta = round(odesModel.eta + delta_eta, 3)
            theta_sol, t_sol = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint)
            qpi = np.transpose(theta_sol)
            r1 = []
            r2 = []
            for j in range(len(qpi[0])):
                points = qpi[:, j]
                sum_1 = 0
                sum_2 = 0
                for k in range(0, numElements):
                    sum_1 += np.exp(1j * points[k])
                    sum_2 += np.exp(2j * points[k])
                r1.append(np.abs(sum_1 / numElements))
                r2.append(np.abs(sum_2 / numElements))
            data = [m, odesModel.eta, tau, kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(
                f'[{round(elapsedTime,2)}s.] eta = {odesModel.eta} in {filename} saved, {round(-odesModel.eta / delta_eta)} iterations left')
            if not is_cyclope(qpi[:, -1], odesModel.eta):
                break
        else:
            print(f'[eta = {odesModel.eta}] Process completed!')


if __name__ == '__main__':
    # default_theta_neuron(5, 15, 0.8, 0.2, 0.04, 5, 2000, 200000)
    # visualize_eta_change('results_eta_positive.csv', 700, 3000, 5000)
    # visualize_data('default.csv', 2, 10000)
    processes = [multiprocessing.Process(target=stretching_eta, args=(0.01, 70, 2000, -1, 'results_eta_positive.csv',)),
                 multiprocessing.Process(target=stretching_eta, args=(-0.01, -1, 2000, -1, 'results_eta_negative.csv',))]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
