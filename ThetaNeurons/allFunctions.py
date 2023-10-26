import odesExplicitSolvers as odesES
import odesSystemDescription as odesSD
import csv
import numpy as np
import timeit
import matplotlib.pyplot as plt
import multiprocessing
from typing import List, NoReturn
import click

odesMethod = ['rk4', '0']


def add_noise(points: List[float], scale: float) -> List[float]:
    noise = np.random.uniform(low=-scale, high=scale, size=len(points)-1)
    #noise = np.random.normal(scale=scale, size=len(points)-1)
    noise = np.append(noise, 0)
    return points+noise


#  для исключения больших значений
def initial_point_fixer(filename: str) -> NoReturn:
    with open(filename, 'r', newline='') as f:
        file_reader = csv.reader(f, delimiter=",")
        lines = list(file_reader)
        for i in range(1, len(lines)):
            for j in range(7, 18):
                lines[i][j] = float(lines[i][j]) % (2 * np.pi)
        f.close()
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)
        f.close()


#  исходный код из MainScript.py
def default_theta_neuron(m: float, eta: float, tau: float, kappa: float, epsilon: float, N: int, times_zero_eps: int, times_w_eps: int, filename: str = 'default.csv') -> NoReturn:
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
    plt.title(f'$\eta$ = {odesModel.eta} $\kappa$ = {odesModel.kappa}\n')
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
    for j in range(7, 18):
        data[j] = float(data[j]) % (2 * np.pi)
    if click.confirm(f'[MainScript] Update file "{filename}"?', default=False):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['m', 'eta', 'tau', 'kappa', 'epsilon', 'mean r1', 'mean r2', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6',
             'th7', 'th8', 'th9', 'th10', 'th11', 's'])
            writer.writerow(data)
            f.close()
        print(f'[MainScript] {filename} updated')


#  string_index - номер строки в соответствии с нумерацией в csv файле
def visualize_data(filename: str, string_index: int, times: int, noise_scale: float = 0) -> NoReturn:
    try:
        m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, string_index-1)
    except ValueError:
        return print('[Visualization] Error: Incorrect data string')
    initialPoint = add_noise(initialPoint, noise_scale)
    numElements = len(initialPoint) - 1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Visualization] Calculating theta for eta = {odesModel.eta}, kappa = {odesModel.kappa}, tau = {odesModel.tau}, noise_scale = {noise_scale}')
    theta_sol, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
    qpi = np.transpose(theta_sol)
    is_cyclope(qpi[:,-1], odesModel.eta, odesModel.kappa, odesModel.tau, np.ndarray([]), False)
    is_cyclope_v2(initialPoint, qpi[:,-1], odesModel.eta, odesModel.kappa, odesModel.tau, np.ndarray([]), False)
    for n in range(0, numElements):
        v = np.angle(np.exp(1j * (qpi[n] - qpi[0])))
        itspan, itstride = np.int_(np.where(t == 30)[0][0]), 5
        vmean = [np.trapz(v[it:it + itspan], t[it:it + itspan]) / (t[it + itspan] - t[it]) for it in
                 range(0, len(t) - itspan, itstride)]
        plt.plot(t[0:-itspan:itstride], vmean)
    plt.grid(color=[.5, .5, .5], linestyle=':', linewidth=1)
    plt.xlabel("$t$")
    plt.ylabel(r"$\varphi$")
    plt.title(fr'$\eta$ = {odesModel.eta} $\kappa$ = {odesModel.kappa} $\tau$ = {odesModel.tau}'+'\n')
    plt.show()


def is_cyclope(points: List[float], eta: float, kappa: float, tau: float, data: np.ndarray, write_to_file: bool = True) -> bool:
    if len(np.unique(np.round(points, 1))) == 4:
        return True
    else:
        if write_to_file:
            with open('suspect.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
        print(f'[Suspect state] eta = {eta}, kappa = {kappa}, tau = {tau} (is_cyclope)')
        return True


def is_cyclope_v2(initial_points: List[float], final_points: List[float], eta: float, kappa: float, tau: float, data: np.ndarray, write_to_file: bool = True) -> bool:
    half = int((len(initial_points)-1)/2)
    initial = []
    final = []
    for n in range(0, len(final_points)-1):
        i = np.angle(np.exp(1j * (initial_points[n] - initial_points[0])))
        f = np.angle(np.exp(1j * (final_points[n] - final_points[0])))
        initial = np.append(initial, i)
        final = np.append(final, f)
    initial_first = initial[1:half+1]
    initial_second = initial[half+1:]
    final_first = final[1:half + 1]
    final_second = final[half + 1:]
    initial_first, initial_second, final_first, final_second = np.sort([initial_first, initial_second, final_first, final_second])
    initial_diff = np.max(np.abs([initial_first[-1]-initial_first[0], initial_second[-1]-initial_second[0]]))
    final_diff = np.max(np.abs([final_first[-1] - final_first[0], final_second[-1] - final_second[0]]))
    if final_diff > initial_diff:
        if write_to_file:
            with open('suspect_v2.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
        print(f'[Suspect state] eta = {eta}, kappa = {kappa}, tau = {tau} (is_cyclope_v2)')
        return True
    else:
        return True


#  график перехода
#  string_index - номер строки в соответствии с нумерацией в csv файле
#  times - t для каждого значения
def visualize_change(varname: str, filename: str, string_index: int, times1: int, times2: int, noise_scale: float = 0) -> NoReturn:
    if string_index < 2 and string_index != -1:
        return print('[Changes Visualisation] Error: String index incorrect')
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
        return print('[Changes Visualisation] Error: String index incorrect')
    try:
        data_first = np.append(data_first, [float(x) for x in data_str[0]])
        data_second = np.append(data_second, [float(x) for x in data_str[1]])
    except ValueError:
        return print('[Changes Visualisation] ValueError: Incorrect data string')
    m = data_first[0]
    print(f'[Changes Visualisation] noise_scale = {noise_scale}')
    if varname == 'eta':
        eta1, eta2 = data_first[1], data_second[1]
        tau = data_first[2]
        kappa = data_first[3]
        epsilon = data_first[4]
        odesS = [0., times1]
        odesNS = [10 * np.int_(odesS[1]), 10]
        initialPoint = add_noise(data_first[7:], noise_scale)
        numElements = len(initialPoint) - 1
        odesModel = odesSD.thetaNeurons(tau, eta1, kappa, m, epsilon)
        odesModelDescription = [odesModel, 'rhsFunction']
        print(f'[Changes Visualisation] Calculating theta for eta = {eta1} & eta = {eta2}')
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
        qpi1, t1 = np.transpose(qpi), t
        odesModel.eta = eta2
        initialPoint = add_noise(qpi1[:, -1], noise_scale)
        odesS = [0., times2]
        odesNS = [10 * np.int_(odesS[1]), 10]
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
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
        plt.ylabel(r"$\varphi$")
        plt.title(f'$\eta$ = {eta1} to $\eta$ = {eta2}\n')
        plt.axvline(x=(t1[-1]-30), color='red')
        plt.axvline(x=(t1[-1]), color='red')
        plt.show()
    elif varname == 'kappa':
        eta = data_first[1]
        tau = data_first[2]
        kappa1, kappa2 = data_first[3], data_second[3]
        epsilon = data_first[4]
        odesS = [0., times1]
        odesNS = [10 * np.int_(odesS[1]), 10]
        initialPoint = data_first[7:]
        numElements = len(initialPoint) - 1
        odesModel = odesSD.thetaNeurons(tau, eta, kappa1, m, epsilon)
        odesModelDescription = [odesModel, 'rhsFunction']
        print(f'[Changes Visualisation] Calculating theta for kappa = {kappa1} & kappa = {kappa2}')
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
        qpi1, t1 = np.transpose(qpi), t
        odesModel.kappa = kappa2
        initialPoint = qpi1[:, -1]
        odesS = [0., times2]
        odesNS = [10 * np.int_(odesS[1]), 10]
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
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
        plt.ylabel(r"$\varphi$")
        plt.title(f'$\kappa$ = {kappa1} to $\kappa$ = {kappa2}\n')
        plt.axvline(x=(t1[-1]), color='red')
        plt.show()
    elif varname == 'tau':
        eta = data_first[1]
        tau1, tau2 = data_first[2], data_second[2]
        kappa = data_first[3]
        epsilon = data_first[4]
        odesS = [0., times1]
        odesNS = [10 * np.int_(odesS[1]), 10]
        initialPoint = data_first[7:]
        numElements = len(initialPoint) - 1
        odesModel = odesSD.thetaNeurons(tau1, eta, kappa, m, epsilon)
        odesModelDescription = [odesModel, 'rhsFunction']
        print(f'[Changes Visualisation] Calculating theta for tau = {tau1} & tau = {tau2}')
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
        qpi1, t1 = np.transpose(qpi), t
        odesModel.tau = tau2
        initialPoint = qpi1[:, -1]
        odesS = [0., times2]
        odesNS = [10 * np.int_(odesS[1]), 10]
        qpi, t = odesES.ivpSolution(odesMethod, odesS, odesNS, odesModelDescription, initialPoint, 'visualisation')
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
        plt.ylabel(r"$\varphi$")
        plt.title(fr'$\tau$ = {tau1} to $\tau$ = {tau2}'+'\n')
        plt.axvline(x=(t1[-1]), color='red')
        plt.show()
    else:
        return print(f'[Changes Visualisation] Error: Incorrect varname "{varname}"')


def get_string_data(filename: str, string_index: int) -> tuple:
    data = []
    with open(filename, 'r', newline='') as f:
        file_reader = csv.reader(f, delimiter=",")
        data_str = list(file_reader)[string_index]
        f.close()
    data = np.append(data, [float(x) for x in data_str])
    return data[0], data[1], data[2], data[3], data[4], data[7:]


def stretching_eta(delta_eta: float, eta_final: float, times: int, start_string_index: int, filename: str, noise_scale: float = 0) -> NoReturn:
    m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, start_string_index)
    if delta_eta < 0 and eta_final > eta:
        return print(f'[finalEta = {eta_final}] unreachable: [eta = {eta}, deltaEta = {delta_eta}]')
    numElements = len(initialPoint)-1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Stretching eta] ({filename}) m = {m}, eta = {eta}, tau = {tau}, kappa = {kappa}, epsilon = {epsilon}, numElements = {numElements}, deltaEta = {delta_eta}, finalEta = {eta_final}, odesSpan = {odesS}, noise_scale = {noise_scale}')
    initialPoint = add_noise(initialPoint, noise_scale)
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
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(f'[{round(elapsedTime,2)}s.] eta = {odesModel.eta} in {filename} saved, {round((eta_final-odesModel.eta)/delta_eta)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
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
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(
                f'[{round(elapsedTime,2)}s.] eta = {odesModel.eta} in {filename} saved, {round(-odesModel.eta / delta_eta)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
                break
        else:
            print(f'[eta = {odesModel.eta}] Process completed!')


def stretching_kappa(delta_kappa: float, kappa_final: float, times: int, start_string_index: int, filename: str, noise_scale: float = 0) -> NoReturn:
    m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, start_string_index)
    if delta_kappa < 0 and kappa_final > kappa:
        return print(f'[finalKappa = {kappa_final}] unreachable: [kappa = {kappa}, deltaKappa = {delta_kappa}]')
    numElements = len(initialPoint)-1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Stretching kappa] ({filename}) m = {m}, eta = {eta}, tau = {tau}, kappa = {kappa}, epsilon = {epsilon}, numElements = {numElements}, deltaKappa = {delta_kappa}, finalKappa = {kappa_final}, odesSpan = {odesS}, noise_scale = {noise_scale}')
    initialPoint = add_noise(initialPoint, noise_scale)
    if delta_kappa > 0:
        while round(odesModel.kappa+delta_kappa, 3) <= kappa_final:
            elapsedTime = timeit.default_timer()
            odesModel.kappa = round(odesModel.kappa + delta_kappa, 3)
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
            data = [m, eta, tau, odesModel.kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(f'[{round(elapsedTime,2)}s.] kappa = {odesModel.kappa} in {filename} saved, {round((kappa_final-odesModel.kappa)/delta_kappa)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
                break
        else:
            print(f'[kappa = {odesModel.kappa}] Process completed!')
    if delta_kappa < 0:
        while round(-odesModel.kappa/delta_kappa, 3) > 0:
            elapsedTime = timeit.default_timer()
            odesModel.kappa = round(odesModel.kappa + delta_kappa, 3)
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
            data = [m, eta, tau, odesModel.kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(
                f'[{round(elapsedTime,2)}s.] kappa = {odesModel.kappa} in {filename} saved, {round(-odesModel.kappa / delta_kappa)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
                break
        else:
            print(f'[kappa = {odesModel.kappa}] Process completed!')


def stretching_tau(delta_tau: float, tau_final: float, times: int, start_string_index: int, filename: str, noise_scale: float = 0) -> NoReturn:
    m, eta, tau, kappa, epsilon, initialPoint = get_string_data(filename, start_string_index)
    if delta_tau < 0 and tau_final > tau:
        return print(f'[finalTau = {tau_final}] unreachable: [tau = {tau}, deltaTau = {delta_tau}]')
    numElements = len(initialPoint)-1
    odesModel = odesSD.thetaNeurons(tau, eta, kappa, m, epsilon)
    odesModelDescription = [odesModel, 'rhsFunction']
    odesS = [0., times]
    odesNS = [10 * np.int_(odesS[1]), 10]
    print(f'[Stretching tau] ({filename}) m = {m}, eta = {eta}, tau = {tau}, kappa = {kappa}, epsilon = {epsilon}, numElements = {numElements}, deltaTau = {delta_tau}, finalTau = {tau_final}, odesSpan = {odesS}, noise_scale = {noise_scale}')
    initialPoint = add_noise(initialPoint, noise_scale)
    if delta_tau > 0:
        while round(odesModel.tau+delta_tau, 3) <= tau_final:
            elapsedTime = timeit.default_timer()
            odesModel.tau = round(odesModel.tau + delta_tau, 3)
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
            data = [m, eta, odesModel.tau, kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(f'[{round(elapsedTime,2)}s.] tau = {odesModel.tau} in {filename} saved, {round((tau_final-odesModel.tau)/delta_tau)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
                break
        else:
            print(f'[tau = {odesModel.tau}] Process completed!')
    if delta_tau < 0:
        while round(-odesModel.tau/delta_tau, 3) > 0:
            elapsedTime = timeit.default_timer()
            odesModel.tau = round(odesModel.tau + delta_tau, 3)
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
            data = [m, eta, odesModel.tau, kappa, epsilon, np.sum(r1) / len(r1), np.sum(r2) / len(r2)]
            data = np.append(data, qpi[:, -1])
            for j in range(7, 18):
                data[j] = float(data[j]) % (2 * np.pi)
            with open(filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()
            elapsedTime = timeit.default_timer() - elapsedTime
            print(
                f'[{round(elapsedTime,2)}s.] tau = {odesModel.tau} in {filename} saved, {round(-odesModel.tau / delta_tau)} iterations left')
            if (not is_cyclope(qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)) or (not is_cyclope_v2(initialPoint, qpi[:, -1], odesModel.eta, odesModel.kappa, odesModel.tau, data)):
                break
        else:
            print(f'[tau = {odesModel.tau}] Process completed!')
