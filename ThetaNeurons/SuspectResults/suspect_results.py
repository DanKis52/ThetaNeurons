import csv
import numpy as np
import matplotlib.pyplot as plt

with open('../suspect_v2.csv', 'r', newline='') as f:
    suspect_eta = []
    suspect_kappa = []
    suspect_tau = []
    csv_reader = csv.reader(f)
    next(csv_reader)
    for row in csv_reader:
        m, eta, tau, kappa, epsilon, mean_r1, mean_r2, th1, th2, th3, th4, th5, th6, th7, th8, th9, th10, th11, s = (list(map(float, row)))
        if tau == 0.8 and kappa == 0.2:
            suspect_eta.append(list(map(float, row)))
        if eta == 0.3 and tau == 0.8:
            suspect_kappa.append(list(map(float, row)))
        if eta == 0.3 and kappa == 0.2:
            suspect_tau.append(list(map(float, row)))
    f.close()
sus_eta = np.array(suspect_eta)
sus_kappa = np.array(suspect_kappa)
sus_tau = np.array(suspect_tau)
sus_eta = sus_eta[sus_eta[:, 1].argsort()]
sus_kappa = sus_kappa[sus_kappa[:, 3].argsort()]
sus_tau = sus_tau[sus_tau[:, 2].argsort()]
with open('suspect_eta.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['m', 'eta', 'tau', 'kappa', 'epsilon', 'mean r1', 'mean r2', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6',
         'th7', 'th8', 'th9', 'th10', 'th11', 's'])
    writer.writerows(sus_eta)
    f.close()
with open('suspect_kappa.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['m', 'eta', 'tau', 'kappa', 'epsilon', 'mean r1', 'mean r2', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6',
         'th7', 'th8', 'th9', 'th10', 'th11', 's'])
    writer.writerows(sus_kappa)
    f.close()
with open('suspect_tau.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['m', 'eta', 'tau', 'kappa', 'epsilon', 'mean r1', 'mean r2', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6',
         'th7', 'th8', 'th9', 'th10', 'th11', 's'])
    writer.writerows(sus_tau)
    f.close()


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(sus_eta[:,1], sus_eta[:,3], sus_eta[:,2])
ax.scatter(sus_kappa[:,1], sus_kappa[:,3], sus_kappa[:,2])
ax.scatter(sus_tau[:,1], sus_tau[:,3], sus_tau[:,2])
ax.set_xlabel('$\eta$')
ax.set_ylabel('$\kappa$')
ax.set_zlabel(r'$\tau$')
plt.show()

