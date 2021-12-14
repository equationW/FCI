#########################################################################################
#filename:	CASCI.py												                    #
#author:	WuFangcheng																    #
#date:		2021-07-30  																#
#function： implementation of CASCI						                                 #
#########################################################################################
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as plt
import copy
import scipy.linalg


# 输入参数n、U 
n = 3
Us = np.arange(0,10,1e-2).tolist()
Us = [4]
es = []

for U in Us:
    print('U='+str(U))
    #根据n、U 对h1、eri作出定义
    h1 = np.zeros((n**2, n**2))
    for i in range(n):  # n-1个
        for j in range(n-1):
            if (i+j) % 2 == 1:
                h1[i*n+j, i*n+j+1] = h1[i*n+j+1, i*n+j] = -1.0
            h1[i*n+j, (i-1)*n+j] = h1[(i-1)*n+j, i*n+j] = -1.0
        if i %2 == 0:
            h1[i*n, i*n+n-1] = h1[i*n+n-1, i*n] = -1.0
        h1[i*n+n-1, (i-1)*n+n-1] = h1[(i-1)*n+n-1, i*n+n-1] = -1.0

    eri = np.zeros((n**2, n**2, n**2, n**2))
    for i in range(n**2):
        eri[i, i, i, i] = U

    #S = np.load('S.npy')
    #f_1e = np.load('1e.npy')
    #f_2e = np.load('2e.npy') #K^4大小
    S = np.zeros((n**2, n**2))
    f_1e = h1
    f_2e = eri
    C = np.eye(n**2)
    f_2e_T = f_2e.swapaxes(1, 3) #转置生成交换积分
    f_2e_total = f_2e - 1/2 * f_2e_T #总二电子积分函数
    K = S.shape[0] #基函数数目
    
    N = 4 #电子数
    E_nu = 203.030749104 #核势能
    #C = np.load('C.npy')
    norb = n**2 #活性空间轨道数目
    orb = range(1,n**2+1) #活性空间轨道序号


    def plus_one(a, norb, n_spin):
        b = (copy.deepcopy(a)).astype(int)
        for i in range(norb-1):
            if b[norb-1-i] == 0 and b[norb-2-i] > 0:
                b[norb-i-1] = b[norb-i-2]
                b[norb-i-2] = 0
                #print(b[norb-i-1])
                for j in range(0, n_spin-b[norb-i-1]):
                    #print(j)
                    b[norb-i+j] = b[norb-i-1]+j+1
                for j in range(norb-i+n_spin-b[norb-i-1], norb):
                    b[j] = 0
                break
        return b


    def compare(a1, a2, b1, b2, no):  # number of orbital
        cursor0 = 0
        cursor1 = 0
        ns = 0
        for i in range(norb):
            if a1[i] > 0 and a2[i] == 0:
                ns += 1
            if b1[i] == 0 and b2[i] > 0:
                ns += 1
        #print(ns)
        if ns > 2:
            return ns, no
        for i in range(norb):
            if a1[i] > 0 and a2[i] == 0:
                no[0, cursor0] = 1  # alpha 为1 beta为2
                cursor0 += 1
                no[0, cursor0] = i
                cursor0 += 1
                no[0, cursor0] = a1[i]
                cursor0 += 1
            if a1[i] == 0 and a2[i] > 0:
                no[1, cursor1] = 1
                cursor1 += 1
                no[1, cursor1] = i
                cursor1 += 1
                no[1, cursor1] = a2[i]
                cursor1 += 1
            if b1[i] > 0 and b2[i] == 0:
                no[0, cursor0] = 2
                cursor0 += 1
                no[0, cursor0] = i
                cursor0 += 1
                no[0, cursor0] = b1[i]
                cursor0 += 1
            if b1[i] == 0 and b2[i] > 0:
                no[1, cursor1] = 2
                cursor1 += 1
                no[1, cursor1] = i
                cursor1 += 1
                no[1, cursor1] = b2[i]
                cursor1 += 1
        if no[0, 0] != no[1, 0]:
            tem = copy.deepcopy(no[1, 0:3])
            no[1, 0:3] = no[1, 3:6]
            no[1, 3:6] = tem
        return ns, no

    H_total = np.empty(shape=(0,0))
    psi_alpha_total = np.zeros((1, norb))
    psi_beta_total = np.zeros((1, norb))
    #for n_alpha in range(norb+1):
    for n_alpha in range(2,3):
        if n_alpha > N:
            break
        #对alpha、beta电子数作出规定
        n_beta = N - n_alpha #beta电子数
        if n_beta > N:
            continue
        k = math.factorial(norb) // (math.factorial(n_alpha) * math.factorial(norb-n_alpha)) * math.factorial(norb) // (math.factorial(n_beta) * math.factorial(norb-n_beta))
        print(k)
    '''
        #k determinants数目
        H = np.zeros([k, k])
        psi_alpha = np.zeros([k, norb]) #储存电子占据状态
        psi_alpha[0] = np.concatenate((range(1,n_alpha+1), np.zeros(norb - n_alpha)))
        psi_beta = np.zeros([k, norb])
        psi_beta[0] = np.concatenate((range(1, n_beta+1), np.zeros(norb - n_beta)))


        def H_calculate(no, ns, i, j, P_total):
            E = 0
            P_beta = np.zeros([K,K], int)
            P_alpha = np.zeros([K,K], int)
            if ns > 2:
                return 0
            if ns == 2:
                t = 1 
                if abs(no[0,2]-no[1,2]) % 2 == 1:
                    t *= -1
                if abs(no[0,5]-no[1,5]) % 2 == 1:
                    t *= -1
                m_p = np.tensordot(np.transpose(C[..., orb[no[0,1]]-1]), np.transpose(C[..., orb[no[1,1]]-1]), 0)
                n_q = np.tensordot(np.transpose(C[..., orb[no[0,4]]-1]), np.transpose(C[..., orb[no[1,4]]-1]), 0)
                E += np.sum(np.tensordot(m_p, n_q, 0) * f_2e) * t
                if no[0, 0] == no[0, 3]:
                    E -= np.sum(np.tensordot(m_p, n_q, 0).swapaxes(1,3) * f_2e) * t
                return E
            if ns == 1:
                t = 1
                if abs(no[0,2]-no[1,2]) % 2 == 1:
                    t = -1
                for count in range(norb):
                    if psi_alpha[i, count] > 0 and psi_alpha[j, count] > 0:
                        P_alpha = P_alpha + np.tensordot(np.transpose(C[..., orb[count]-1]), np.transpose(C[..., orb[count]-1]), 0)
                    if psi_beta[i, count] > 0 and psi_beta[j, count] > 0:
                        P_beta = P_beta + np.tensordot(np.transpose(C[..., orb[count]-1]), np.transpose(C[..., orb[count]-1]), 0)
                m_p = np.tensordot(np.transpose(C[..., orb[no[0,1]]-1]), np.transpose(C[..., orb[no[1,1]]-1]), 0)
                E += np.sum(np.tensordot(m_p, P_total+P_alpha+P_beta, 0) * f_2e) * t
                if no[0,0] == 1:
                    E -= np.sum(np.tensordot(m_p, P_total/2 + P_alpha, 0).swapaxes(1,3) * f_2e) * t
                else:
                    E -= np.sum(np.tensordot(m_p, P_total/2 + P_beta, 0).swapaxes(1,3) * f_2e) * t
                E += np.sum(m_p * f_1e) * t
                return E
            if ns == 0:
                for count in range(norb):
                    if psi_alpha[i, count] > 0:
                        P_alpha = P_alpha + np.tensordot(np.transpose(C[..., orb[count]-1]), np.transpose(C[..., orb[count]-1]), 0)
                    if psi_beta[i, count] > 0:
                        P_beta = P_beta + np.tensordot(np.transpose(C[..., orb[count]-1]), np.transpose(C[..., orb[count]-1]), 0)
                E -= np.sum((np.tensordot(P_total/2+P_beta, P_total/2+P_beta, 0)).swapaxes(1,3) * f_2e)/2
                E -= np.sum((np.tensordot(P_total/2+P_alpha, P_total/2+P_alpha, 0)).swapaxes(1,3) * f_2e)/2
                E += np.sum(np.tensordot(P_total+P_alpha+P_beta, P_total+P_alpha+P_beta, 0) * f_2e)/2
                E += np.sum((P_total+P_alpha+P_beta) * f_1e)
                return E

        P_total = np.zeros([K,K], int)
        bct = 0  # beta change times
        for i in range(k):
            #print(i)
            if bct ==  (math.factorial(norb) // (math.factorial(n_beta) * math.factorial(norb-n_beta))):
                bct = 1
                psi_alpha[i] = plus_one(psi_alpha[i-1], norb, n_alpha)
                psi_beta[i] = psi_beta[0]
            else:
                if i > 0:
                    psi_alpha[i] = psi_alpha[i-1]
                if bct > 0:
                    psi_beta[i] = plus_one(psi_beta[i-1], norb, n_beta)
                bct += 1
            for j in range(i+1):
                #print('i='+str(i)+'  j='+str(j))
                no = np.zeros([2, 6], int)
                ns, no = compare(psi_alpha[i], psi_alpha[j], psi_beta[i], psi_beta[j], no)
                H[i, j] = H_calculate(no, ns, i, j, P_total)
                #print('H['+str(i)+','+str(j)+']='+str(H[i,j]))
                H[j, i] = H[i, j]
        psi_alpha_total = np.concatenate((psi_alpha_total, psi_alpha),0)
        psi_beta_total = np.concatenate((psi_beta_total, psi_beta),0)
        print(np.shape(psi_alpha_total))
        H_total = scipy.linalg.block_diag(H_total, H)
    print(np.shape(H_total))
    e, v = linalg.eigh(H_total)
    print(e)
    psi_alpha_total = np.ceil(psi_alpha_total * 0.25)
    psi_beta_total = np.ceil(psi_beta_total * 0.25)
    density_list = []
    '''
    '''
    for i in range(v.shape[0]):
        density = np.transpose(
            np.tile(v[:, i] ** 2, (4, 1))) * (psi_alpha_total+psi_beta_total)[1:, :]
        density = np.mean(density, axis=0) * density.shape[0]
        density_list.append(density.tolist())
    print(density_list)
    plt.imshow(density_list, cmap=plt.get_cmap('PRGn'), vmax=-2, vmin=2)
    plt.colorbar(shrink=0.5)
    plt.show(block=True)
    '''
    #es.append(e)
np.save('es.npy',es)

