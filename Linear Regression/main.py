import numpy as np
from matplotlib import pyplot as plt

def linear_regression():
    X=[]
    Y = []
    years=[]
    salary=[]
    data = np.genfromtxt("salary.csv", delimiter=',', dtype=float)
    for element in data:
        years.append(element[0])
        salary.append(element[1])

    for element in years:
        row=[]
        row.append(element)
        row.append(1)
        X.append(row)

    for element in salary:
        Y.append(element)

    X.pop(0)
    Y.pop(0)

    X_years=np.array(X)
    Y_salary=np.array(Y).transpose()
    p_inversaX=np.linalg.pinv(X_years)


    print("PSEUDO INVERSA X\n", p_inversaX)
    w_tr=np.matmul(p_inversaX, Y_salary)

    W=np.transpose(w_tr)
    print("---------------- X -----------------\n", X_years)
    print("---------------- Y -----------------\n", Y_salary)
    print("---------------- W -----------------\n", W)


    plt.title("Grafic")
    plt.xlabel("Ani experienta")
    plt.ylabel("Salariu")
    y=np.array(years)
    s=np.array(salary)

    model=W[0]*y+W[1]
    plt.plot(y,s, 'bo')
    plt.plot(y,model, 'r')
    plt.show()

    exp=0
    while exp != -1:
        exp = float(input("Ani experienta = "))
        pred = exp * W[0] + W[1]
        print(f"Salariu = {pred}")




if __name__ == '__main__':
    linear_regression()