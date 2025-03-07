```python
import numpy as np 
import scipy.stats as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Ejercicio 1.



```python
theta = 3
aces = np.linspace(theta-3,theta+3)
ns = [25,50,100,200,400,800,1600,3200]

def EM_normal(a,mu,m,n):
    # Hyperparams
    hat = mu
    tol = 1e-5 
    maxiter = 300
    i = 0
    delta = 1

    while (np.abs(delta)>tol and i < maxiter):
        hat0 = hat
        hat = mu*m/n + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/n
        delta = hat-hat0
        i+=1
    if i==maxiter:
        print("Se alcanzó el número máximo de iteraciones.")
    #print(f'θ estimado: {hat:.4f}, número de iteraciones: {i}')
    return hat
fig, ax = plt.subplots(1,2,figsize=(15,5))
for n in ns:
    theta_hat = []
    for a in aces:
        x = sc.norm.rvs(loc=theta,size=n)
        #xm = x[x<a]
        #xn = x[x>=a]
        censored = x>=a
        m = n-len(x[censored])
        #n = k-m
        x[censored] = a
        xm = x[np.logical_not(censored)]
        mu_xm = np.mean(xm)
        theta_hat.append(EM_normal(a,mu_xm,m,n))
    #sns.kdeplot(x=theta_hat)
    ax[0].plot(aces,theta_hat)
    ax[1] = sns.kdeplot(x=theta_hat)
plt.legend(ns)
plt.axvline(x=theta,linestyle='--',color='k',alpha=0.5)
plt.xlabel('$\hat{\\theta}$')
plt.title('Densidad de $\hat{\\theta}$ a diferentes valores de a y diferente tamaño de muestra')
ax[0].legend(ns)
ax[0].axhline(y=theta,linestyle='--',color='k',alpha=0.5)
ax[0].set_xlabel('a')
ax[0].set_ylabel('$\hat{\\theta}$')
ax[0].set_title('$\hat{\\theta}$ vs valor de a')
```

    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)


    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.





    Text(0.5, 1.0, '$\\hat{\\theta}$ vs valor de a')




    
![png](Tarea4_files/Tarea4_2_5.png)
    


Puede verse que el algoritmo $EM$, para estos datos normalmente distribuidos censurados por la derecha, es particularmente resiliente tanto a tamaño de muestra como al lugar en el que se censura la distribución normal. Por lo tanto, si se tiene una buena razón para asumir que la distribución es normal, a pesar de no poder conocerse adecuadamente los datos, puede ser un algoritmo muy valioso para encontrar los parámetros de normalidad de una población. 

## Pregunta 2.

Si la verosimilitud observada es dada por la siguiente ecuación: 
$$\mathcal{L}(\theta|\overrightarrow{x})\propto\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m}$$
Entonces, para maximizarla, derivando con respecto al parámetro $\theta$: 
$$\frac{d\mathcal{L}}{d\theta} \propto \left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m} + \left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\frac{d}{d\theta}\left[1-\Phi(a-\theta)\right]^{n-m}$$
Derivando, se obtiene: 
$$\frac{d\mathcal{L}}{d\theta} \propto \left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m} + (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m-1}\phi(a-\theta)$$
Se obtiene el máximo cuando esta derivada es 0, por lo que evaluando en una $\theta^*$ tal que la derivada de la verosimilitud sea 0, puede encontrarse el máximo. Adicionalmente, se hace abuso de notación, y se anota esta $\theta^*$ hasta llegar al resultado final. Continuando con la igualdad: 
$$\begin{gather*}
0=\left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m} + (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m-1}\phi(a-\theta)\\
-\left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m} = (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m-1}\phi(a-\theta) \\
-\left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]= (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\left[1-\Phi(a-\theta)\right]^{n-m-1}\left[1-\Phi(a-\theta)\right]^{m-n} \phi(a-\theta) \\
-\left[\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)\right]= (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\frac{\phi(a-\theta)}{1-\Phi(a-\theta)}
\end{gather*}$$
Evaluando la derivada se obtiene lo siguiente: 
$$\begin{gather*}

\frac{d}{d\theta}\prod_{i=1}^m\phi(x_i;\theta,1)= \sum_{i=1}^{m}\left[ \phi(x_i;\theta,1)'\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1) \right]
\end{gather*}$$
Donde la comilla denota la primera derivada. Entonces, sustituyendo en nuestra ecuación: 
$$\begin{align*}
-\sum_{i=1}^{m}\left[ \phi(x_i;\theta,1)'\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1) \right] &= (n-m)\left[\prod_{i=1}^m\phi(x_i;\theta,1)\right]\frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\ 
-\frac{\sum_{i=1}^{m}\left[ \phi(x_i;\theta,1)'\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1) \right]}{\prod_{i=1}^m\phi(x_i;\theta,1)} &= (n-m)\frac{\phi(a-\theta)}{1-\Phi(a-\theta)} 
\end{align*}$$
Siendo más explícitos con la productoria, que puede ser descrita de la siguiente manera: 

$$\prod_{i=1}^m\phi(x_i;\theta,1) = \phi(x_i;\theta,1)\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1)$$
Se utiliza esto para cuando se haga la sumatoria para cada elemento $i$, con el fin de facilitar el álgebra. Entonces, retornando a la ecuación: 
$$\begin{align*}

-\frac{\sum_{i=1}^{m}\left[ \phi(x_i;\theta,1)'\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1) \right]}{\phi(x_i;\theta,1)\underset{i\ne j}{\prod_{j=1}^{m}}\phi(x_j;\theta,1)} &= (n-m)\frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
-\sum_{i=1}^{m}\frac{\phi(x_i;\theta,1)'}{\phi(x_i;\theta,1)} &=(n-m) \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} 
\end{align*}$$

Vale la pena recordar que la derivada para cada elemento $i$ puede calcularse de la siguiente manera: 
$$\begin{align*}
\frac{d}{d\theta}\phi(x_i;\theta,1) &= \frac{d}{d\theta}\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(x_i-\theta)^2}{2}\right) \\

&= -\frac{1}{\sqrt{2\pi}}\exp\left(\frac{(x_i-\theta)^2}{2}\right)(x_i-\theta)(-1) \\
&= -\frac{1}{\sqrt{2\pi}}\exp\left(\frac{(x_i-\theta)^2}{2}\right)(\theta-x_i) \\
&= -\phi(x_i;\theta,1)(\theta-x_i)
\end{align*}$$

Entonces, retornando a la ecuación: 

$$\begin{align*}

-\sum_{i=1}^{m}\frac{\phi(x_i;\theta,1)'}{\phi(x_i;\theta,1)} &=(n-m) \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
-\sum_{i=1}^{m}\frac{-\phi(x_i;\theta,1)(\theta-x_i)}{\phi(x_i;\theta,1)} &=(n-m) \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
\sum_{i=1}^{m}(\theta-x_i) &= (n-m) \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
m\theta &= \sum_{i=1}^{m}x_i + (n-m) \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
\theta &= \frac{\sum_{i=1}^{m}x_i}{m} + \frac{n-m}{m} \frac{\phi(a-\theta)}{1-\Phi(a-\theta)} \\
\theta^* &=\bar{x}_{\text{obs}} + \frac{n-m}{m} \frac{\phi(a-\theta^*)}{1-\Phi(a-\theta^*)}

\end{align*}$$
Y con esto se llega al resultado esperado, y concluye la demostración. $\blacksquare$


```python
theta = 3
ns = [200,400,800,1600,3200]
aces = np.linspace(theta-2,theta+2)
def L_normal(a,mu,m,n):
    # Hyperparams
    hat = mu
    tol = 1e-5 
    maxiter = 300
    i = 0
    delta = 1

    while (np.abs(delta)>tol and i < maxiter):
        hat0 = hat
        hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
        delta = hat-hat0
        i+=1
    if i==maxiter:
        print("Se alcanzó el número máximo de iteraciones.")
    #print(f'θ estimado: {hat:.4f}, número de iteraciones: {i}')
    return hat

def run_iterations(ns,theta,aces):
    for n in ns:
        theta_hat = []
        for a in aces:
            x = sc.norm.rvs(loc=theta,size=n)
            #xm = x[x<a]
            #xn = x[x>=a]
            censored = x>=a
            m = n-len(x[censored])
            #n = k-m
            x[censored] = a
            xm = x[np.logical_not(censored)]
            mu_xm = np.mean(xm)
            theta_hat.append(L_normal(a,mu_xm,m,n))
        plt.plot(aces,theta_hat)
    plt.legend(ns)
    plt.yscale('log')
    plt.axvline(x=theta,linestyle='--',color='red',alpha=0.5,label='$\\theta$')
    plt.axhline(y=theta,linestyle='--',color='k',alpha=0.5)
    #plt.ylim(bottom=0,top=14)
    plt.xlabel('$a$')
    plt.ylabel('$\hat{\\theta}$')
    plt.title('$\hat{\\theta}$ vs valor de a')

run_iterations(ns,theta,aces)
```

    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:250: RuntimeWarning: overflow encountered in square
      return np.exp(-x**2/2.0) / _norm_pdf_C
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:14: RuntimeWarning: overflow encountered in double_scalars
      hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:15: RuntimeWarning: invalid value encountered in double_scalars
      delta = hat-hat0


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:250: RuntimeWarning: overflow encountered in square
      return np.exp(-x**2/2.0) / _norm_pdf_C
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:14: RuntimeWarning: overflow encountered in double_scalars
      hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:15: RuntimeWarning: invalid value encountered in double_scalars
      delta = hat-hat0


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:250: RuntimeWarning: overflow encountered in square
      return np.exp(-x**2/2.0) / _norm_pdf_C
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:14: RuntimeWarning: overflow encountered in double_scalars
      hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:15: RuntimeWarning: invalid value encountered in double_scalars
      delta = hat-hat0


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:250: RuntimeWarning: overflow encountered in square
      return np.exp(-x**2/2.0) / _norm_pdf_C
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:14: RuntimeWarning: overflow encountered in double_scalars
      hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:15: RuntimeWarning: invalid value encountered in double_scalars
      delta = hat-hat0


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:250: RuntimeWarning: overflow encountered in square
      return np.exp(-x**2/2.0) / _norm_pdf_C
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:14: RuntimeWarning: overflow encountered in double_scalars
      hat = mu + sc.norm.pdf(x=a-hat0)*(n-m)/(n*(1-sc.norm.cdf(x=a-hat0))) + (n-m)*hat0/m
    /var/folders/bh/tbkbc3qx5gvcp228jqgbmqfc0000gn/T/ipykernel_9545/3460605151.py:15: RuntimeWarning: invalid value encountered in double_scalars
      delta = hat-hat0


    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.
    Se alcanzó el número máximo de iteraciones.


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/ma/core.py:6849: RuntimeWarning: overflow encountered in power
      result = np.where(m, fa, umath.power(fa, fb)).view(basetype)



    
![png](Tarea4_files/Tarea4_5_11.png)
    


A comparación con el algoritmo $EM$, puede observarse que este algoritmo no converge en lo absoluto cuando $a<\theta$, aunque el $EM$ sí lo logre. Requiere que la censura sea algo mayor que la media para que el estimador se aproxime al valor real. Esto quiere decir que la aproximación no es muy buena, porque conforme crece el número de valores observados, entonces el segundo término de la ecuación tiende a 0, por lo que tiende a tener una menor importancia, y cuando son más valores censurados, el algoritmo da terribles predicciones de la media de la población, por lo que resulta mucho más efectivo el algoritmo $EM$.

## Ejercicio 3
Sea $X\sim \text{Exp}(\theta)$, donde $\theta$ representa el valor esperado. Sean $x_1,...,x_m$ los datos observados y $x_{m+1}>a,...,x_n>a$ los datos censurados por la derecha. La verosimilitud de esta distribución es la siguiente: 

$$ f(x|\theta) = \frac{1}{\theta}\exp(-\frac{x}{\theta})$$
Y su logverosimilitud: 
$$\mathcal{L}(\theta;x) = \log f(x|\theta) = \log\frac{1}{\theta} - \frac{x}{\theta }$$
Tomando todos los datos obtenidos, entonces se obtiene que: 
$$ \log f\left(\overrightarrow{x}|\theta\right) = \sum_{i=1^n}\log\frac{1}{\theta} - \frac{1}{\theta}\sum_{i=1}^n x_i$$
Y separándose en los valores observados y no observados: 
$$ \log f\left(\overrightarrow{x}|\theta\right) = n\log\frac{1}{\theta} - \frac{1}{\theta}\left(\sum_{i=1}^m x_i+\sum_{i=m+1}^n x_i\right)$$
Tomando el valor esperado: 
$$ \begin{align*} Q(\theta|\theta^{(0)}) &= n\log\frac{1}{\theta} - \frac{1}{\theta}\mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=1}^m x_i+\sum_{i=m+1}^n x_i\right) \\
&= n\log\frac{1}{\theta} - \frac{1}{\theta}\sum_{i=1}^m x_i-\frac{1}{\theta}\mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=m+1}^n x_i\right) \\
&= -n\log{\theta} - \frac{1}{\theta}\sum_{i=1}^m x_i-\frac{1}{\theta}\mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=m+1}^n x_i\right)
\end{align*}$$

Se sabe que los datos censurados son mayores que $a$. Entonces, la probabilidad de que sean mayores estos datos es la siguiente: 
$$\begin{align*}
P(X>a)  &= \int_{a}^\infty \frac{1}{\theta}\exp\left(-\frac{1}{\theta} x\right) dx\\
        &= \left.-\exp{\left(-\frac{1}{\theta} x\right)}\right|_a^\infty \\
        &= \exp{\left(-\frac{1}{\theta} a\right)} \\
        &= \theta f(a|\theta)
\end{align*}$$

Para conocer el valor esperado de los datos censurados, se sabe que vienen de una distribución normal truncada con soporte en $[a,\infty)$ y con tasa de $\frac{1}{\theta}$. Es decir: 
$$\mathbb{E}(X_i|Y_i) = \theta + \frac{\mathbb{1}_{[a,\infty)}f(a|\theta)}{\theta f(a|\theta)}$$
$$\omega_\theta = \theta + a$$

Y con esto se calcula el valor esperado. Se toma para el algoritmo EM la estimación con el último valor de $\theta$ Haciendo esta consideración, entonces, calculando la derivada, y evaluando en el siguiente punto ($\theta^{(1)}$):  
$$ \begin{align*} \frac{d}{d\theta}Q(\theta|\theta^{(0)}) 
&= \frac{d}{d\theta}\left(-n\log{\theta} - \frac{1}{\theta}\sum_{i=1}^m x_i-\frac{1}{\theta}\mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=m+1}^n x_i\right)\right) \\
0 &= -\frac{n}{\theta^{(1)}} + \frac{1}{{\theta^{(1)}}^2}\sum_{i=1}^m x_i+\frac{1}{{\theta^{(1)}}^2}\mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=m+1}^n x_i\right) \\
n\theta^{(1)} &= \sum_{i=1}^m x_i+ \mathbb{E}_{X\sim\text{Exp}(\theta^{(0)})}\left(\sum_{i=m+1}^n x_i\right)  \\
n\theta^{(1)}  &= \sum_{i=1}^m x_i+ (n-m)\omega_{\theta^{(0)}} \\
\theta^{(1)}  &= \frac{1}{n}\sum_{i=1}^m x_i+ \frac{n-m}{n}\omega_{\theta^{(0)}} \\
\theta^{(1)}  &= \frac{1}{n}\sum_{i=1}^m x_i+ \frac{n-m}{n}(\theta^{(0)}+a)
\end{align*}$$



Y su valor esperado es el siguiente: 
$$\begin{align*}
\mathbb{E}_{X\sim \text{Exp}(\theta)} &= \int_{a}^\infty \frac{x}{\theta}\exp\left(-\frac{1}{\theta} x\right)dx \\
                &= \theta\left[-\theta x\exp{\left(-\frac{1}{\theta}x\right)}\right]_{a}^\infty - \frac{1}{\theta}\int_{a}^{\infty} - \theta\exp{\left(-\frac{1}{\theta} x\right)} dx \\
                &= \left[-x\exp{\left(-\frac{1}{\theta} x\right)}- \theta\exp{\left(-\frac{1}{\theta}x\right)} \right]_{a}^\infty \\
                &= \theta{\left(\frac{a}{\theta}+1\right)\exp\left(-\frac{a}{\theta}\right)} \\
                &= {\left(a+\theta\right)\exp\left(-\frac{a}{\theta}\right)}
\end{align*}$$

Y con esto se llega al resultado esperado. 

Calculando el valor esperado de la logverosimilitud de los datos censurados: 
$$\begin{align*}
\mathbb{E}_{X>a} &= \int_{a}^\infty \theta\log\left(\frac{1}{\theta}\exp\left(-\frac{x}{\theta}\right)\right)d\theta \\
&= \int_{a}^\infty (-\theta\log(\theta)-\theta\frac{x}{\theta} )d\theta \\
&= \int_{a}^\infty (-\theta\log(\theta)-x )d\theta 
\end{align*}$$






```python
theta = 3
ns = [200,400,800,1600,3200]
aces = np.logspace(0,2,num=10)

def EM_exponential(a,x,m,n):
    
    s = np.sum(x)+(n-m)*a
    # Hyperparams
    tol = 1e-8 
    maxiter = 300
    i = 0
    delta = 1
    hat = np.mean(x)

    while (np.abs(delta)>tol and i < maxiter):
        hat0 = hat
        hat = (s + (n-m)*hat0)/n
        delta = hat-hat0
        i+=1
    if i==maxiter:
        print("Se alcanzó el número máximo de iteraciones.")
    print(f'θ estimado: {hat:.4f}, número de iteraciones: {i}')
    return hat

fig, ax = plt.subplots(1,2,figsize=(15,5))
for n in ns:
    theta_hat = []
    for a in aces:
        x = sc.expon.rvs(scale=theta,size=n)
        #xm = x[x<a]
        #xn = x[x>=a]
        print(x)
        censored = x>=a
        m = n-len(x[censored])
        #n = k-m
        x[censored] = a
        xm = x[np.logical_not(censored)]
        mu_xm = np.mean(xm)
        theta_hat.append(EM_exponential(a,xm,m,n))
    #sns.kdeplot(x=theta_hat)
    ax[0].plot(aces,theta_hat)
    ax[1] = sns.kdeplot(x=theta_hat)

plt.legend(ns)
plt.axvline(x=theta,linestyle='--',color='k',alpha=0.5)
plt.xlabel('$\hat{\\theta}$')
plt.title('Densidad de $\hat{\\theta}$ a diferentes valores de a y diferente tamaño de muestra')
ax[0].legend(ns)
ax[0].axhline(y=theta,linestyle='--',color='k',alpha=0.5)
ax[0].set_xscale('log')
ax[0].set_xlabel('a')
ax[0].set_ylabel('$\hat{\\theta}$')
ax[0].set_title('$\hat{\\theta}$ vs valor de a')

```

    [5.83455700e+00 1.48311749e+00 2.37777294e+00 9.41553105e-01
     4.56166639e+00 3.88169314e+00 2.02117122e+00 6.32380569e-01
     1.45692538e+00 5.68620649e+00 4.51397054e-01 6.86083071e-01
     3.99870370e+00 1.89120374e+00 4.51149130e+00 1.34005888e-01
     2.89664916e+00 2.11244819e+00 2.47430801e-01 4.84724832e+00
     1.43236674e+00 3.09354720e+00 3.38790832e+00 6.42709293e+00
     5.69584650e+00 1.86659468e+00 2.93652753e+00 1.52582998e+00
     6.25766713e+00 1.05350547e+00 1.07370246e-01 8.73970905e-01
     3.17297993e+00 2.47249044e+00 4.68807237e+00 5.53521269e-01
     2.69940096e+00 1.19483136e+01 4.93592369e-01 2.89952242e+00
     1.66676843e+00 9.89690803e-01 9.07327468e+00 2.67945183e+00
     9.62828932e-01 5.71792953e+00 1.15033708e+00 1.06554751e+00
     5.77833824e-01 2.81758818e+00 2.50810585e+00 8.33581148e-01
     3.28091662e-02 4.98205253e-01 4.00200712e+00 5.29295355e+00
     3.62233204e+00 1.46773759e+00 1.44805934e+00 3.92804020e+00
     6.19802920e+00 2.75981539e+00 2.23831481e+00 1.47211793e+01
     1.10596875e+00 4.96075345e+00 7.12971957e+00 2.38373779e+00
     6.06425143e-01 1.34395678e+00 3.19574159e+00 2.93170200e+00
     1.55613848e+01 6.60732989e+00 1.62933324e+00 1.66576838e+00
     1.87453644e+00 8.60275501e+00 4.78241818e-01 4.70101553e+00
     1.89960227e-01 2.91171038e+00 6.29277359e+00 4.14703135e+00
     7.47163619e-01 5.96171325e+00 5.91451914e+00 9.01657549e+00
     2.34551738e+00 7.20025577e-01 9.89939313e+00 1.53738585e+00
     2.89673803e+00 4.72300576e+00 1.61644357e+01 2.59952253e-01
     3.17757543e+00 1.18552900e-01 3.23413057e+00 6.05570987e-01
     6.73437386e+00 6.65890225e-01 1.27521581e+00 1.81650664e+00
     4.84197744e-01 1.81993726e-01 1.74139298e+00 1.44747225e+00
     1.00272234e+01 5.10599708e-01 7.96418276e+00 9.38543071e-01
     9.07423007e-01 9.98801097e-01 3.78137817e+00 7.98582726e+00
     4.59114223e-01 1.48540439e-01 2.63450851e+00 1.43798268e-01
     6.25466702e-01 7.85189273e-01 1.31596874e+00 5.40558197e+00
     5.76427275e-01 3.19960999e+00 8.06243009e+00 1.04084791e+01
     5.27924704e-01 1.61472474e+00 3.22239720e+00 4.21858308e-01
     6.07636777e+00 9.37612157e-01 1.40142737e+00 1.19621282e+00
     4.69943709e+00 1.44803927e+00 3.10995122e+00 4.76912584e+00
     2.58455103e+00 6.45705272e-02 3.23253831e+00 9.65610211e-01
     1.76371888e+00 9.58937861e-02 7.46639240e-01 3.60475803e+00
     4.23064119e-01 1.17306688e+01 2.14651375e+00 9.88998061e+00
     3.94148267e+00 6.16961666e-02 2.13932249e+00 6.80860899e+00
     1.07710953e+00 3.21708027e+00 2.10413731e+00 2.07929475e+00
     1.31120491e+00 7.89773992e-01 4.06197687e+00 5.12795167e-01
     2.49824212e+00 1.71938528e+00 1.33293175e+00 4.16811344e+00
     5.99940613e+00 3.98310700e+00 3.14856148e+00 4.12476845e+00
     4.61697834e+00 1.78424858e-01 1.26171597e+00 7.84636727e-01
     1.10267772e+01 2.84141367e+00 9.12509192e-01 2.67760975e+00
     2.64509415e+00 1.32732450e+00 1.20776419e+01 1.71878864e+00
     2.28546062e+00 4.88726953e+00 3.93556733e-01 2.22158504e+00
     6.55428169e-01 4.15975416e-01 5.90505106e+00 7.15117926e+00
     1.91698608e-01 1.19513330e+00 3.68285764e-01 9.58190264e-03
     2.42990557e+00 3.29967968e+00 4.62025338e+00 2.42420855e+00]
    θ estimado: 3.0285, número de iteraciones: 55
    [ 3.93971021  0.28849575  1.92991409  0.55738717  8.84011209  5.7516882
      2.65533557  4.94744907  5.39817504  0.27234752  0.96117477  2.82426523
      0.55371904  1.999141    0.12533843  2.70253411  4.27561384  5.07114522
      5.66099997  6.77236133  0.27834867  0.10613021  3.26449616  2.16735811
      2.78595081  3.54572479  3.15784617  4.109438    2.48798794  1.8611776
      1.47170593  1.37956543  0.27471678  1.89288912  0.10340987  0.75668041
      6.75110426  2.08422105 12.70198192  3.65570312  7.31563947  0.48824825
      0.1841318   2.53055285  0.95164035  0.26107559  0.90525749  0.2921902
     15.16346949 10.81290968  1.90828629  0.87396928  0.40244149  5.26650639
      2.91774326  6.55598721  2.50236843  0.17296886  8.17273681  0.20495058
      2.01967492  5.42776861  2.95142699  1.41494084  1.95196203  1.11800628
      5.47119078  1.4108626   0.70234677  1.51202595  0.77215214  3.98620621
      3.53111828  4.79645592  1.11734842  3.29044747  8.71572198  0.97018191
      0.15953164  2.45187229  0.52021611  2.22524884  9.74332495  1.51819548
      1.62405342  4.27727473  0.34545458  4.72041189 11.31407394  4.03738248
      2.87303246  0.2453758   7.05447782  0.72525336  0.9804678   1.58534935
      0.21868206  1.57918491  1.12433338  0.84597912  0.25655937  3.4642305
      2.21381914  0.73775214  0.32536103  6.26568674  3.36866677  1.67193662
      0.12643299  1.04978797 12.15551141  3.79668551  0.58278368  0.59991258
      4.42809748  1.46171855  0.80723693  0.82238435  2.49100609  1.28924196
      2.19378321  0.38301832  0.61598212  1.19758462  0.49365645 10.2171884
     10.63009847  1.96473089  2.84496567  2.70438372  2.11373746  1.65609904
      3.62224397  0.5229201   3.18496535  8.24115097  4.90288611  0.13873221
      0.9995986   7.84410132  4.03713437  1.76257229  3.68281455  3.39499813
      2.85340439  7.21538933  1.22808277  2.49002554 11.36249663  0.84857208
      0.19956929  7.96704443  9.57998855  0.34593723  3.33679477  1.40255816
      0.78751098  4.32828794  0.83606823  4.62304599  2.97398878  5.83317079
      0.27435074  0.37802295  2.56972449  5.55291285  0.83113412  4.17191947
      3.74983758  3.2087344   4.78298382  1.07995824  2.49231352  6.52411061
      3.36000358  2.90228122  1.13236874  0.65477575  1.85634338 15.14006745
      5.0713729   3.98752539  2.31263812  1.39888803  7.02161156  8.39233261
      2.98991692  0.79884865  1.15184573  9.16486789  2.77065378  0.29730763
      7.93587026  2.08869362  1.03725586  1.24101421  2.15571339  6.37127965
      2.04116764  1.79929339]
    θ estimado: 3.1608, número de iteraciones: 36
    [5.70249565e+00 4.07836884e-01 6.91607387e+00 1.80779005e+00
     3.94008039e+00 1.32492761e+00 3.49684837e+00 7.41639944e-01
     4.30486178e+00 2.51062472e+00 3.52273331e+00 8.72206574e-01
     4.27354601e+00 5.35478446e+00 1.75670329e+00 5.39313294e-01
     1.44073737e+00 1.15268746e+01 3.49296379e-01 4.27114947e-02
     1.56286025e+00 5.13509989e+00 8.19237341e+00 2.94999579e+00
     2.92423512e+00 5.69568665e-01 2.24587655e+00 4.99218031e+00
     5.68425756e+00 4.23583026e+00 4.09861142e-01 8.78316005e-01
     2.87858623e+00 1.98745224e+00 1.47168868e+00 1.77744456e+00
     1.31272404e+00 1.08140083e-02 6.30535705e-01 7.76429281e-01
     4.27273263e+00 1.93360397e+00 2.84897724e+00 9.42453365e-01
     1.70161595e+00 8.62773908e-02 2.76749996e+00 1.13641528e-01
     1.11740465e+00 1.27911199e+00 7.27986212e-01 4.01266244e+00
     2.20590376e+00 8.01934748e-01 1.97358007e+00 3.13554162e+00
     1.26526882e+01 2.20137739e+00 5.09901863e-01 9.02705062e-01
     3.38670356e+00 5.75316820e-01 3.87491112e+00 5.75144706e+00
     3.00598586e-01 1.43792116e-02 1.71718589e+00 6.77468901e-01
     1.84142228e+00 2.93676211e+00 1.09043927e+00 6.55846639e+00
     5.86669037e-01 4.32553848e-01 2.30781444e+00 2.21533554e-01
     4.08717833e-01 2.39289697e+00 8.75915055e+00 6.75372873e-01
     6.84298050e+00 1.97295557e+00 1.67512230e+00 9.34799430e+00
     8.77693311e+00 2.57094946e+00 3.61477203e+00 4.93072521e+00
     1.05728427e+01 1.71644402e+00 7.79746238e+00 2.53067376e+00
     1.42495749e+00 1.65013225e+00 5.29035842e-01 4.34315404e+00
     5.20670399e-01 7.54067055e-01 2.54327881e+00 5.25628320e+00
     7.97815159e+00 7.72272222e-01 7.84553792e-01 2.94516873e-01
     1.63834949e+00 6.04239041e-01 6.70178577e-01 6.91139434e+00
     9.80961431e-01 4.79924787e+00 7.80423585e-01 1.27557326e+00
     8.82051355e+00 6.80133834e+00 3.43774990e+00 2.03569873e+00
     1.88018161e+00 2.23311258e-01 2.35681499e+00 2.38700851e+00
     1.74216644e+00 4.85273821e+00 1.67581967e+00 1.25903512e+00
     5.25028111e-01 3.10327594e+00 5.28794303e+00 1.17331700e+01
     2.90584855e+00 3.91147065e+00 3.81141347e+00 5.26890477e-01
     7.57015348e+00 1.03283201e-01 2.21692246e+00 2.33996163e+00
     4.47131197e+00 7.36777664e+00 2.26668920e-01 4.68608833e+00
     6.15088578e-01 1.23728123e+00 1.71254529e+00 1.57435188e+00
     4.48530110e+00 3.75074283e-01 1.20839113e+00 6.41507837e-01
     5.75556293e+00 3.82568171e-01 1.61259138e+01 6.79916011e-01
     1.67231169e+00 1.07556294e+00 1.10817673e+00 1.08260443e+00
     5.34203144e+00 5.83406316e-01 7.44955993e-02 2.61877765e+00
     4.45381696e-01 5.23942014e+00 1.14704927e+00 2.89319596e+00
     2.75487494e-01 5.69030656e+00 2.04440073e+00 7.45753728e-01
     2.97193890e+00 1.90778784e+00 2.47540635e+00 1.73466851e+00
     7.03810131e-01 1.33424489e-01 1.52648729e-02 5.98511394e+00
     1.53190714e+00 5.92367507e-02 5.72052144e+00 2.39283252e+00
     6.26848802e-01 2.77935222e+00 9.19205178e+00 6.12656612e-01
     1.62088148e+00 1.19227305e+01 3.35446014e+00 2.97687517e+00
     8.25264270e+00 4.71611583e+00 5.77865179e+00 8.89024135e-02
     8.75512186e+00 2.83968833e+00 1.80500637e-01 8.94947039e-02
     5.44597279e-01 2.71116278e+00 2.16382906e+00 1.06361941e+00]
    θ estimado: 2.7637, número de iteraciones: 20
    [5.21939064e-01 1.05131560e+01 2.00941963e+00 2.12693770e+00
     3.47649381e+00 6.35223809e+00 4.35598863e+00 2.96194188e+00
     1.42813324e+00 4.29546774e-01 5.05050207e-01 1.65954575e+00
     1.34475910e-01 3.45932461e+00 1.69947526e+00 1.64039484e+00
     6.22646011e+00 4.87959636e+00 3.17225150e+00 3.67615048e+00
     1.98379705e+00 3.54109343e-01 6.02081040e-01 2.09943635e+00
     1.12557472e-01 1.07528384e+00 1.84499306e+00 3.07142298e+00
     2.66514037e+00 2.21120729e+00 6.37615965e+00 6.91985481e+00
     8.52300250e-01 4.87375050e+00 3.43138009e+00 8.53772645e-01
     1.74134085e-01 7.47423120e-01 3.45987821e-01 1.18284520e+00
     2.23008319e+00 3.19491772e+00 2.45306954e+00 5.84237293e-01
     3.39616789e+00 4.08737541e+00 1.34298122e+01 2.11680038e+00
     1.31920715e-01 7.96094110e+00 1.98080868e+00 2.66829482e+00
     6.88062108e-01 4.03713345e-02 2.15828691e+00 1.41364199e+01
     2.11117592e+00 7.57714792e+00 2.13953468e+00 8.43941432e-01
     9.55992758e+00 5.39881394e-01 2.38892369e+00 3.41269169e+00
     6.65690625e+00 4.03833672e-01 2.35627957e+00 8.25881377e-01
     4.65415458e-01 2.11540839e+00 1.23946208e+00 5.00771803e-01
     3.26358180e+00 2.17596571e+00 1.14388026e+00 6.25573823e-01
     5.41673824e-01 9.77874752e-01 2.03138021e+00 5.12225040e+00
     1.98748148e-01 3.07582586e+00 1.40928381e-01 5.61656883e-01
     1.15818499e+00 6.10531956e+00 2.11671352e+00 1.98380581e+00
     8.45125866e+00 3.98190477e+00 1.19609986e+00 2.61522268e+00
     5.36926178e+00 3.87318878e-01 4.81317375e-01 8.20409450e+00
     1.50680598e+00 6.07536916e+00 1.85127279e+00 3.11937489e+00
     8.85985055e+00 3.56786613e-01 9.47459872e+00 2.63769412e+00
     4.71704392e+00 7.73768067e+00 1.54699568e+00 1.06187492e+01
     1.00289926e+01 6.16013908e+00 1.86689793e+00 7.51693101e+00
     4.59705350e+00 2.34957790e-02 4.25693787e+00 1.37302756e-02
     2.18850735e+00 3.15201287e+00 1.58980268e+00 8.79911839e+00
     2.96466643e+00 1.91339607e+00 9.73259302e-01 2.91391590e+00
     2.70583913e+00 6.74214631e+00 2.22360444e+00 3.50228884e+00
     6.89249326e-01 7.21243013e+00 9.72887442e-01 7.61327621e+00
     3.74297977e+00 8.13888793e+00 1.01913773e+00 3.63074109e+00
     7.63648862e-04 1.51770568e+00 7.56974673e-01 5.11032224e+00
     7.23708823e-01 8.57170271e-01 8.77309618e-01 4.26760047e+00
     2.83696422e+00 3.52032900e+00 8.64290761e+00 1.18071223e-01
     2.84699911e+00 2.80300897e+00 1.27682282e+00 1.24938270e+01
     5.37728674e+00 4.59910906e+00 2.44183173e+00 2.24588174e+00
     1.05810050e+00 3.86001030e+00 6.24966641e-01 6.61436484e+00
     6.83566494e-01 8.65771884e-01 1.70057330e+00 6.40291904e+00
     5.13204436e+00 6.17628009e-01 7.24443770e-01 3.90771709e+00
     5.19366439e+00 6.02893459e+00 1.73739875e+00 1.91629137e+00
     3.40728571e+00 8.04814022e-01 1.32525234e+00 8.15077840e+00
     9.05536660e+00 7.44087062e-01 9.53881384e-01 2.20196454e-01
     3.82405165e+00 1.78215206e+00 3.73046615e+00 5.19236032e-01
     9.04626417e+00 1.11345350e+00 2.63699425e+00 1.18379275e+00
     3.83075716e+00 2.07393119e+00 3.11764034e-01 4.24773220e+00
     2.36004469e+00 1.00922700e+00 3.72764396e+00 2.99606787e+00
     5.27532094e+00 1.85137787e+00 1.87442814e-02 1.16172034e+01]
    θ estimado: 3.2086, número de iteraciones: 14
    [2.64187068e-01 3.54661375e+00 4.05779400e+00 2.17144737e+00
     1.19965310e+00 1.08607674e+00 1.75956155e+01 1.62809088e+00
     2.35270806e+00 7.22110441e-01 6.01563174e-01 6.93623325e-01
     1.91727740e+00 3.48549350e-01 7.83544576e+00 8.36514218e-01
     2.24118657e-01 3.53749161e+00 1.68277880e+00 8.15284618e+00
     1.29885781e-01 2.68192208e+00 2.30295411e+00 1.04104122e+01
     8.07814555e+00 1.74577201e+00 4.72318647e+00 7.88127297e-02
     1.39112593e+01 8.80671672e+00 2.42965805e+00 3.43394653e+00
     1.64595934e-01 2.87270055e-02 4.90174407e-01 2.14691634e-01
     2.29581644e+00 6.63924531e+00 1.43258949e+00 5.56331673e+00
     7.49524185e+00 5.30184127e+00 3.22069248e+00 2.69633884e+00
     2.68077809e+00 6.67074137e+00 2.04482929e+00 1.00403697e+00
     5.45619394e-01 3.75000634e+00 9.82663052e-01 1.33678484e+00
     4.40139478e+00 1.95187753e+00 4.40141967e-01 7.50352030e+00
     6.46363327e+00 4.01350623e+00 2.61204562e+00 1.05701111e+01
     2.38291098e+00 8.44565764e-01 3.75568528e+00 9.38646482e-02
     1.83413076e+00 2.55698638e+00 1.47928717e+00 9.85293585e+00
     3.28608909e-01 1.27456373e+00 2.33788013e-01 4.53717184e+00
     4.55921968e-01 1.19480566e+00 1.53155512e+00 8.05973344e+00
     1.25475013e+00 7.08720544e+00 1.29775747e+00 8.80470576e-01
     1.34905017e+00 7.32665998e+00 3.46456875e+00 5.83925221e+00
     5.06656849e+00 2.58776002e+00 8.85067583e-01 2.68617788e+00
     5.47646275e+00 2.43128333e+00 2.31951516e+00 8.16193891e+00
     6.88070972e+00 1.61035412e+00 9.56987308e+00 1.38477239e+00
     1.08806074e+00 1.31069658e+00 4.37968385e+00 6.58667549e-01
     5.29931246e-01 1.09618551e+00 2.28543838e+00 3.33688743e+00
     8.11638511e+00 8.83247138e-01 3.29015303e-01 7.10693449e-01
     2.09772380e+00 4.01979701e+00 8.88551811e-01 7.98479671e+00
     6.19172131e-01 3.30913835e+00 9.98615728e-01 1.57321211e+00
     1.69625224e+00 2.99030625e+00 1.95689366e+00 3.87627080e-01
     1.41065288e+00 1.46083560e-01 1.42525178e+00 9.74027564e-02
     1.38528502e+00 5.12807938e+00 8.26126881e-01 5.16703112e+00
     5.56821968e-02 1.41042149e+00 2.43295535e+00 1.26931615e+00
     4.04692063e-01 9.19131178e-02 4.24595691e+00 7.89247977e+00
     1.15419716e+00 1.74910108e+00 2.49604858e+00 4.08079622e+00
     9.09361179e-03 3.85049346e+00 8.81306463e-01 5.65857973e-01
     3.95666889e-01 7.06239680e+00 1.06461059e-01 4.93242360e-01
     5.47204755e+00 1.61710893e-01 5.03494098e+00 5.29908128e-01
     7.10788407e-02 4.63179197e+00 4.89905676e+00 8.30606097e-01
     3.47280149e+00 1.82395320e+00 1.30580136e+00 2.24544693e-01
     6.85864930e-01 9.21038035e-01 2.60501298e+00 1.57320738e+00
     3.78251181e+00 7.68813833e+00 5.76359595e+00 3.74820385e+00
     8.28101783e+00 8.97069368e+00 8.03463783e-02 2.82823307e+00
     1.94386065e+00 1.07668284e+00 1.28976414e+00 3.87777330e+00
     3.51077904e+00 2.74755247e+00 4.10127809e+00 8.82974677e-01
     8.12528601e+00 3.01650683e+00 5.97830112e+00 1.93455841e+00
     2.24095305e+00 4.40981801e+00 1.38361793e+00 5.44270013e+00
     6.87093687e-01 1.54948982e+00 1.05270325e+00 3.26629027e+00
     4.34556617e-01 4.14035714e-01 3.49158299e-01 3.04766521e+00
     2.59368437e+00 1.04486783e-01 2.04377517e+00 7.52067975e-01]
    θ estimado: 3.0318, número de iteraciones: 9
    [ 0.50577179  0.12409449  5.86687191  9.61394415  0.24487668  2.8672065
      5.21227945  1.58939233  2.36846971  1.22300914  4.38590859  0.06473229
      2.13355756  3.22527531 12.16490052  1.56748221  1.82060227  3.1687668
     12.57824778  5.77323225  4.91425454  1.18967902  0.659454    0.69683757
      3.36639394  2.07832137  2.71581157  1.73934167  1.44842221  0.4897134
      0.38034196  3.17938125  5.17353493  8.26022583  2.59060798  7.42168742
      0.22053454  4.43736514  6.7355398   0.07601244  4.00059164  2.63594462
      3.99377006  6.39346003  1.02465795  3.4707413   0.37092999  7.50195983
      2.09568799  5.2407937   1.44275504  6.22771141  1.91251089  0.10749178
      0.24157694  0.52390154  2.23566746  4.74735192  2.47244889  1.09749537
      6.17089404  0.26030456  1.24987632  5.35838945  4.0056476   6.41386726
      0.48398113  3.08449268  1.84931471  0.18029389  8.83448197  7.29792352
      1.30139253  2.2217465   1.80503403  0.09247552 10.82132733  7.3339582
      8.7530717   1.22621958  0.07734764  0.82624464  8.70716528  2.3904045
      1.95949773  4.27942702  3.5465846   0.39207429  0.05955044  3.29179134
      5.18967634  1.97502679 19.90909044  1.9523689   1.52651761  2.47526942
      4.22933976  2.3953543   5.41131535  6.85397201  0.19263685  2.13045067
      3.45354356 14.10896434  1.92366417 15.15516333  3.91104338 13.90201088
      0.97450937  5.54992018  2.97866359  2.63924956  0.74784948  1.87461936
      9.05567209  2.85988443  3.77661016  1.75483074  0.87006756  0.48542448
      6.20638154  2.6096065   0.48824459  4.58146542  1.28021851  2.0662388
      2.47055499  1.83962437  1.34748237  1.07215758  5.93879717  0.76919149
      0.926865    6.47735221  3.2781785   1.75475993  2.5367507   3.41573566
      0.94525706  0.20907807  1.33374897  0.29679161  0.16232214  4.77931322
      1.40442405  0.731507    0.28665446  0.81508422  4.36533186  0.11160467
      2.27886329  5.23800875  8.8505732   5.72596125  3.00429332  1.97746758
      2.12088936  5.55074714  0.376932    2.40700586  0.86264331  6.16262855
      0.53373502  0.89015525  0.04654568  0.80660517  0.11794754  0.87520521
      8.64918099  3.69897016  2.57882426  2.36542795  4.37871244  1.35109159
      4.45334151 10.53522355  6.94591937  5.32522499  0.92271468 10.76764498
      1.18581003  2.06680724  3.55840893  1.31927866  1.67250068  1.41999384
      3.73292114  3.04955837  0.04878026  1.46235858  2.64419794  5.38015852
      2.15741166  0.09492298  8.83830843  0.52295691  3.60414837  0.14538881
      1.04028294  0.72516023]
    θ estimado: 3.2862, número de iteraciones: 6
    [1.29508303e-01 2.16404858e+00 4.75851752e+00 1.06052215e+00
     2.86205571e+00 8.49021190e-01 5.67681747e-01 1.11639697e+00
     8.29587921e-01 4.71237060e+00 9.41350089e+00 2.26147981e+00
     1.03564309e+00 2.86394887e+00 5.24120831e+00 1.76253407e+00
     4.50108816e+00 1.14664264e+00 6.04806159e-01 3.42917834e+00
     3.73425370e-01 1.98719314e+00 6.22703398e-01 5.54812837e+00
     3.67230621e-01 1.24272607e+00 5.09894606e+00 4.11378722e-01
     1.27850129e-01 3.30506629e+00 2.81618714e+00 5.17852852e-01
     5.06325496e+00 1.18872366e-01 4.72863426e+00 3.72299058e+00
     4.63398445e+00 4.18575899e+00 1.16747406e+00 2.77900990e+00
     1.79433635e+00 2.76279672e+00 1.85432393e-01 4.14774673e-01
     9.91355477e-01 2.47919245e+00 7.85113600e+00 2.76576800e-01
     2.54840085e+00 1.04916911e+00 3.19829334e+00 1.17536739e+00
     2.72746697e+00 6.29428237e+00 8.61845403e-01 4.47809169e+00
     1.37266096e+00 8.14439376e-02 3.44402516e+00 1.45134668e+00
     2.33910710e-01 1.36570043e+00 1.12687216e+01 3.67048935e+00
     5.79335575e+00 2.36731402e+00 3.51089869e+00 5.13060156e-01
     6.55552008e+00 3.42482206e-01 1.64260911e-01 5.85989692e+00
     1.75151691e+00 3.09895680e+00 5.68481427e-01 4.95889869e+00
     2.81616013e+00 2.48460876e-01 5.03867157e-02 1.92377477e+00
     5.42426704e+00 4.78567687e+00 8.14018178e-01 2.50835041e+00
     6.59484409e-01 3.48246923e+00 1.21101619e+00 9.88225312e+00
     3.33233345e-01 1.69248541e+00 3.92112352e+00 3.72164713e-01
     1.94830217e+00 7.17838901e-02 3.23808387e-01 3.04965392e+00
     1.32085479e+00 6.32549907e+00 1.36555762e+00 6.73115229e-01
     2.68877300e+00 1.56215735e+00 1.19340898e-01 6.63428625e+00
     1.92254860e-02 2.77668113e-01 1.90148027e+00 6.63122964e-01
     2.79439181e+00 6.46825875e+00 8.74464562e-01 2.33805919e+00
     5.75644365e+00 3.15726012e+00 2.16365768e+00 4.73867318e-02
     1.20085372e+00 2.50727472e-01 7.73055088e+00 2.31459098e-01
     1.19424253e-01 4.99731007e+00 2.44074252e+00 3.09642918e+00
     4.53886220e+00 3.35634815e+00 4.44813474e+00 3.22990147e+00
     3.18043740e+00 1.18438086e+00 1.96128389e+00 1.02503647e+00
     1.37239275e+00 1.19888281e+00 1.39100299e-01 6.74992116e+00
     3.12214839e+00 5.14641788e-01 8.72349972e+00 1.63067138e-02
     1.10871223e+00 1.18925076e+00 1.80166771e+00 7.84882202e-01
     8.52788808e-01 1.16244725e+00 1.71659803e+00 1.09903892e+01
     1.59489064e+00 6.36540541e+00 3.32309594e-01 3.55320589e+00
     8.60555719e-01 1.03391673e+00 2.72975885e+00 5.96901081e-01
     1.18749996e+00 2.59941110e-01 1.29927536e-01 5.55961317e+00
     1.32205731e+00 5.86407899e-01 1.60434755e+00 2.14880561e+00
     3.96020066e+00 1.48034913e+00 3.79277723e+00 6.36525131e-01
     5.65896387e+00 1.09808626e-03 4.13293232e+00 5.05100365e-01
     1.30760450e+00 1.12831220e+01 8.46209787e+00 6.64861622e-01
     8.83706202e+00 1.13149852e+01 4.77772053e+00 2.51569472e+00
     2.93340087e+00 1.41961169e+01 4.90040675e-01 4.08355002e-02
     4.45565380e+00 2.52976103e+00 5.95003241e+00 2.21143275e+00
     2.97167388e+00 8.01396406e-01 7.91910163e+00 2.01349860e+00
     1.57542975e+00 2.77523582e-01 1.62667954e+00 1.18973519e+01
     2.51695474e+00 9.92590917e+00 3.37897879e+00 5.29590699e-01]
    θ estimado: 2.7607, número de iteraciones: 1
    [ 0.66221163 14.21244011  0.1283438   2.2651989   4.4894359   0.6514661
      2.7662616   0.28775618  4.21829272 14.79060358  1.16294458  3.41894376
      8.98961954  2.25665838  3.09048391  0.10264501  2.90287054  0.588298
      1.20796881  8.15780831  0.39384493  8.98347959  4.7990254   2.57276611
      9.94439553  5.45978845  1.26253031  2.75268829  1.46690365  2.09694364
      2.17379778  3.3396725   0.66078396  1.57783728  2.37992354  7.00200355
      1.04242626  3.92870124  0.5330089   0.9607122   0.93361555  4.53733609
      4.09244726  1.3700661   5.08933083 10.48619846 10.31622755  0.03622096
      0.45189191  0.41289559  3.85377334  2.0332859   1.03522522  0.04553015
      2.83144589  7.31567035  4.31458813  0.61239391  7.077983    3.32317855
      1.64663845  1.16614344  0.13851537  7.41905889 12.49775611  4.32510601
      1.06260499  2.8104352   5.33075597  6.18559101  4.63362136  0.46052442
      1.21660651  0.74669939  0.56549043  1.2156384   1.20500788  0.4547416
      6.27563275  1.36909226  1.12931186  4.51757994  1.74497887  2.74880571
      1.09777179  6.3014595   1.0677942   0.72792852  2.20524146  4.04283919
      0.84240124  0.34135086  3.34751386  1.15336002  6.93981272 12.9546568
      0.47572916  0.99187649 10.55156875  1.25246942  7.52681795  8.039274
      0.44472384  3.08665252  1.91390235  1.44023463  2.28028319  0.79862429
      1.65307993  2.04728227  3.34563136  3.60480707  1.84041896  1.33330372
      0.2687763   4.90488926  1.32880662  2.75101964  0.78772623  1.2250872
      2.2909206   2.33288887  2.57941731  2.88705344  0.93660061  6.48753335
      0.60947646  2.03418011  3.46979394  0.41592783 11.63027919  7.69696197
      0.72188677  2.45086259  5.48450965  6.26501314  3.00735788  1.36592191
      6.40532747  1.26176807  7.76550059  2.46349277  1.83809367  4.0714101
      4.1464221   4.12797727  0.96709814  0.18692606  4.91429946  3.02759907
      4.44271226  3.45314559  0.13384644  2.51380529  0.53976028  0.19390005
      3.3471585   6.08968209  2.14029871  0.40477677 12.47460546  2.44548377
      3.90683367  1.28490215  1.56299144  2.26753588  5.76433443  0.79876707
      0.82464986  1.65724371  2.66341035  0.80693005  0.80758032  0.75978299
      9.845383    2.983032    1.94300905  2.95615177  0.34154966  1.57911109
      1.36725596  6.90521032  1.98098364  0.27384679  3.75878822  0.72388599
      0.25676757  1.29535756  1.4446216   2.16514463  2.19571598  1.28622036
      3.55132064  5.83384498  0.20701571  2.73348919  0.74281981  0.58825815
      3.35789405  2.57807745]
    θ estimado: 3.0683, número de iteraciones: 1
    [2.07190446e+00 2.06916929e+00 9.41148328e-01 2.09301444e+00
     4.92025670e-01 5.12277061e+00 4.20770490e-01 4.79591446e-02
     5.66944736e-01 1.25886661e+00 3.69161597e-01 3.19840772e+00
     1.51129444e-03 4.45171846e+00 1.01858422e-01 8.21062299e+00
     2.76648789e+00 2.82217272e+00 7.03324295e-01 1.37919001e+00
     1.13601451e+00 1.39867556e+01 3.32524897e+00 6.39805910e-02
     5.46061639e+00 2.74438675e-01 7.18047340e+00 8.70709770e-02
     4.71110379e-01 1.08113208e+00 5.59787398e+00 5.25181606e+00
     1.87191157e+00 1.10692637e+00 7.82053965e-01 3.86900431e-02
     4.73095545e-02 2.59845494e-01 6.34322791e+00 2.49136094e+00
     1.49887134e+00 3.76601145e-01 9.00675855e-01 6.53865462e+00
     1.63881878e+00 1.68690659e+00 1.26348248e+00 2.40352988e-01
     1.68996810e+00 3.71195641e+00 1.86138722e+00 5.24298092e+00
     7.12349089e+00 2.01374090e+00 9.49293621e-01 4.04288186e-01
     3.40733595e-01 2.06316729e+00 1.42044760e+00 1.04411030e-01
     5.79986751e+00 1.69094129e-01 3.14316606e+00 2.72674095e+00
     2.88900549e+00 1.08857853e+01 9.55100729e-02 1.80420953e+00
     5.84961761e-01 8.13738190e-01 9.03746385e-02 1.10325719e+00
     3.05935374e+00 2.10253554e+00 7.66456463e-02 7.35955884e-01
     1.34751785e-01 1.17744938e+00 5.28590385e+00 5.37136843e+00
     2.83751841e-01 3.64302564e+00 1.33173960e+00 8.28178011e-02
     3.65201959e+00 8.57326602e+00 6.57418012e-01 3.23670817e+00
     9.60873041e-01 4.07586944e+00 1.02696368e+01 8.43113027e-01
     3.82819773e+00 1.89805962e+00 6.72197710e-01 6.29281613e-01
     2.83593756e+00 2.59386297e-01 2.49018479e+00 6.50214872e-01
     2.14466690e+00 3.56991945e+00 4.22609659e+00 1.64098074e+00
     2.59571944e+00 5.34107397e+00 2.75728446e+00 6.96377807e+00
     1.66225017e+00 3.75772319e-01 2.05306046e+00 1.14427958e-01
     5.55516617e+00 3.38585064e-02 7.00772430e+00 2.46800092e+00
     5.31320863e-01 1.19533981e+00 2.64367423e-01 6.43684429e-01
     4.80872406e-02 8.32808416e-02 2.56352105e+00 5.51174281e-01
     1.96638969e+00 1.57218534e-01 2.89793291e+00 8.94790369e-01
     2.65515018e+00 1.90429246e+00 2.80679185e+00 7.14874980e-02
     4.05632722e+00 9.89369031e-01 9.28543015e+00 5.44578302e+00
     7.38700385e-01 1.99670598e+00 8.23840830e+00 3.97311045e+00
     5.91731345e+00 1.63729640e+00 1.45653400e+01 5.26330957e-01
     1.16153426e+01 1.43793698e+01 1.01795339e+00 7.93651109e-02
     8.57794750e+00 5.52548744e+00 1.08673433e+00 6.05518773e+00
     5.36157493e+00 8.04350683e+00 1.04332954e+01 4.21491734e-01
     1.83688282e+00 1.65969371e+00 3.00781249e-01 1.19658019e+00
     1.61943974e+01 5.69509707e+00 2.50706712e+00 4.37364995e+00
     4.58800612e+00 2.60291183e+00 8.26774295e+00 1.45958128e+01
     1.27460660e+00 2.65610045e+00 1.56755145e+00 7.55157056e+00
     5.54594845e-01 4.10393168e+00 2.81103360e-01 9.55831846e-02
     1.73326258e-01 3.67574602e+00 4.33441930e+00 4.25284744e+00
     6.21036191e+00 1.32496298e-01 1.78160057e+00 1.33635034e+00
     1.70159533e+00 1.36347760e+00 3.61556156e+00 3.40866382e+00
     1.78619598e-01 5.85722912e-01 3.28610006e+00 1.26711325e+00
     5.22966108e+00 2.82887320e+00 1.00494302e+00 1.64167058e-01
     1.57728182e+00 4.15625869e+00 2.47713439e+00 1.63124289e+00]
    θ estimado: 2.8596, número de iteraciones: 1
    [3.59784310e+00 7.33857014e-01 1.52646204e+00 7.95657132e-02
     2.38227721e+00 2.62074875e+00 4.24054490e-01 1.09746192e+00
     2.23794127e+00 5.21158702e+00 2.51461668e+00 5.43624441e+00
     1.33432278e+00 3.70739134e+00 2.53004139e+00 2.19284576e+00
     1.14718213e+00 1.65029872e+00 4.15776509e+00 5.40155933e+00
     4.87043374e+00 4.54326784e+00 1.03688300e+00 2.83885660e+00
     2.54801077e+00 5.98053041e+00 1.15794064e+00 4.12414672e+00
     3.66356955e+00 1.48063051e+00 2.59938031e+00 5.25131101e+00
     1.58175793e+00 6.20006736e-01 7.33826233e+00 2.32253585e+00
     6.40315933e+00 8.90243847e-01 5.31788435e-01 1.34647750e+00
     1.67987712e+00 4.60432795e-01 1.21156152e+00 3.69384534e+00
     2.76920818e+00 6.97879854e+00 2.55167845e+00 1.87010488e+00
     1.23538001e+00 1.10775674e+01 4.03950525e+00 2.20913688e+00
     1.03879098e+01 3.36955264e-01 2.43575372e+00 4.40262555e+00
     9.60196204e-02 5.94261149e-01 6.57128809e+00 5.42920974e+00
     8.31154961e-01 2.68962900e+00 4.58654828e-01 9.79993155e-01
     5.27035223e+00 6.84821978e-01 9.63804438e-01 2.78621783e+00
     1.29931370e+00 1.21027368e+00 2.92642255e-01 6.11168970e+00
     1.27403069e+00 2.32608414e+00 8.38690626e+00 8.38371222e-01
     1.07959149e+00 1.36994603e+00 1.13577878e+00 1.78295370e+00
     2.54396212e+00 1.53504620e+00 8.96214420e-01 7.97342104e-01
     9.58033989e-01 4.61408632e+00 2.71306222e+00 4.18261071e+00
     3.49863910e+00 7.14444218e+00 1.54823785e+00 2.91784464e+00
     2.61516163e+00 1.99659195e+00 3.18756186e-01 1.13994998e+01
     4.36382723e+00 4.17809085e+00 6.41431716e+00 6.79668759e+00
     2.41059255e+00 1.24047015e+00 1.89291420e+00 3.57551041e+00
     1.11370810e+00 7.78714685e+00 2.09497911e+00 2.47992173e-01
     2.25272200e+00 2.35932468e+00 1.20710908e+00 3.15264112e+00
     1.50479554e+00 4.15470066e-01 2.06985071e+00 6.59256915e+00
     4.92604554e+00 1.42909050e+00 1.05006610e+01 1.37074340e+00
     4.85426672e+00 8.12626127e+00 1.68123689e+00 7.59907674e+00
     2.35523803e+00 8.13669526e-01 2.62927574e-01 8.42079231e+00
     6.84434206e+00 4.61000357e+00 9.88780041e-02 7.15022925e+00
     6.04657268e-01 1.24807399e+01 2.95391390e+00 5.15721091e-01
     6.55571780e-01 2.42908501e+00 2.99067447e+00 4.21554462e+00
     5.87464612e+00 1.19585002e+00 1.02799248e+00 1.63453245e+01
     5.02419247e+00 1.77238565e+00 2.07105891e+00 1.53576818e+00
     5.46773475e-02 1.21016030e+00 4.18587589e+00 1.25770532e-02
     2.76311572e-01 9.72566362e+00 2.62189303e-01 2.20481556e+00
     1.62745192e-01 5.68463650e+00 1.65926652e+00 7.86942871e-01
     5.01742510e+00 8.92192680e-01 5.83385392e+00 6.34357168e+00
     2.18624444e+00 1.04707685e+00 1.75554052e+00 3.56558066e+00
     3.53082531e+00 2.62863582e+00 2.38450636e+00 4.15677270e+00
     1.11876732e-01 5.39970621e-01 1.17035750e+00 5.73167631e+00
     1.72604610e+00 6.12773715e+00 8.30516254e-01 7.27623702e-01
     2.39734685e-01 3.84791943e-01 1.05245212e-01 4.01288863e+00
     3.01152563e+00 7.59333148e-01 9.02237962e-01 4.21165322e+00
     2.12215656e+00 2.77982077e+00 5.92079137e+00 2.84819706e+00
     4.42686903e+00 1.96408336e-01 2.67034267e-01 3.18711413e+00
     4.86572115e+00 4.54020303e+00 3.21255741e+00 6.14754865e-01]
    θ estimado: 2.9907, número de iteraciones: 1
    [2.20046231e-01 6.04234086e-01 2.36841792e+00 2.35456364e+00
     9.14470989e-01 1.99403260e+00 7.25203651e+00 4.26920017e+00
     1.30914752e+00 1.99171656e+00 8.18837806e+00 3.81816305e+00
     9.67211782e+00 1.04029853e+01 2.81006126e+00 1.78218512e+00
     9.14461777e+00 2.17909097e+00 7.21842930e-01 4.32088960e-01
     4.41369182e-01 4.49901288e+00 3.95411115e+00 2.22433899e+00
     3.31282452e-01 3.02791609e+00 4.59297721e+00 2.75726754e-01
     9.58130115e-01 2.16773123e+00 5.82362913e-01 5.24538888e+00
     3.71777834e+00 9.74087583e-02 1.80037507e+00 1.82403782e-03
     1.33493457e-01 1.40589510e+00 2.58825483e+00 5.32291285e+00
     4.46278228e-01 4.02932020e+00 1.31764766e+00 5.60547253e-01
     5.16318543e+00 1.09991931e-01 1.59230610e+00 1.16282941e+00
     3.92336337e+00 1.17498892e+00 1.22920790e-01 5.38136152e-01
     3.22994597e+00 3.51538713e-01 1.14624391e+00 2.80924898e+00
     1.60034492e+00 1.21927544e+00 8.18888968e-01 2.46702280e+00
     1.72499445e+00 6.25739494e+00 5.70750405e-01 3.48624028e+00
     1.59714601e+00 1.86189764e-01 1.52258124e+00 2.43346619e-01
     5.10715233e+00 2.35258833e+00 2.45875979e+00 2.24010679e+00
     8.91025579e-01 1.07769218e+00 5.98926043e+00 1.84919503e+00
     3.00669087e+00 3.53100289e+00 3.04379878e+00 1.91111255e-01
     1.26051842e+00 1.27303117e+00 2.88639168e+00 1.16850186e+00
     3.31217286e+00 6.73252899e-01 1.57148805e-01 1.72828741e+00
     3.70945147e+00 5.85980353e+00 3.01336534e+00 8.99485331e-01
     8.03586194e+00 1.59886633e+00 8.25129742e+00 2.08570736e+00
     1.82160038e+00 2.40107230e-01 3.24259035e-01 1.47654396e+00
     3.26225548e-01 7.73984035e+00 1.85322176e+00 5.88175194e+00
     7.41341935e+00 1.35435892e+00 6.93300877e-01 8.83696650e-01
     3.69086979e+00 1.61171252e+00 1.20585601e+00 3.21759251e-01
     1.14888681e-01 8.58031171e+00 2.21216557e+00 4.90677730e+00
     2.20563010e+00 6.85181341e+00 9.90127308e-01 1.27292963e+00
     7.20977951e-02 1.17159096e+01 1.14073200e+00 1.60331449e+00
     7.04490494e-01 1.03539556e+00 1.48946615e-01 4.91466219e-01
     1.08456435e+01 7.39287878e+00 4.47766035e+00 7.71738049e+00
     2.31568351e+00 6.49293325e+00 2.98417221e+00 6.71709303e+00
     5.55007044e+00 5.00259718e+00 2.06030067e+00 2.38933684e-01
     1.75333993e-01 2.75076801e+00 5.27333775e+00 1.03661424e+01
     5.77697848e-01 6.28077934e-01 2.19491020e+00 4.30524573e+00
     2.32674123e+00 3.78836044e+00 3.82112702e+00 5.94086201e+00
     7.19431712e-01 9.28450830e+00 1.57183280e+00 2.69229494e+00
     3.57870216e+00 9.37618128e-01 1.19749837e+00 3.39607795e-01
     1.33659280e+00 1.55957091e+00 1.52867156e+00 1.03006703e+00
     6.44863735e-01 2.83363983e-01 2.41268502e+00 3.55655659e+00
     2.85499782e+00 1.17482473e+01 9.74906190e-02 2.29725937e+00
     1.00206291e+00 1.15821671e+00 9.18547720e-01 1.22703957e+00
     3.33882005e+00 2.55916192e+00 2.07432595e+00 9.49939068e-01
     5.87172669e-02 1.36073917e+00 1.35552209e+00 2.64333325e+00
     8.72631160e-02 1.63020420e-03 5.75661941e-01 1.32364359e+00
     8.21220552e+00 5.21508102e+00 6.13626488e-01 1.22240427e+00
     2.51536523e+00 1.97530575e+00 1.38277386e+00 9.72649202e-01
     1.93199486e+00 3.36514003e-03 8.66260644e-01 1.81833124e+00
     1.97462249e+00 2.88377015e-01 1.96279563e+00 1.78181569e+00
     1.19558259e+00 4.86744892e+00 6.98057781e+00 9.84339118e-01
     2.37934797e+00 1.81072970e+00 1.93981538e+00 2.53026913e-01
     3.96407662e+00 6.46336319e-01 1.54807935e+00 1.81045675e+00
     3.55993053e+00 2.25311914e+00 3.62307442e+00 4.18333602e+00
     4.15957606e-01 9.67623571e-01 4.13986844e-02 1.81867032e+00
     8.47869043e+00 1.04107143e+01 8.61477592e-01 1.65913639e+00
     5.25208807e+00 5.78886686e+00 3.60493846e+00 5.50393826e-01
     2.66292464e+00 2.00512208e+00 4.51733609e+00 1.28384818e+00
     4.89628043e+00 6.91803769e-01 5.09505961e+00 1.15450044e+01
     3.05392955e-01 2.68857209e-02 6.37948520e-01 8.49934542e+00
     5.66492416e+00 2.24902078e+00 3.47356724e-02 3.08843023e+00
     6.08047733e+00 1.85063830e+00 2.96158811e-01 3.64063727e+00
     1.03810553e+01 2.23279600e+00 7.42238136e+00 1.17828667e+00
     4.45105120e+00 4.14095178e+00 3.36122568e+00 3.78975960e+00
     8.05758731e-01 2.38812663e+00 3.86856041e-01 2.15693921e+00
     1.51947809e+00 3.75784409e+00 3.66326452e+00 5.77106187e-01
     2.37768895e+00 1.36992138e+00 3.83264429e+00 2.07026021e+00
     1.10979387e+00 2.41243104e-01 5.41736059e+00 2.70008951e-01
     5.66685513e-01 6.24601858e+00 5.79658778e-01 7.94292618e-02
     2.74847392e+00 1.33007160e+00 2.71086168e+00 4.72163328e-01
     4.80327218e-01 5.21686056e+00 2.46113855e+00 1.74214244e+00
     9.28288803e-01 1.39571191e+00 7.66330105e+00 3.22879997e+00
     1.83286336e+00 1.77687719e+00 5.37607169e-01 8.78986987e-01
     1.89624090e+00 3.02622324e+00 9.39750575e-01 6.81945155e-01
     3.51642909e+00 1.05619959e+00 5.35532605e+00 2.10638411e+00
     1.54022630e+00 6.01396889e+00 4.59378744e+00 7.94112857e+00
     7.91142635e-01 4.80296676e-01 3.17858744e-01 2.44127465e+00
     3.45799346e-01 1.69530686e+00 1.02799199e-01 5.46470893e+00
     1.94322274e+00 5.14177781e+00 2.03362667e-01 2.79100333e+00
     3.79154548e-02 1.13864597e+00 7.92497624e-01 9.41804955e-02
     6.20961589e+00 2.70145235e-01 5.19009649e+00 7.27241640e-01
     3.20675350e+00 4.41534596e+00 9.78956458e+00 4.62885821e+00
     3.52590526e+00 1.31943661e+00 6.74965190e-01 7.03036902e-01
     3.08903810e+00 2.36115187e+00 1.92063869e+00 4.74616531e-01
     4.89409120e+00 3.96711961e-01 3.16885342e+00 2.01723880e-01
     5.20212596e+00 1.38911071e+00 1.71702994e+00 7.32523937e-02
     1.83783450e-01 3.04004129e+00 4.01891704e+00 2.48284618e+00
     2.81632623e+00 1.60450136e+00 8.23884114e+00 3.98379811e+00
     1.62615056e+00 9.02752375e-01 2.45551605e+00 2.41414345e+00
     4.52659153e+00 5.41094660e-01 1.08767463e+00 4.54447430e+00
     3.55909506e+00 1.94093719e+00 3.99813712e-01 8.64427019e-02
     1.79586637e+00 4.76545670e-01 2.07803127e+00 2.13751305e-02
     4.93135069e+00 2.78421788e+00 2.72000563e+00 2.41829009e+00
     3.09106998e+00 2.54066090e+00 1.93237960e+00 5.41411210e+00
     2.74596261e+00 4.11400923e+00 7.16375194e+00 1.26991905e+00
     4.12090801e-01 4.98775388e+00 7.92542586e-01 1.67726076e+01
     2.09230290e+00 1.54247457e+00 5.66328562e+00 3.41487488e+00
     1.12595085e+00 5.09836460e+00 2.63082870e+00 7.60929902e+00
     7.27906489e-01 1.47359898e+00 3.68047089e+00 1.85963230e+00]
    θ estimado: 2.8815, número de iteraciones: 54
    [3.10997233e-01 1.44339173e+00 1.52658634e+00 2.75028165e+00
     1.81962925e+00 6.08105170e-01 4.40851483e+00 3.48873636e-01
     2.90750061e+00 1.80152916e+00 1.82753585e-01 1.10983813e+00
     5.54848479e+00 1.16901918e-01 3.82149118e-01 1.10772138e+00
     4.67079657e+00 3.36950275e+00 9.18548213e+00 7.84251209e-02
     2.31099317e+00 2.11689977e+00 2.36340135e+00 3.10768308e-02
     1.89171217e+00 4.00709612e-01 2.83271339e-01 8.83349441e+00
     3.21812369e+00 4.58513480e+00 3.87707563e+00 2.34759426e+00
     2.29955352e-01 7.25620102e-01 2.29952583e+00 3.30273694e-01
     7.97871824e-01 5.61073024e-01 6.19561492e-01 6.18684760e-01
     3.95893887e+00 1.08552134e+00 7.12068845e+00 3.21391106e+00
     6.01870140e-01 7.34920580e+00 1.07945003e+01 5.07042568e+00
     3.11815138e+00 8.97006531e+00 1.37080011e+00 2.51077631e+00
     4.45190314e-01 3.11698836e-01 2.82300368e+00 4.09523721e+00
     4.47096462e+00 3.26496926e+00 4.37078030e+00 2.82974858e+00
     3.38962036e+00 4.10552792e+00 1.13952284e+01 3.58271273e+00
     2.73176666e-01 4.74465010e+00 2.26741826e+00 3.10270817e-01
     2.60174696e+00 2.86507701e+00 6.55593396e-01 1.27209658e-01
     2.82049925e-01 4.57157216e-01 3.62426343e+00 2.73098795e+00
     2.53295047e-01 1.19611251e+00 5.94388609e+00 7.12023047e-01
     2.03085229e+00 2.74492345e-01 1.21300291e+00 2.01246792e+00
     6.31450136e+00 1.96136757e+00 6.72751561e+00 2.88173125e-02
     4.61963217e+00 2.17196004e+00 1.39394608e-01 2.37549557e+00
     5.58752896e-02 2.13879362e+00 7.65327856e+00 7.51947486e+00
     4.59108662e+00 5.86336453e-01 4.83457662e+00 1.05076663e+00
     1.23304925e+00 2.12694954e-01 1.79335404e+00 1.67381899e+00
     5.22686944e+00 4.52714094e+00 1.67581798e+00 7.22837204e+00
     3.76734659e+00 2.81296336e+00 2.64586756e+00 1.32648597e+00
     6.75549460e-01 7.74117981e-01 7.91097122e-01 1.55410419e+00
     1.88901215e+00 1.84246539e-01 1.90160771e+00 1.76556327e+00
     5.44171770e+00 1.61019094e+00 1.59794032e-01 3.76918192e+00
     5.95195876e-01 3.29411288e+00 1.05250662e+00 6.75820616e+00
     1.14312240e+00 9.13073787e-01 2.42961372e+00 6.95469899e+00
     6.48065171e+00 6.77915922e-02 5.44534153e+00 1.17520741e+00
     1.49861808e-01 8.84650463e-02 1.28759297e+00 1.44784349e+00
     3.73468345e-01 3.81854776e+00 1.68586240e+00 1.78806516e+00
     1.69007877e+00 3.01602790e+00 2.46974568e+00 3.71665363e+00
     4.10821670e-01 3.61380839e+00 1.55841665e+00 2.59106452e+00
     8.72286990e+00 5.25393902e+00 2.94123344e+00 6.28141128e-01
     1.37908810e+00 1.05906573e+01 2.78478766e+00 3.86140926e+00
     3.15751275e+00 4.43510887e-01 2.97674330e-01 1.87567051e-01
     7.22265088e+00 2.62388959e+00 1.00113091e+01 3.05307716e-01
     5.52887977e+00 2.42617942e+00 9.91831064e+00 1.52340932e+00
     1.63732941e+00 6.70679867e-01 5.35675061e-01 2.22379720e+00
     2.49743943e+00 3.08781466e+00 1.02371583e+00 1.38446365e+00
     2.65928421e+00 1.45912310e+00 2.74933042e+00 1.08398671e+01
     3.06630307e+00 5.98909243e-01 6.97917125e+00 7.39605391e-01
     1.00836082e+00 3.18057726e-01 6.34673142e+00 3.31087194e+00
     8.04511519e-01 2.50445359e+00 9.33800091e+00 4.63258354e+00
     8.06025788e-01 4.23102449e+00 4.52943317e-01 5.45953404e+00
     3.85368425e+00 5.37462148e+00 1.31680613e+00 4.06296294e-01
     4.60747378e+00 1.15230737e+00 9.22894219e+00 1.43871986e+00
     1.55221201e+00 8.33210321e-01 2.69758310e-01 2.21659947e+00
     2.20564022e-01 3.90340519e+00 1.59619141e-01 2.32613615e+00
     2.50170347e+00 5.55223153e+00 6.59862416e+00 1.24097234e+00
     9.99178297e-01 1.31448460e+00 3.42221346e+00 2.40269263e+00
     4.27331878e+00 1.01102663e+00 3.35333115e-01 1.41272956e+00
     5.60030931e+00 2.03206524e+00 5.48357697e+00 3.86819558e+00
     9.24345671e+00 5.02562814e+00 5.78642236e+00 1.30791304e-01
     1.87131751e-01 3.80304890e+00 1.57319161e-01 7.81455053e+00
     4.30064376e+00 3.77359937e+00 1.47483469e+00 2.23662629e+00
     5.63365060e-01 1.38019205e+01 4.34506519e+00 3.59392668e-01
     1.50078455e-01 1.57891494e+00 1.05203128e+00 5.82641951e+00
     1.37271486e-02 1.94354615e-02 1.49429413e+01 5.67931811e+00
     1.98626690e+00 5.00080260e+00 5.65017212e+00 7.37516742e-01
     2.09486434e-01 6.84320604e-01 1.98436488e+00 4.96735147e+00
     2.35090901e+00 6.91774101e-01 1.59152804e+00 1.87334031e+00
     1.31183336e-01 4.43775066e+00 4.23517587e+00 5.08106200e+00
     7.42814115e-01 8.11059474e+00 7.22176456e-02 1.56234431e+00
     1.95870314e-01 1.85874864e+00 4.42913399e+00 5.04182686e-01
     2.29316615e+00 5.43789882e+00 1.87607914e-01 1.38786591e-02
     6.50793490e-01 5.67207054e+00 3.99559830e+00 2.00483895e-01
     2.65301037e+00 4.02905791e-01 5.27135553e-01 2.25480912e+00
     1.50363545e+00 2.67491802e+00 5.42948009e+00 7.02182429e+00
     4.11024498e+00 1.46732341e+01 3.85624591e+00 4.39982897e-01
     2.05368016e+00 1.96299648e+00 2.16011659e+00 8.90856775e-02
     2.42278276e+00 2.59338871e+00 5.55361476e-01 4.12293924e+00
     7.90644840e-01 4.67313046e+00 2.80650365e-01 3.14244853e+00
     7.90864877e-01 3.94282188e+00 5.39231059e+00 2.56022791e+00
     3.14725374e+00 2.17464956e+00 1.40124076e-02 8.39232546e-01
     1.10037078e+01 3.79641141e-01 6.51602535e-01 3.50506550e-01
     6.50624469e+00 4.07761589e+00 4.60156718e+00 2.76894901e+00
     9.44752304e-01 3.41807183e+00 4.37349647e+00 7.94273583e-01
     2.53321166e+00 1.92152597e+00 2.58292333e+00 5.52442466e-01
     4.55590347e+00 2.62063328e+00 8.29745587e-01 5.80690520e-01
     1.88801553e+00 2.31402973e-01 3.66482131e-01 1.09748914e+00
     5.18736906e-01 6.08283509e+00 1.65169191e+00 1.01534420e+00
     3.66865294e+00 2.02770334e-01 1.97479144e+00 8.05117507e+00
     8.38295225e-01 4.68314248e+00 4.87333338e-01 6.42782132e+00
     5.97236473e+00 1.23468890e+00 1.60165761e-01 1.00143962e+00
     3.68059241e-01 4.12836808e-01 3.64632785e+00 7.23947162e+00
     1.37575330e+00 3.01821811e+00 1.55130480e+00 2.49674863e-01
     8.67566605e-01 1.15138833e+00 1.61377505e+00 5.37821050e-01
     4.83013797e+00 8.80548308e-01 1.91814963e+00 7.75281024e+00
     4.66960572e-01 1.16796716e+00 1.12071218e+01 4.18558566e+00
     2.73849523e-01 8.83023002e+00 3.47595984e+00 1.68373717e+00
     2.66858346e+00 2.79513725e+00 1.83360386e-02 1.69095754e+00
     1.28951494e+00 5.76768015e-01 2.63932030e+00 1.92800796e+00
     3.27816889e-01 2.95917359e+00 3.40830382e+00 5.40443544e+00
     4.15803140e+00 8.13688181e+00 1.36251582e+00 3.61029666e+00]
    θ estimado: 2.8326, número de iteraciones: 33
    [1.57596902e+00 5.30533958e+00 1.63812539e+00 2.90365793e+00
     1.61089455e+00 3.59312035e+00 2.29884015e+00 2.78826691e+00
     1.86463552e+00 3.08196435e+00 6.59406518e-01 3.54588277e-01
     3.31901795e+00 9.39895514e-01 1.03736929e+00 7.55333285e-01
     1.72929450e+00 2.98313474e+00 4.24841879e+00 7.38249201e-02
     3.31174097e+00 8.16894534e-01 1.07475121e+01 1.85033453e-01
     1.68607986e-01 5.63364236e+00 1.25004617e+00 8.20475827e-01
     5.18603083e-01 3.13135089e+00 7.38897259e+00 5.55830302e+00
     5.22346374e+00 1.74209591e+00 4.71980685e-01 4.60107093e+00
     2.19272927e+00 1.86922438e+00 4.68806446e+00 1.80252131e+00
     1.48338443e+00 2.68496452e+00 9.12106059e-02 1.52116227e+00
     1.25608679e+01 2.96968300e+00 9.24897886e-01 1.76573301e+00
     1.22716183e+00 1.25035956e+00 3.36258859e-01 4.60464218e+00
     3.75437132e+00 1.38471628e-01 1.66258471e+00 7.84491453e-01
     8.08027209e+00 3.35696381e+00 2.72406426e+00 1.35037191e-01
     8.26586308e-01 1.92592096e+00 1.88229582e+00 2.60392260e+00
     3.07133229e-01 1.87214322e+00 3.21636816e-01 2.51556438e+00
     9.63606597e+00 3.73089403e+00 2.96952848e+00 1.00994213e+01
     6.35654911e-01 2.14635513e+00 3.72933524e+00 1.45188326e-02
     5.78311245e+00 1.62472534e+00 1.69760323e+00 2.98372570e+00
     5.73743302e+00 5.07469830e+00 1.44306916e-01 9.94416031e+00
     7.10684065e+00 9.77413851e+00 1.54255520e+00 7.57682963e+00
     1.93921388e+00 2.60005688e+00 7.84233831e+00 2.35399980e+00
     5.97827202e-01 7.42854426e+00 5.96750284e-01 9.64654315e-01
     7.62401881e-01 8.53564040e+00 3.45492522e+00 3.66921416e+00
     1.54061931e+00 1.36255623e+00 2.60551216e-01 6.79557303e-01
     1.24328293e+00 4.87307033e-01 2.47598204e-01 4.62581144e+00
     1.22467548e+01 1.44207515e+00 1.33605640e-01 2.08088744e+00
     3.42847479e+00 1.64091821e+00 5.53038448e+00 2.54720261e+00
     1.25438281e+00 7.77676994e-01 2.24344996e+00 3.96692040e+00
     3.00390927e+00 6.62760375e-01 5.65288594e+00 2.13258995e+00
     1.10814698e+00 9.85432722e-01 5.47303408e-01 6.04481941e-01
     3.83905763e+00 2.23019286e+00 7.03239666e-02 2.91326545e+00
     4.54174030e-01 1.08007539e+00 6.90739968e+00 3.03710834e-01
     5.51965806e+00 8.49535540e-01 3.03490276e-01 1.10764047e+00
     2.13112297e+00 2.53451134e-01 2.17041322e-01 1.10680794e-01
     8.09908660e+00 6.68427113e+00 1.47410086e+00 5.26615001e-01
     8.57945042e-02 5.92104329e+00 3.52512916e+00 5.83948010e+00
     7.57086155e-01 3.38645580e+00 4.97027041e+00 2.67191561e+00
     3.82630805e+00 1.65994593e+00 9.06973089e+00 9.54928183e+00
     9.16265799e-01 6.58574328e+00 6.94931506e+00 2.45527640e+00
     6.15448852e+00 2.27183166e+00 2.56752691e-01 6.62359094e+00
     1.85773894e+00 3.29901640e-01 2.87043224e+00 4.47016206e-01
     4.33393416e+00 1.31454482e+00 4.73188453e+00 9.33147156e-01
     1.92204797e+00 1.04948055e+01 9.27046093e-01 3.85049864e+00
     3.02782617e+00 5.38153213e+00 5.23773132e-01 2.40095969e+00
     2.10426240e+00 5.32273410e+00 1.87979640e+00 5.03016070e+00
     2.48232156e+00 6.09091310e-01 4.44749609e-01 1.59820807e+00
     8.05922383e-01 3.23339406e+00 2.29769639e-01 1.71537293e+00
     1.98472483e+00 3.37658235e+00 1.61184429e+00 2.25703044e+00
     8.43307126e-02 7.39120625e+00 6.96047442e+00 1.71010022e+00
     1.23920954e+01 2.49780745e+00 1.44185494e+00 4.99786193e+00
     1.81154708e+00 4.20751841e-01 5.39496866e-02 3.13506892e+00
     5.45037836e+00 4.87995960e-01 2.69298884e+00 1.41979058e+00
     6.03538206e+00 3.40539404e+00 5.05553625e-01 2.69004721e-01
     1.29288596e+00 2.25523534e+00 7.00197084e-01 5.27029146e-01
     1.60343706e+00 4.50357988e+00 5.46231556e+00 1.94314907e+00
     8.81699935e-02 1.00480715e+01 1.79535677e+00 6.81549167e-01
     6.59110847e+00 3.21348360e-01 1.06928096e+01 1.29591472e+00
     2.20776027e+00 3.90009254e-02 1.18380833e+01 5.76083741e-01
     2.06958762e+00 1.55634583e+00 2.67404233e+00 4.70942205e+00
     1.38176024e+00 2.33896691e+00 7.55983542e+00 8.85365170e+00
     3.81453270e+00 7.86501697e+00 8.98101807e+00 1.10736246e+00
     2.20235420e+00 2.45859479e+00 1.64279393e-01 3.33071398e+00
     9.06715700e-01 1.88869145e-01 5.65845258e+00 2.35413274e+00
     2.34952687e+00 6.60344346e-01 9.69950533e+00 1.35633317e+00
     9.85768035e-01 2.03599963e+00 3.30633837e+00 1.82170741e+00
     9.14186869e-01 2.33134836e+00 2.23826578e+00 6.49615137e-01
     1.31247857e+00 4.73875941e+00 4.95906040e+00 1.75714641e-01
     6.34078809e-01 1.89767530e+00 1.27540912e+00 2.01488576e+00
     8.97284190e+00 2.32627479e-01 2.68004417e+00 7.83328600e+00
     3.73599632e+00 3.71214697e-01 1.54025499e+00 1.71395172e+00
     5.98310564e-01 3.62910294e+00 4.09494215e-01 5.16819532e-01
     5.05204697e+00 9.96315654e+00 9.05897073e-01 6.81172030e+00
     3.43889293e+00 6.13067954e+00 3.53789535e-01 4.63975715e-01
     3.56431485e-01 1.64844905e+00 3.24969024e+00 7.46046647e-01
     2.40440638e+00 1.50468561e+01 2.27721058e+00 4.89809999e+00
     3.64425300e+00 1.01507287e+00 1.00680976e+01 3.00416170e+00
     2.00919936e+00 6.50004586e+00 4.66504037e+00 1.76649487e+00
     5.75895450e+00 3.88179680e-01 2.13552113e-01 9.07350772e+00
     5.25582518e-01 1.73871533e+00 7.79763220e-01 2.92889445e+00
     5.45431173e+00 3.13987418e+00 4.06253006e+00 4.10282451e-02
     1.09136794e+00 1.82705406e-01 7.31413523e+00 3.68282519e-01
     5.33440226e-01 3.69044833e+00 2.79633745e-02 1.01319099e+01
     1.01907289e+01 5.42038987e-01 1.26315289e+00 1.31802847e-01
     5.42400819e-01 3.44630700e+00 9.10248972e+00 2.51662864e+00
     2.21049540e+00 6.57355759e+00 3.87158359e+00 1.71806697e+00
     1.92194087e+00 4.30314790e+00 1.42947315e+00 5.13390195e+00
     1.08293897e+01 2.99711471e-01 5.76653966e+00 3.70860436e+00
     2.42004695e+00 2.69183697e-02 9.50768770e+00 6.27518281e-01
     2.61529945e+00 3.33925986e-01 9.88950678e+00 1.52937330e+00
     8.49377316e-01 4.88133743e+00 5.27583489e+00 2.57538287e-02
     4.02118717e+00 2.02437040e+00 8.49949585e-02 4.45458043e+00
     1.93710229e+00 1.81024326e+00 2.59060038e+00 5.26684336e-01
     2.38913592e+00 2.37408804e-01 5.02841404e+00 3.76951789e+00
     8.84371433e-01 7.53856163e+00 2.81928800e+00 2.92030103e+00
     2.11241281e-01 1.95063132e+01 1.48700908e+00 1.24225943e+01
     1.93995837e+00 7.34742275e-01 2.68942275e+00 3.47103119e-01
     5.41164154e-01 8.88873588e+00 2.88793682e+00 1.33924053e+00
     2.78090751e+00 9.00949118e-01 1.44299228e+00 2.06696457e+00]
    θ estimado: 2.9459, número de iteraciones: 21
    [9.80302392e+00 1.42859876e+00 6.04334284e+00 6.65747769e+00
     1.74533483e+01 9.91990103e-01 4.86973326e+00 1.63851014e+00
     3.60762904e+00 5.16751711e+00 2.78480121e+00 7.13088263e+00
     2.14985265e+00 1.63498062e-01 3.07094408e-01 2.48741656e+00
     1.00350622e+00 1.26373901e+00 6.56182075e+00 6.08790104e+00
     2.34572563e+00 1.84150470e+00 4.70801258e+00 3.11715944e+00
     6.51490004e-02 2.61581192e+00 1.98892865e-01 2.50170454e-01
     1.12393055e+01 5.94729101e+00 1.02372086e+00 7.04618612e+00
     6.94900938e+00 3.46322716e+00 1.17984768e-01 2.60011228e-01
     4.20575676e+00 2.19532471e+00 9.51783269e-01 8.97360073e-01
     1.42674850e+00 3.14430421e+00 3.87899411e+00 3.45730025e+00
     4.94430663e+00 3.58970848e+00 3.94646302e-01 1.11398783e+00
     8.85523713e-01 1.36118068e-01 3.29549106e+00 5.98261853e-02
     1.51552660e+00 7.49381977e-01 2.21764241e+00 1.63443323e+00
     3.25672129e-02 7.12644551e-01 1.15589315e+01 4.94958932e-01
     3.85588336e+00 3.78350154e+00 3.70363086e+00 8.93950123e-01
     7.12362090e+00 1.39731170e+01 3.30641002e-02 1.89145174e+00
     2.01628524e-01 1.16308982e+00 6.57553644e+00 2.50039194e+00
     3.33285505e-01 2.66526504e+00 1.33347471e-01 1.26929598e+00
     6.79839027e+00 9.17578838e+00 6.37301897e-01 7.99385612e-01
     3.19950996e+00 3.75173754e+00 4.21363881e+00 2.89960236e+00
     1.59172600e+00 5.31732345e+00 4.65349106e+00 8.40289616e-01
     1.69715131e+00 3.44979727e+00 1.67500700e+00 2.20198253e+00
     6.62521409e+00 1.01231064e+00 1.57288705e+00 7.17104378e-01
     2.53141946e+00 1.16864644e+00 1.31063376e+00 1.56556411e+00
     3.44449962e+00 1.74306235e+00 7.61160253e+00 1.31167991e-01
     1.22060081e+00 7.34909904e+00 4.22999717e-01 2.14141525e+00
     1.71945109e+00 5.05534155e+00 2.11237406e+00 2.54250458e+00
     3.74948209e-01 8.88429818e-01 1.61707189e+00 1.86795375e+00
     4.52835435e+00 1.70227270e+00 1.51291140e+01 1.75052985e+00
     5.45423572e+00 1.57510211e+00 1.35360987e+00 3.72384910e+00
     9.35798235e+00 4.23255074e+00 3.27254733e-01 1.99710562e-01
     2.14884347e+00 6.73227053e-03 6.32857389e-01 1.20849154e+00
     3.70599644e+00 1.62651338e+00 2.63760677e+00 2.95703617e+00
     1.49973436e+00 5.10705501e+00 2.02830728e+00 4.39639293e+00
     7.24377772e+00 8.25373946e+00 6.26104593e+00 6.13471986e+00
     5.70470108e-01 1.88420010e+00 2.59805608e+00 7.89638903e-01
     1.62818432e+00 3.37232496e+00 1.14780158e+00 1.07354529e+00
     3.77461295e+00 1.73375746e+00 3.54550328e-02 3.98902241e-01
     6.21160697e+00 8.24682249e-01 3.66187983e-01 3.06283752e+00
     1.62829407e+00 5.30768519e+00 2.57517231e+00 5.50130952e+00
     4.26180901e+00 8.45676406e-01 3.06027841e+00 2.17366498e+00
     6.28700613e+00 8.85461619e+00 3.65170816e+00 3.30312927e-01
     2.00347645e-01 3.13147980e+00 2.26629796e+00 1.90367669e+00
     1.77962412e+00 9.18105818e-02 3.71557203e-03 2.35085622e+00
     1.33073443e+00 1.04673912e+00 2.35179966e+00 1.78416005e+00
     9.88067597e+00 1.11091305e+01 1.35835421e+00 1.28196318e+01
     6.94625372e+00 3.30883496e+00 1.67226615e-01 1.30337206e+00
     9.77238373e-01 3.21354065e+00 4.14134374e+00 3.03062929e+00
     6.87904540e-01 1.13967771e-01 2.73189385e+00 3.76778037e-01
     2.43638957e-01 2.90817001e+00 2.08306990e+00 9.65947825e-01
     2.68196423e-01 1.05398305e+00 9.51635357e-01 2.32456152e+00
     4.52563045e-02 3.25732752e+00 8.44675630e+00 1.05003681e+00
     1.43120307e-02 1.75319327e+00 4.59039441e+00 3.35810397e+00
     5.23589910e+00 1.70856807e+01 1.91350473e+00 2.38414782e-01
     1.51445183e+00 3.42347560e+00 1.03289460e+00 3.15236887e+00
     2.77138697e+00 2.65256273e+00 1.84889495e+00 1.72518998e+00
     1.46451380e+00 4.21284379e-01 3.57568650e-02 9.56407734e-02
     6.79877315e+00 7.87126353e-01 3.86596701e-01 5.60358566e+00
     7.02340864e-01 1.70939256e+00 3.32252886e+00 1.83339638e+00
     5.67660125e+00 4.19400288e+00 8.53679414e+00 5.01277092e+00
     7.35130811e-01 4.23375346e+00 3.47432179e+00 5.22759207e-01
     2.01976787e-01 2.53416694e+00 2.04495196e+00 6.08817678e+00
     5.81221310e+00 5.75203495e-01 4.00517114e+00 3.55244435e-01
     4.74159907e+00 1.13773333e+01 1.00619605e+00 5.68292166e-01
     1.85174630e+00 7.92765174e-02 1.73333215e+00 8.75707970e-01
     4.48468673e+00 5.94899585e+00 1.76759770e+00 2.05679079e+00
     1.08538467e-01 3.49994795e+00 5.02474734e-02 1.54710316e+00
     5.65330445e+00 4.81254597e+00 3.17724196e+00 3.90253994e+00
     4.57048690e-01 4.17884932e-02 1.79429297e+00 8.87649388e-01
     3.99916251e+00 5.03192839e+00 3.69643364e+00 2.38137243e+00
     6.87496630e+00 5.75351520e-01 1.29931415e-01 3.04326704e+00
     2.93734019e+00 1.31697484e-01 1.31780859e+00 7.63782333e-02
     6.02084572e-01 3.01829368e-01 2.63801930e-01 3.74086698e-01
     7.47880885e-01 1.20273028e-01 6.04231165e+00 5.23867812e+00
     5.75786311e+00 1.39118641e-02 2.03418494e+00 4.77662961e+00
     1.22284794e+00 7.69215156e+00 1.54472177e+00 1.01384593e+00
     2.33189929e+00 2.42764187e+00 5.50896673e+00 2.42928308e+00
     1.97737314e+00 1.07159527e+00 1.31821930e+00 1.81584494e+00
     5.02420617e+00 7.29340569e+00 6.90420936e+00 1.34496279e+01
     5.93305820e-01 2.92715932e+00 8.63801725e+00 7.40095634e-01
     7.85966972e-01 1.20474259e+00 1.17283587e+00 1.24196249e+01
     2.70136593e+00 5.25901889e-01 1.36954547e+00 2.48368884e+00
     1.94939339e+00 7.60261847e+00 5.48427608e+00 1.52569055e+00
     5.56612282e+00 3.10762581e+00 2.16969974e+00 4.37729889e-01
     2.37731544e+00 5.39610649e-01 8.73677420e+00 9.11951216e-01
     4.64397939e-01 1.42239549e-01 7.39138906e+00 1.02165474e+00
     5.83869022e+00 2.40190767e+00 2.37805059e+00 2.10351255e+00
     2.34460681e+00 5.35662199e-01 2.24175152e+00 4.03168047e-01
     2.31502506e+00 1.21588737e+00 9.81342877e-01 2.74973183e+00
     1.45527171e+00 1.07787754e-01 9.03958229e-01 1.33654483e+00
     3.26286877e-01 3.23890712e+00 4.81252808e+00 2.14046343e+00
     3.03132782e+00 7.62431579e-03 3.07163998e+00 1.17157713e+01
     6.68662623e-01 4.04157020e+00 6.87425616e-01 1.90322942e+00
     9.98950943e+00 1.54496473e+00 5.34366966e-01 1.04929373e+00
     7.76024778e-02 5.96859645e-01 1.93096935e+00 9.65027360e-01
     1.63690623e-01 4.98218275e+00 1.68972276e+00 7.07919304e-01
     7.73165480e-01 3.05613903e+00 1.52525203e+00 1.55350766e+00
     1.94503805e-01 6.21810602e+00 3.67893084e+00 8.69754232e-01
     3.84505133e-01 6.42992780e+00 3.74571355e+00 3.17304372e+00]
    θ estimado: 2.9090, número de iteraciones: 13
    [4.11129401e+00 4.38261162e+00 7.31895725e-01 1.37751590e+00
     8.40846156e+00 3.29232865e+00 6.90614332e+00 7.16783274e-01
     9.74919630e+00 6.00503965e+00 1.47646195e+00 2.29764758e-01
     2.60832184e-01 2.15935457e+00 9.21964225e+00 1.52724877e-01
     9.13098456e+00 7.80914677e+00 3.03657635e-01 4.06977997e-01
     6.86165772e+00 3.65166260e+00 5.08108100e-01 3.62282509e+00
     1.35951551e-01 1.20052552e+00 7.03231454e+00 7.19106418e+00
     6.20865659e+00 1.03609045e+00 5.46527052e-03 1.81179765e-01
     6.37363046e-01 4.22199712e+00 1.26477102e-01 1.99329988e-01
     5.93829014e+00 2.86805545e+00 3.70720691e+00 2.13580353e-01
     2.99353545e+00 3.72271266e+00 9.02497166e-02 1.61563019e+00
     1.32774181e+00 1.77738633e+00 1.44060612e+00 1.43441168e+00
     8.89325499e-01 5.06584581e-01 2.54071863e-01 5.00064523e-01
     1.73369810e+00 2.89635244e+00 7.18098603e-01 1.61943460e+00
     5.94418473e-02 4.83542002e+00 4.09770797e-02 1.92320749e+00
     1.63322843e+00 8.34035844e+00 3.15565144e+00 5.35026142e+00
     8.80673764e-01 7.03034851e-01 3.96444436e+00 1.12028620e+01
     1.64719777e+00 4.86429524e-01 1.60732185e+00 7.38028783e-01
     1.90456120e-01 7.60261152e-01 4.12657768e-01 1.83763371e-01
     3.93843396e-01 2.06878078e+00 2.07501461e-01 8.50792397e-01
     1.21236074e+01 6.49369112e+00 1.36044085e+00 3.13934686e+00
     1.58734189e+00 1.31136354e+00 3.34862073e-02 4.34246980e+00
     4.86734581e-01 1.42965576e+00 5.66844589e+00 1.35338498e-01
     4.78447381e-01 1.02850804e+01 8.36629894e+00 3.80905239e+00
     2.77792416e+00 1.36862822e+01 3.42189219e-01 1.56360781e+00
     2.59771675e+00 3.23461791e+00 9.34623702e-01 1.67287803e+00
     7.62501438e+00 1.68002155e+00 1.72544103e+00 1.41090992e+00
     2.87335986e-01 1.81701358e+00 8.96762869e-01 4.70085188e-01
     4.77384796e-01 2.59504613e-01 8.86174704e-01 4.84280743e-01
     9.36384019e+00 2.45777261e+00 1.63740185e+00 7.09051163e+00
     5.65891277e+00 3.62489149e+00 1.86028609e+00 2.38469743e-01
     2.48695156e+00 1.27528229e+00 3.17838322e+00 3.38613890e+00
     3.58912899e+00 3.95905847e-01 5.76440784e+00 3.01265132e-01
     1.86058500e+00 5.41822229e+00 3.55161994e+00 2.08971708e+00
     5.58572875e+00 7.06939902e-01 3.08118539e+00 6.59029732e+00
     1.59193388e+00 4.88109787e-01 1.76601804e+00 1.65439908e+00
     2.97329997e+00 2.36792003e+00 8.49397718e+00 5.50087091e+00
     5.30308604e-01 9.09821456e+00 9.78282299e-01 6.05518238e-01
     3.37109902e-01 4.89584904e+00 8.23488794e+00 4.27987502e-01
     3.28740938e-01 9.57064502e-01 1.23234434e+00 2.54775587e+00
     4.82053739e+00 1.16022867e+00 4.58403438e-01 1.52097034e+00
     2.95593378e-01 2.58009626e-01 5.12479641e-01 6.98073288e+00
     2.32516726e+00 1.16329092e-01 3.37819592e+00 1.48839468e+00
     1.79852324e+00 7.17726477e-01 4.48600350e+00 6.78636722e+00
     2.88940286e+00 2.95728540e-01 6.75596460e-01 2.75231085e+00
     5.89447240e+00 6.43605958e+00 3.36492333e+00 6.46187069e-01
     1.61513038e+00 2.95797034e+00 3.90913435e+00 6.15784697e+00
     6.09222104e+00 4.26922627e-01 2.22940210e+00 4.63240514e+00
     1.16380590e+01 2.09662033e+00 4.84876090e-01 4.39556737e+00
     3.71244446e+00 1.65579001e+00 4.01833557e+00 5.98856505e+00
     1.86770693e+00 1.68236519e-01 1.88967004e+00 2.71016571e+00
     3.55873979e+00 8.73853551e-01 1.19763159e+01 5.72533564e-01
     1.16130165e+00 8.29529789e-01 1.83145917e+00 2.21557344e-01
     7.82125330e-01 9.76807822e-01 1.25565802e+01 2.95429436e+00
     1.05798266e+00 6.45917138e+00 1.23845045e+00 1.67457598e+00
     2.59234347e+00 3.15644392e+00 3.02135147e+00 1.05276305e-01
     9.06383149e-01 1.26172588e+00 4.45146578e-01 9.43473703e+00
     9.15841057e-01 2.09199358e+00 9.78796383e-01 1.61564573e+00
     1.79515154e+00 2.73778507e+00 5.07275560e-01 4.76113350e+00
     4.51715956e+00 1.59562876e+00 2.20711321e+00 3.42470636e+00
     2.57428346e+00 6.73150158e-01 4.65383213e+00 6.82523178e+00
     1.96103901e+00 4.28304676e+00 4.31130411e-01 1.78242169e+00
     2.11926308e+00 2.75280443e+00 6.61051337e+00 1.90994377e+00
     1.03999921e+00 9.17637626e+00 5.40010095e-01 1.25772345e+00
     4.12234912e+00 3.04864889e+00 2.51975007e+00 4.57445317e+00
     2.95235160e+00 1.98644100e-01 3.69710698e+00 4.27955904e+00
     8.03478365e-01 1.81865704e-02 4.01162081e-01 8.37921391e-01
     6.74024964e-01 1.19009614e+00 3.03065234e+00 2.03081877e+00
     7.92530090e+00 2.43011276e+00 4.72232824e-01 2.32732083e+00
     3.13106644e+00 8.93416983e-01 2.45387270e-01 1.92486909e-01
     6.31217274e+00 3.65400956e+00 1.45793725e+00 3.11465792e+00
     7.28354351e-01 8.51129405e-01 1.17748126e+00 2.43701957e+00
     7.94105248e+00 5.80553846e+00 1.94076773e+00 6.71624593e-01
     2.01630969e+00 2.81812744e+00 4.47976801e+00 3.44802742e+00
     7.32249273e-01 2.78308879e+00 1.89853419e+00 4.71609907e-01
     1.59182400e+01 5.23112915e+00 3.40260875e+00 5.93894688e+00
     7.54838729e-01 1.99043272e+00 1.23461783e+01 1.28788373e+00
     2.40235610e+00 4.62013156e+00 4.47402359e+00 4.81147702e-01
     1.24200067e+00 2.71282253e+00 4.99342268e-01 5.52615858e+00
     7.83596198e-01 3.39656989e+00 1.54477261e+00 2.18969538e+00
     5.71586207e+00 1.76333807e+00 5.95884594e+00 4.02781034e+00
     1.06301781e+00 6.24877301e+00 7.88367029e+00 4.94207267e+00
     7.63951471e+00 9.23105517e+00 1.58055692e-01 1.30916686e-01
     9.47466030e-01 2.57613511e+00 3.01678267e-01 1.26898574e-01
     3.70229070e-01 7.73213386e-01 9.45542390e-01 3.00990015e+00
     4.27509194e+00 1.64062384e+00 1.11414728e+00 4.41327218e+00
     6.11850370e+00 2.57047580e+00 6.26047852e-01 7.83723838e+00
     1.62484810e-02 7.17104215e-01 1.31490468e+00 3.12909080e+00
     5.08274975e+00 4.55863586e-01 4.35337039e+00 1.63067326e+00
     9.31727154e+00 1.30449077e+00 3.52483474e+00 6.68453348e-01
     2.99587698e+00 4.18799874e+00 5.74702838e+00 8.74286704e+00
     3.91442389e-01 2.04331707e+00 2.41242509e+00 1.85236668e+00
     2.29050610e+00 9.12294357e+00 8.04257945e+00 1.07791629e+01
     6.30818650e+00 7.89403830e+00 4.42530922e+00 5.50318151e+00
     5.14678794e+00 5.62767097e+00 8.74581357e-01 5.54593925e-01
     3.70291420e+00 7.96666530e+00 7.75245428e-01 6.19723951e+00
     3.02307929e+00 9.37842407e-01 8.65418235e+00 6.73307837e+00
     1.06649428e+00 2.24321027e-01 2.13958798e+00 4.09487571e+00
     4.91115997e+00 3.90733440e+00 1.30649508e+01 1.67578484e+00
     3.03304342e+00 7.54982199e-01 9.10497922e+00 4.68540618e+00]
    θ estimado: 3.1381, número de iteraciones: 9
    [1.76253775e+00 1.00524413e+00 4.69069504e+00 1.91170055e+00
     1.51964966e-01 8.76346615e-01 1.39419155e+00 1.56674204e-01
     5.87663910e-01 2.30288748e-01 2.17505138e-01 1.31821252e+00
     7.68644750e-01 5.76283201e-01 7.82758864e-01 1.37350235e+00
     1.21030744e+01 2.67801786e+00 1.04931797e+00 2.56797664e+00
     1.33158029e+00 3.29009205e+00 6.60679866e+00 1.00532976e+00
     4.09574915e-01 4.55202470e-02 7.80080788e-01 5.47698944e-01
     4.46311549e-01 5.04570730e+00 8.76095012e+00 1.35032757e+00
     9.77361653e+00 1.22399055e+00 1.49494981e-01 2.95616834e+00
     6.82930249e+00 3.05350520e+00 1.17217769e+00 3.28255169e+00
     4.19962333e+00 1.92625150e+00 3.55171482e+00 2.51224488e+00
     4.20590791e-01 2.97330899e+00 9.42314319e+00 2.73891136e+00
     2.61690754e+00 1.79229830e+01 5.89401460e-01 6.77629948e-02
     5.41336568e-01 1.08105264e+00 1.00527902e-01 4.26221118e+00
     1.52642271e+00 1.29797851e+01 3.57240275e+00 1.47103098e+00
     7.48819062e-01 4.91067802e-01 1.30775519e+00 2.46905383e-01
     2.39276893e+00 6.37498450e+00 4.23315938e+00 2.41339266e+00
     1.86127953e+00 8.76363102e-01 2.15912204e+00 8.08161364e-01
     3.04460728e+00 2.65496046e+00 2.72986111e+00 2.60782491e-01
     1.05277857e+00 7.42324482e-01 5.29934663e+00 1.43548981e-01
     2.20490974e-01 1.93727227e+00 6.70166205e+00 1.13886873e+01
     7.16207917e-01 2.18092736e+00 1.62210978e+00 3.71025840e-01
     6.17846611e+00 1.00463666e+00 8.15999684e+00 3.73147692e+00
     3.53600040e+00 2.79940138e-01 4.32357464e+00 5.97614063e-01
     1.90692108e-01 3.98525060e+00 5.76021955e+00 4.32989770e+00
     6.61454029e+00 4.36824142e-01 3.02510225e-02 5.83953909e+00
     1.90162641e-01 1.47422026e+01 2.10274219e+00 3.55556612e+00
     2.36378419e+01 8.10415402e+00 3.85105880e+00 5.17510427e+00
     1.29051090e+00 2.83625305e+00 2.58490029e+00 7.80891099e+00
     1.59465305e+00 1.14338437e+00 1.49795805e+00 4.89272279e-01
     3.83240600e+00 1.07191146e+00 1.67518602e+00 1.19683336e+00
     9.11127660e+00 8.21974794e+00 4.04478943e+00 1.96659172e+00
     1.21747448e+00 6.70740928e+00 1.57158044e+00 2.40359759e+00
     3.69136871e-01 3.27670900e+00 5.42785857e-01 2.69876177e+00
     6.41508122e-01 9.07802855e-02 4.98537904e+00 1.11304047e+00
     8.40546611e+00 6.63309106e+00 1.72133725e+00 1.50038527e+00
     4.14665233e+00 2.12949422e+00 2.93814689e-01 9.42209971e-01
     7.83977653e-01 8.71576304e+00 1.54750509e+00 2.47135045e-01
     8.32549254e-02 7.73189635e+00 8.83497694e+00 1.57677879e+00
     1.29413669e+00 1.75139468e-01 5.88531576e-01 1.85588499e+00
     1.04422439e+00 2.14161239e+00 1.00188134e+00 2.39914404e+00
     8.94270488e+00 5.58760910e+00 7.59154069e+00 8.57777489e-01
     8.29396953e+00 5.35100294e-01 9.65064325e+00 2.70464712e+00
     1.70812380e+00 2.65625260e-01 3.81544120e+00 2.50813784e+00
     5.96868680e-01 2.82997145e-01 4.21712901e+00 1.48996862e+00
     1.24126254e+00 5.46103445e+00 3.14229943e-01 4.41257372e+00
     3.57879805e+00 4.47864175e+00 6.44518987e+00 2.75016179e+00
     1.91576345e+00 1.46608859e+00 5.14200468e-02 2.55673209e+00
     2.37015701e+00 3.46294452e-01 6.11601838e+00 5.61215806e-01
     4.72715530e+00 5.03751579e+00 3.37553735e+00 4.01344695e-01
     2.06789306e+00 4.36187571e+00 1.89155779e+00 1.19471555e+00
     1.33074433e+00 4.06626952e+00 2.52111756e+00 6.51192568e-01
     6.33345853e+00 8.86027450e+00 4.34162132e+00 6.71366618e-01
     1.95274112e+00 3.23581556e+00 1.32429322e+00 4.93356218e+00
     6.21148413e-01 9.09712521e-01 5.74433982e-01 3.24549515e+00
     4.70618184e-01 2.38757448e-01 2.60970262e+00 6.69707092e-02
     7.37178613e+00 2.72767304e+00 1.25839292e+01 5.03862962e+00
     1.08081373e+00 1.96435500e+00 5.23746156e-02 2.49514012e+00
     1.45155409e+00 2.24882170e+00 3.03688914e+00 1.03766592e+01
     7.78048921e-01 6.00318768e+00 2.93143209e+00 1.18336391e+00
     2.19756775e+00 3.52572876e-01 5.57392339e+00 1.00698100e+01
     4.78252838e+00 1.07113051e+00 7.20382064e-01 3.64697329e+00
     1.70826831e-01 4.77980784e+00 2.04720048e+00 1.07270938e+00
     3.53434658e+00 2.12718142e-01 2.61724293e+00 3.20174166e+00
     2.61125352e+00 6.65607055e-01 2.96875578e+00 1.33799753e+00
     3.03206107e-01 4.09528136e+00 3.78638338e+00 8.56952129e-01
     1.67661306e+00 2.50724633e+00 5.77139107e+00 8.76898728e+00
     1.05348606e+01 2.75844111e+00 2.17855225e+00 8.96093377e+00
     2.20533242e+00 6.20121929e+00 7.32718227e+00 2.36674596e-01
     1.20021304e+01 3.37274969e+00 8.97269936e-01 9.48127710e-01
     4.12796629e+00 3.55536159e+00 3.03259663e+00 4.86666852e+00
     1.38694230e+00 8.38703202e-01 1.06041902e-01 2.77600651e+00
     1.91717685e+00 8.22772223e-01 3.01450026e-02 8.53674384e-01
     7.19647347e-01 3.32058762e+00 8.58098481e-01 5.21390663e+00
     1.28056060e-01 1.41515233e+00 2.09204834e+00 1.86408364e-01
     1.15490877e+00 6.68615981e-01 9.69084560e-01 1.77172603e+00
     8.40466008e+00 1.49708819e+00 5.35085659e+00 1.38445474e+00
     4.69123073e+00 8.26339349e-01 5.02138838e-01 8.17843389e-01
     2.28052652e+00 2.46011048e+00 2.96962650e+00 3.63787674e+00
     3.14426583e+00 1.34427404e+00 1.34482870e+00 7.34120818e-01
     6.42452819e-01 4.62544640e+00 9.32922473e+00 5.53104889e+00
     8.91581125e+00 2.85773628e-01 6.76111889e+00 2.92314546e+00
     3.68455330e+00 3.83654016e+00 4.08563739e+00 2.06089899e+00
     4.87851249e+00 5.90932964e+00 7.18181043e-01 4.46765561e-01
     2.19171695e-01 1.26559654e-01 3.64868282e-02 8.12843793e-01
     5.98063973e-01 1.19497550e-01 9.29934821e-03 8.66010633e-01
     1.70546977e+00 6.23000559e-01 4.80842752e+00 6.11439582e+00
     1.11019824e+00 3.50777562e-01 1.76033304e+00 6.83788671e-01
     3.96831823e-01 2.07709567e+00 3.89351307e+00 1.45233335e+00
     6.60205888e+00 1.01685658e+00 2.27558111e+00 3.02110563e+00
     1.58286242e-01 2.14788813e-01 7.23104327e+00 2.58107821e+00
     5.69688407e-02 6.77595786e+00 1.16392505e+00 2.28171447e+00
     1.66235387e+00 7.72401586e-01 5.24009991e+00 7.26426426e+00
     1.22208111e+00 3.13269257e+00 1.19704296e+00 1.84930846e+00
     4.25339252e+00 1.04452032e+00 6.27950278e+00 7.46474959e+00
     5.80304679e+00 5.97009846e-01 7.97126570e+00 2.15396051e+00
     6.84326287e+00 1.00787201e+01 3.30135985e+00 7.22884790e-01
     9.80486339e+00 2.22694245e+00 9.21523087e-01 8.54743379e-01
     2.19732388e+00 8.03571659e-01 1.33199678e+01 2.25050252e+00
     4.91726813e-01 1.57397591e+00 1.98502407e+00 3.98831613e+00]
    θ estimado: 2.9837, número de iteraciones: 5
    [1.53682963e+00 1.48015512e+01 1.40293683e+00 2.95428992e-01
     2.43397070e+00 3.10388711e+00 5.40823111e+00 5.77064246e+00
     3.13856141e+00 3.59769005e-01 4.47887492e+00 2.62669879e+00
     4.16133529e-02 1.07894856e+00 4.86793828e-01 1.56981290e+00
     6.42052260e-01 4.21870401e+00 1.30027165e+00 4.49236852e+00
     7.22446929e-01 1.61842006e+00 2.85484912e+00 2.39340743e+00
     4.30223549e+00 1.00202126e+00 1.58839405e+00 3.28793234e+00
     8.32364530e+00 1.54775381e+00 2.74066660e+00 1.09388393e+00
     3.71601749e+00 1.13353516e+01 2.09359429e+00 2.89489009e+00
     6.79327668e+00 1.96498778e+00 6.69514141e+00 8.78438027e+00
     1.22303383e+01 6.11420700e-01 4.65622484e+00 6.51636197e+00
     1.57057518e+00 1.00141905e+00 4.95017336e+00 1.49509728e-01
     3.85539388e+00 1.48888592e+01 7.57537148e+00 1.69494329e+00
     2.60616121e-01 1.83097792e+00 3.02544328e+00 8.16339122e-02
     2.05241515e-01 2.81330392e+00 2.93460008e+00 1.78089191e+00
     6.29921326e-01 2.17148450e+00 2.55884526e+00 2.16364946e+00
     1.95893557e-01 5.22248037e-01 4.94428485e+00 3.01560856e+00
     1.16169474e+00 1.53299025e-01 2.38062911e-01 8.15918210e+00
     1.17633961e+00 2.57362070e+00 1.11975298e+00 3.74686567e+00
     2.22018097e+00 5.19846215e+00 1.31203923e+00 4.32982250e-01
     2.72648926e-01 1.45521996e+00 5.35695369e+00 2.01047845e+00
     1.59420195e+00 1.50158371e+00 1.16345701e+00 1.87149795e+00
     3.96213357e+00 7.69410229e-01 1.28812927e-01 1.93373338e+00
     2.95206480e+00 5.66639973e-01 3.92564240e+00 3.93333974e+00
     1.54388090e+00 4.45459358e-01 2.90118066e-01 1.14752554e+00
     2.42904655e+00 5.96763028e-02 7.39145610e+00 1.27239821e-01
     6.61968648e-01 5.79462545e+00 9.36481622e-01 1.70093294e+00
     6.57458318e+00 1.18410244e+00 4.18429034e+00 7.35474903e+00
     7.82233854e+00 4.28712677e+00 1.15701361e+00 2.93378865e+00
     2.18855851e+00 3.21037187e+00 1.03764524e+00 3.69024628e+00
     6.00912154e-01 1.11469777e+00 9.03075763e-01 1.29772770e+00
     1.75333847e+00 7.92732401e+00 7.23561953e-01 9.80050870e-01
     4.05562765e+00 1.16449754e+00 2.00367184e+00 3.06435045e+00
     4.92371972e-01 2.91970877e+00 4.44784912e-01 3.16903838e+00
     1.11578843e+00 4.17473403e-01 2.84625920e+00 1.87096752e+00
     3.51374271e+00 7.02525030e+00 1.17895546e+00 3.21488533e-01
     1.03833492e+01 1.63306874e+00 1.38571679e+00 1.85278452e+00
     8.03472387e-01 5.13142971e-01 8.06969840e-01 1.04082314e+00
     8.79382994e+00 2.19798560e+00 7.17129564e+00 4.34697803e+00
     9.62995189e+00 7.83018443e+00 9.60661733e-01 3.88037562e+00
     5.34980595e-01 3.57577920e+00 5.76978075e-01 1.45829843e+00
     1.20618583e+01 8.65429414e+00 1.09155549e+01 1.22789859e+00
     9.81800364e+00 2.70682703e+00 2.41234456e+00 1.01337746e+00
     7.78864223e+00 2.84119162e+00 3.20706818e+00 1.93734876e+00
     2.95775607e+00 5.05461052e+00 5.55585435e-01 1.94494141e+00
     3.21367441e+00 5.45008733e-01 4.33358001e-01 1.39772930e-01
     7.23784067e+00 1.63324194e+00 1.83338588e+00 1.16967881e-02
     1.89949602e+00 2.49740674e-01 5.08571373e+00 9.32294976e-02
     2.98830966e+00 3.26795511e+00 3.47569742e+00 3.11219662e+00
     6.43111925e+00 7.31035578e-01 3.18457187e+00 1.29709426e+00
     9.56739600e-01 6.03991124e-01 8.11409352e+00 1.10707004e+00
     1.82528833e+00 4.93453810e-01 7.15029420e-01 1.45760133e+00
     2.30483938e-01 1.36029818e+00 2.56727440e-01 1.93488754e+00
     3.91195958e+00 3.88871556e-01 9.56399827e-01 1.34722275e+00
     4.53935833e+00 1.94468203e+00 3.45301757e+00 3.97758861e-01
     2.76202581e-01 1.82343949e+00 2.06882634e+00 1.11788319e+01
     3.55003236e+00 4.09123702e+00 3.98678986e-01 1.43538298e+00
     1.05073426e+00 6.38759481e+00 3.62391927e-01 3.46112949e+00
     1.92449411e+00 4.12214772e+00 1.10552789e+01 1.91572075e+00
     5.00516003e+00 1.13106719e+00 1.45677255e+00 7.08296359e-01
     1.47233220e+00 5.64787628e-01 3.86601314e+00 4.86427722e+00
     1.65990862e+00 6.83430490e-01 1.89504970e-02 8.27827656e+00
     1.67401855e+00 1.67107122e+00 3.62942028e-01 1.37636885e+00
     6.04695052e-01 3.81401435e-01 5.69082432e-01 6.44276922e+00
     2.23033887e+00 1.07534270e+00 2.38791446e+00 6.98068026e+00
     1.62581440e+00 2.93171285e+00 2.97715100e+00 7.70179453e-01
     3.38640353e-01 4.65734739e+00 5.46255597e+00 4.50285931e+00
     1.32054687e+00 8.58460342e-01 1.31874185e-02 9.54460050e-01
     1.35941429e+01 2.52575959e+00 6.40715543e+00 8.18677605e-02
     1.74779403e+00 4.87322666e+00 3.93277959e+00 1.56648374e+00
     4.95174414e+00 3.35457788e+00 6.87097833e-02 8.96276110e-01
     1.82775431e+00 5.28367008e+00 3.25346515e-01 5.34172259e-01
     3.34220593e+00 5.21043648e+00 4.11294990e-01 6.34384120e+00
     9.69808776e-01 2.02350989e+00 1.83095841e-01 1.70738794e+00
     6.88572143e+00 1.62783575e+00 8.86013929e-01 2.01162464e+00
     3.03901120e+00 4.38402320e+00 2.25744066e+00 5.27986411e+00
     3.16286119e+00 1.45413449e+00 1.03238246e+00 4.64045746e+00
     4.04220235e+00 4.84553553e+00 7.28610024e+00 6.68920080e+00
     1.21106062e+00 5.92722159e+00 6.65395879e-01 1.93526540e+00
     1.10549925e+00 1.14883448e+00 1.10844661e+00 1.86487250e+00
     1.62387910e-02 2.51184403e+00 3.71816669e+00 1.28500721e+00
     7.12923855e+00 3.92942660e+00 6.16020616e-01 7.28459414e-02
     3.69258961e+00 7.23872192e-01 9.19508880e+00 5.53343230e-01
     3.43435105e-01 3.95008059e+00 3.27717633e+00 9.80886038e+00
     1.10309388e+00 1.56598480e+00 2.32764973e+00 2.66618390e-01
     4.84458412e+00 3.39406025e+00 1.61542191e-01 3.31876389e+00
     9.04726954e-01 2.30802442e-01 1.11980400e+00 5.53767346e-01
     2.38465761e+00 1.01758143e+00 5.30306364e+00 2.81073575e+00
     3.25918199e+00 5.98800607e+00 4.73466325e+00 3.44837314e+00
     4.08096494e+00 6.65098473e+00 5.18537778e-01 1.94917550e+00
     6.88351619e-01 5.65169025e+00 2.51280129e+00 1.43611095e+00
     1.17692274e+00 1.15388758e+00 7.59590094e+00 1.15265299e+01
     5.56260986e+00 5.32577277e+00 4.44597159e+00 5.03262090e-02
     5.10102762e-01 1.30306965e+01 3.86903958e+00 2.05505873e-01
     4.90771271e+00 4.31314795e-01 1.37128363e+01 1.28976645e-01
     1.43329877e+00 3.27120376e+00 4.87171898e+00 2.12211705e+00
     4.40290128e+00 1.25510219e+00 1.56885323e+00 5.81443280e+00
     2.06138942e+00 2.34534861e+00 4.56882269e+00 1.03090228e+01
     1.63094636e+00 7.32398486e+00 5.99636691e+00 1.27769990e+00
     2.43057238e+00 2.94851749e+00 6.29507646e-01 5.64432292e+00]
    θ estimado: 2.9582, número de iteraciones: 1
    [3.57213154e-01 3.11731786e+00 3.02219261e+00 7.16897009e+00
     5.27510340e+00 3.20825603e+00 2.54707653e+00 3.22149267e-01
     1.89142367e+00 8.88249270e-02 2.44498731e+00 5.85458742e-01
     6.37753160e+00 7.51836310e+00 1.27259163e-01 1.01470404e+00
     2.97375172e+00 1.11104238e+01 5.39675424e+00 5.68448106e+00
     7.62422092e-05 3.08778976e+00 7.15958451e-01 2.97579636e+00
     1.23385878e+01 2.03254837e+00 3.36002681e+00 1.43938517e+00
     1.44541149e+00 4.85167486e-01 2.56141492e-01 1.47376363e+00
     1.64270313e+00 4.09375221e+00 2.43131577e+00 6.37303148e-01
     4.59423399e+00 3.22227277e+00 3.44044155e+00 6.51971687e-02
     3.20647246e+00 1.38463002e+00 2.45758658e+00 1.20508490e-01
     7.33424097e-01 6.97624027e-01 1.07204680e+00 7.63364608e+00
     1.31781142e+00 4.80899265e+00 2.82911311e-01 2.25685657e+00
     3.19629329e+00 1.99740955e+00 2.30493954e+00 2.56891091e+00
     1.52988473e+00 3.89510048e+00 3.72695980e+00 1.41457775e+00
     3.49002913e+00 1.68606679e+00 9.15048679e+00 1.14987078e+01
     2.73485909e+00 2.03023301e+00 2.82712978e+00 2.24302397e+00
     6.46422376e+00 1.02801682e+01 2.57264880e+00 1.35056611e+00
     5.66994811e+00 5.99489744e-01 3.23004658e+00 3.14444045e+00
     2.30108681e+00 5.40159104e-01 4.58676181e+00 1.38858096e+00
     7.58486903e-01 1.08693073e+01 1.82106406e+00 6.61860877e+00
     5.06070856e+00 3.86650511e+00 7.88152204e-01 4.69739775e+00
     7.62006583e-01 2.17736663e+00 2.85849665e+00 2.48946434e+00
     3.89642119e-01 1.03523935e+00 6.79807132e+00 4.27124139e-01
     1.05220625e+01 7.85846390e+00 1.06308671e+01 4.49808612e+00
     1.40751466e+00 8.83837323e-02 3.07967619e+00 1.63144270e+00
     6.78164075e+00 2.91710342e+00 7.93331046e+00 1.43688125e+00
     2.36982449e+00 1.16326225e+00 7.82791947e+00 3.00774726e+00
     9.56840893e+00 2.07619707e-01 5.24574719e-01 9.73976644e-01
     1.27522035e+00 1.52406704e+00 3.52966139e-01 1.05299088e+00
     2.24999092e+00 8.18841576e+00 4.50727386e+00 2.13834762e+00
     6.92800871e+00 6.21608979e+00 1.21652657e+00 6.59438167e+00
     6.48053917e+00 3.41067159e-01 3.78754987e+00 1.63032735e+00
     6.89537353e+00 1.02085458e+00 1.42773949e+00 3.20581053e+00
     3.91081827e+00 2.09888459e-01 9.49210300e+00 8.96403630e+00
     9.29647645e+00 1.79587933e+00 4.84829682e+00 3.01854766e+00
     7.75050039e+00 4.31889652e+00 4.51809614e-01 1.38741530e+00
     1.19367280e-01 3.35667266e-01 2.10325481e+00 1.66432329e+00
     4.46970432e+00 1.91313278e+00 1.16181443e+01 5.57317977e+00
     3.83117158e+00 5.38352895e+00 2.22627486e+00 9.44424970e-01
     2.65441197e+00 5.07595367e+00 2.19842033e+00 2.46925949e+00
     1.13548645e+00 1.65741651e+00 1.63310366e+00 2.52694698e+00
     3.61036714e+00 2.25983449e+00 1.20861719e+00 1.40490531e+00
     6.54869771e-01 9.92029417e+00 9.79346285e-01 3.64165258e-01
     1.19741655e+00 1.35677691e+00 4.62963243e+00 1.21581144e+01
     1.31907747e+00 1.92609046e+00 1.81422121e+00 8.70177995e+00
     2.49941057e+00 3.36871666e+00 6.33692708e+00 4.46027432e+00
     4.91809274e+00 2.38730450e-01 1.38343932e+01 7.28719718e+00
     3.06603282e-01 5.26346383e+00 3.08382717e+00 1.80006975e+00
     4.22453832e-01 7.06436948e+00 1.03226413e+00 4.53832677e+00
     3.89258584e+00 7.49594663e+00 5.33564815e+00 9.07537751e+00
     3.32716711e+00 3.63232087e+00 1.73315429e+00 3.37878934e+00
     2.11735249e+00 1.08732519e+00 4.21999394e-02 5.53440885e+00
     5.88093636e+00 8.79794401e-01 3.06475483e-01 2.61755268e+00
     7.40272130e-01 2.76319718e+00 8.67218238e+00 3.74125755e+00
     2.15824507e+00 1.55020380e+00 3.26361748e-02 2.54933902e+00
     1.77103861e+00 5.64902036e+00 3.48715503e+00 2.32953156e+00
     1.77290210e+00 4.72524408e-01 9.46094036e-01 1.54174670e+00
     4.33081645e+00 1.09919307e+00 6.75066447e+00 1.52127009e+00
     5.42101061e+00 6.81932134e-02 6.89476276e+00 3.71480787e+00
     2.83103762e+00 1.93924026e+00 3.14501668e+00 2.16644532e+00
     1.60164156e+00 2.94277831e+00 6.34357410e+00 2.27065126e+00
     1.40960961e+00 5.04557433e-01 1.01794188e+00 3.49437528e+00
     3.23437376e+00 3.68904171e-01 2.00299809e+00 2.74647738e+00
     2.07722460e-01 8.07357226e+00 3.17263514e+00 6.08566952e-01
     2.63086427e-01 2.15157617e+00 2.09797427e+00 5.69190800e+00
     3.47359392e+00 1.53580443e+00 1.20438454e+01 1.50748467e+00
     5.47357419e+00 6.26989952e+00 2.40014183e+00 6.37916863e+00
     1.53023366e+00 1.16497394e+00 6.97875719e-01 1.79645482e+00
     8.48415629e+00 3.72103701e+00 9.35840948e-01 4.79569595e+00
     1.31749172e+00 2.32405461e-01 3.29639743e-01 1.54884057e-02
     3.11230122e+00 1.86116277e-01 1.67461366e+00 9.12638467e+00
     6.10090972e+00 1.04322315e+00 1.42345027e+00 9.00476998e-01
     2.64431319e+00 6.04931003e-01 1.06327638e+00 3.55441160e+00
     9.87435616e-01 2.98300096e+00 5.68075388e+00 8.87496994e-01
     1.02717602e+00 1.20223833e-01 2.83393564e-02 1.03915202e+00
     1.45128147e+00 1.38715346e+00 1.67980749e+00 1.98854482e+00
     1.95027754e+00 2.98123949e+00 1.91529730e+00 8.48685647e-01
     5.50939327e+00 2.34333467e+00 3.02208459e+00 4.53969715e+00
     3.66026423e+00 1.20106161e+01 1.07245844e+00 4.04713864e-01
     5.04269285e-01 6.05859491e+00 1.78242634e-01 7.24210533e+00
     1.58735709e+01 1.42607932e+00 1.20470868e+01 1.86400346e+00
     9.56099350e+00 4.77980252e+00 9.52914970e-01 4.19132852e+00
     2.30276504e+00 1.68904067e-01 3.38857373e+00 2.03729084e-01
     2.05235984e+00 5.62017630e+00 3.10172773e+00 2.19272705e+00
     5.10863016e-01 5.69165892e+00 2.99926654e-01 1.19647402e-02
     4.84813706e+00 2.72511777e+00 7.03702652e+00 4.17860063e-01
     6.39575416e-01 3.32555196e+00 3.93452634e+00 6.27318309e-01
     2.66612142e+00 1.34900567e+00 1.39774314e+00 4.39663243e+00
     1.15702462e-01 1.61326190e+00 1.45214490e+00 1.80629102e+00
     4.05531255e+00 4.72604059e+00 2.07136722e+00 1.19697138e+00
     6.48977834e+00 5.22802892e-01 1.47395471e+00 5.64760168e-02
     4.28177276e+00 2.13498757e-01 3.03182366e-02 1.83291681e+00
     3.33676031e+00 2.67280925e+00 7.56480456e+00 8.98097810e-01
     9.10339437e-01 5.73398177e-01 1.17758833e+00 4.53262420e+00
     9.30164697e-01 2.03955978e+00 3.72460830e+00 1.46126876e+00
     9.66767994e-01 7.45657265e-01 1.48719896e+00 6.32476701e-01
     1.24620262e+00 1.30840537e+00 1.80079493e+00 2.18905592e+00
     2.87471548e+00 2.48688499e+00 1.50209302e+00 2.32765123e-01
     3.35364802e+00 1.91759004e+00 7.29809992e+00 7.18624877e-01]
    θ estimado: 3.0999, número de iteraciones: 1
    [2.11064600e+00 8.42270289e-01 1.72053594e+00 2.27728072e+00
     8.36842338e+00 4.53327156e+00 1.12990723e+00 1.70888915e+00
     4.61200819e+00 4.08523709e+00 3.60426101e+00 1.45074889e+00
     6.06257600e+00 1.62083343e-01 3.50698134e+00 5.18564640e+00
     7.53730360e-01 9.23424941e+00 2.53298725e+00 3.59297549e+00
     8.25215530e-01 6.17710866e-01 2.64203546e+00 1.88514316e+00
     6.68027827e+00 7.80169031e-01 7.75047805e+00 1.14217749e+01
     4.51077273e+00 2.42122416e+00 3.80890248e+00 1.24825185e+01
     2.38473718e+00 1.34333134e+00 2.67590086e+00 4.75418816e-01
     3.16277085e+00 2.39438817e+00 1.23533559e-02 2.93771788e+00
     9.06808435e-01 4.97901879e+00 1.00886364e+00 6.34031573e+00
     7.82136982e-01 6.47702106e+00 4.38438424e+00 1.59838791e+00
     1.27125814e+01 2.25142636e-01 6.68038952e-01 1.43671722e+00
     1.33989418e+00 1.41812526e+00 1.89721260e+00 1.59798004e+00
     3.67685293e+00 4.84518884e+00 5.19365814e-01 7.21561717e-01
     4.11502783e+00 5.88528167e+00 4.12339417e+00 1.63116736e+00
     1.32519559e-01 2.09686746e+00 6.24002421e+00 2.98829406e+00
     7.60845896e+00 5.46855532e+00 4.80408180e+00 1.41315790e+00
     3.10170646e+00 8.55470242e-01 3.88248227e+00 9.98272571e+00
     3.98083910e+00 4.33366951e-01 3.22267751e+00 9.57996370e-01
     2.25776324e+00 1.09388179e+01 3.85680973e+00 1.37580952e+00
     2.74295120e+00 3.33838462e+00 3.91499233e+00 1.50586804e-01
     2.58567764e+00 8.34905568e+00 2.38645740e+00 5.86016833e+00
     1.95275100e+00 5.79434665e+00 2.45138272e+00 3.00673814e+00
     6.69523036e-01 1.13018543e+00 9.40794485e+00 5.75474616e+00
     1.94174885e+00 6.41676431e+00 6.87972182e+00 4.49054178e-01
     1.77373854e+00 6.94862917e+00 1.89931213e+00 3.01603621e+00
     5.45443271e+00 1.20001354e+01 9.89411927e+00 2.90305833e-01
     3.73783624e+00 2.12411398e+00 6.56720975e+00 9.47456667e-01
     2.31920982e+01 2.83519873e+00 3.55438749e+00 8.99362649e-03
     2.58464345e+00 8.51947054e-01 3.94181208e+00 2.46122280e+00
     4.29558792e-01 1.99473449e+00 1.65817429e+00 1.58219340e+00
     3.17945867e-01 7.76725479e-02 3.25927377e+00 6.09100545e+00
     1.24314054e+00 5.54432752e+00 1.52512264e+00 4.90675400e+00
     3.27413078e+00 9.73655558e+00 2.07672768e+00 3.84140271e-01
     7.58047048e-01 2.78280876e-01 3.84256635e-01 4.77076208e-01
     3.03835983e+00 4.29016711e-01 3.56770381e+00 4.78913599e-01
     6.25305298e+00 3.22799690e-01 4.81478872e+00 3.94215682e-01
     4.70827109e+00 6.41236605e+00 1.59517663e+00 2.36699730e+00
     7.45259951e-01 1.36123571e+00 7.05707628e-01 2.17339059e+00
     1.24333930e+01 1.27112356e+00 2.07841852e+00 8.80959780e+00
     2.20046073e+00 3.07712627e+00 5.54974091e-01 1.76893117e+00
     3.23766005e+00 3.03036129e+00 4.07227996e+00 7.18347854e+00
     6.94535253e+00 2.42152017e+00 8.73512432e-01 7.39851602e+00
     8.99507493e-01 4.57319340e-01 9.03659861e-01 1.54678491e+00
     2.95810556e+00 1.21258897e+00 2.47417192e+00 2.77743191e-01
     2.04686765e+00 8.84431305e+00 2.11179072e+00 8.99650180e+00
     1.17396631e+01 2.47249879e+00 3.62284384e-01 8.79881407e+00
     7.93388674e+00 1.50294793e+01 1.77941739e+00 3.69862535e+00
     1.90252341e+00 3.40525951e-01 6.17492192e-01 1.39789939e+00
     2.71228328e+00 2.04330854e+00 6.73072629e-01 5.31611464e+00
     2.39343483e+00 2.47035299e+00 1.53327274e+00 2.53733498e+00
     3.57454397e-01 4.56324855e-01 1.70067276e+00 7.67746631e-01
     4.63992587e-01 5.43513520e-01 3.97642145e-01 7.75412703e-01
     1.22091888e+01 2.00982043e+00 5.17942720e+00 4.24193974e+00
     2.31536448e+00 4.24260396e+00 7.99314808e+00 5.23137444e+00
     1.80604764e+00 3.14637167e+00 3.97765549e+00 1.22770106e+00
     4.06583871e+00 2.13691044e+00 3.03286242e+00 2.36213000e+00
     1.03930958e+00 2.77224573e-01 3.79596741e+00 2.33046863e+00
     7.00462476e-01 1.68169571e+00 1.89506385e-03 5.43006856e-01
     1.26270148e+00 1.24482370e+00 2.87534497e+00 2.94008620e-01
     3.92086184e-01 6.40866730e+00 1.78524688e+00 8.85511237e-02
     2.08603479e+00 1.52426594e+01 1.05580841e+01 5.16313775e+00
     1.67792845e+00 1.59439259e+00 5.56502400e+00 3.37327679e-01
     3.09082309e+00 1.19257100e+01 9.77522820e+00 1.17491249e+00
     5.94384958e+00 1.99822140e+00 3.36496650e+00 1.96235547e+00
     1.08214280e+00 1.69915678e-01 6.99977188e-01 3.15669628e+00
     1.13359271e+00 5.76224155e+00 4.19766693e-01 3.70196780e+00
     1.90973702e+00 6.07962428e-01 2.14023330e-01 5.47747751e-01
     1.65738075e+00 6.48932578e+00 2.56662964e+00 4.84946632e-01
     4.00732875e+00 1.08829475e+01 1.26058467e+00 1.84120958e-01
     1.25683146e-01 1.71036248e-01 1.54230154e+00 2.97770471e+00
     2.12661670e-01 5.02362598e+00 1.24612003e+00 5.47091905e+00
     1.93502535e+00 2.13120532e-01 2.50864823e+00 5.09692602e+00
     5.54605688e+00 4.35945758e+00 1.24454011e-01 7.69061431e+00
     1.23766956e+00 3.01090429e+00 1.21410062e+00 2.08416469e+00
     4.88495449e+00 3.03836862e+00 2.99888515e+00 8.55070640e-01
     4.05030456e+00 4.99009194e-01 2.19940610e-01 3.82555714e+00
     1.44591037e+00 3.79755741e+00 9.51841593e-01 2.04812908e-01
     1.66390997e+00 2.31438321e+00 1.48072001e+00 2.62282435e+00
     2.61477858e+00 2.40711638e+00 2.04319662e+00 4.92030770e-01
     2.67901456e+00 2.55180204e+00 1.69681873e+00 7.42036604e+00
     1.33729673e-01 2.54405738e+00 4.87671634e-01 1.20889142e+01
     3.82888092e+00 4.32687814e+00 4.09655456e-01 4.94250314e+00
     2.59511096e-01 1.08018024e+00 3.64126347e+00 1.57238888e+00
     2.69715747e+00 3.77196489e+00 3.41578207e-01 1.04467067e+00
     1.50070093e+00 1.02985710e+00 2.62155291e+00 1.89218336e+00
     2.91613064e+00 2.26073054e+00 1.64014979e+00 5.24473512e-01
     3.63348780e+00 1.84467376e+00 1.21839056e+01 4.89566944e+00
     7.19058846e+00 5.27791110e-02 3.70953302e+00 1.04196591e-01
     2.00099955e+00 3.01587977e+00 2.33090253e+00 9.69834716e+00
     1.06361544e+00 3.02056498e+00 1.68191191e+00 9.12352458e+00
     1.15137154e+01 3.60999893e+00 2.20884365e+00 8.67450481e-01
     1.28175885e+01 5.06667368e-01 8.25169132e-01 3.22747281e+00
     6.05687123e-01 2.36945035e+00 4.92337465e-01 2.65588148e-01
     9.20715113e-01 1.12164474e+00 6.70792659e+00 1.10453583e-01
     3.47141540e+00 1.39764896e+00 2.54741000e+00 1.28090082e+00
     1.72613258e+00 7.76913785e-01 5.81362448e-01 5.71747775e+00
     2.13533529e+00 1.03098909e+00 2.31651340e+00 3.15008551e-01
     1.99390909e+00 9.10801649e+00 2.08976212e-01 2.99980980e+00]
    θ estimado: 3.1481, número de iteraciones: 1
    [2.45712490e+00 2.14463111e+00 6.23411205e+00 3.59734702e+00
     1.18961922e-02 4.90909871e-01 5.21348291e+00 1.86173043e+00
     1.02917587e+00 3.37168350e-01 3.00706193e-02 1.00897048e+01
     1.65949452e+00 4.05993563e+00 1.37272104e+00 3.35713377e+00
     2.36473695e+00 1.89099149e+00 5.09343566e+00 1.19732042e+00
     2.59398239e+00 1.47272274e+00 2.76316630e+00 1.64143700e+00
     3.24774048e+00 6.63193340e-01 6.77072174e-01 1.77017532e+00
     1.32703021e+00 6.05390148e+00 6.95328108e+00 1.41474005e+01
     3.57412875e+00 1.77300941e+00 1.59375728e+00 3.05161196e-01
     3.72614481e+00 4.26510714e+00 1.20174603e+00 3.10268064e-01
     8.36483699e-01 6.86569715e+00 6.68408455e+00 2.59323710e-01
     8.86721972e-02 2.98136596e+00 4.00493137e+00 3.42756753e+00
     8.17393631e+00 3.56542853e+00 3.21803904e+00 7.34570039e+00
     1.94504909e+00 7.03367989e+00 1.02688298e+01 2.99446399e+00
     2.32601436e+00 5.05676046e+00 1.45512358e+00 1.55000850e+00
     3.68712326e+00 6.02662188e+00 1.32864566e-01 4.05958805e+00
     1.08632852e+00 6.61081842e+00 8.15435974e-01 1.11795037e+01
     1.73966880e+00 5.15286442e-01 2.79370842e+00 3.30984952e+00
     1.62968103e+00 2.47465744e-01 7.94769859e-01 3.97888492e+00
     6.95888239e+00 4.54867380e-01 1.70122360e+00 4.26383835e+00
     9.69145477e+00 3.17478279e+00 3.49682998e+00 2.38349824e+00
     2.29355576e+00 3.28020580e-02 2.26840849e+00 4.14299695e+00
     5.92978712e+00 1.39145616e+00 2.62430119e+00 2.76970428e+00
     6.07102069e+00 1.58046205e+00 9.00636703e-01 2.97624621e+00
     2.33546488e+00 2.90380701e+00 8.36572776e+00 2.59619582e+00
     2.27755256e-01 7.07402222e-01 8.51637906e+00 1.14452334e+00
     5.31388907e+00 2.02018724e+00 2.31751326e+00 1.10022076e+01
     4.32855223e+00 3.83969473e+00 2.60204182e+00 2.67771094e-01
     4.20355541e+00 4.05187749e-01 9.05364315e-01 2.05189208e+00
     7.97600617e-01 4.64683672e+00 1.27931158e+00 7.81898520e-01
     1.20732966e+00 3.69097365e+00 2.61809429e+00 3.65890522e-01
     6.44594869e+00 6.99109719e+00 2.53299554e+00 6.58490742e+00
     1.20185589e+01 3.70392379e-01 2.56567845e+00 1.37654825e+00
     8.55670655e+00 2.53297224e+00 2.11149987e+00 4.33999401e+00
     3.98065521e-01 4.16081981e-01 2.76363392e+00 2.55748897e+00
     2.93874578e+00 2.63755077e+00 1.79974875e-01 1.14974228e+00
     2.09857180e+00 2.43502661e-01 1.01839200e+00 1.16882507e+00
     5.95021949e-01 2.24676879e+00 7.48083741e+00 1.95865557e+00
     3.88181268e-01 2.99868681e+00 1.73476600e-01 3.38475358e+00
     4.05432003e-01 1.38671731e+00 1.31274460e+00 8.88709729e-01
     3.69785566e+00 3.05639214e+00 4.63314814e-01 1.99241899e+00
     4.45926572e+00 2.88581182e+00 3.86546564e+00 1.07586073e+00
     1.31611593e+00 2.57612714e+00 3.33943244e-01 1.37094064e+00
     9.10072662e+00 1.16879428e+00 8.20617999e-01 4.26989364e+00
     1.74346309e-01 4.61803307e-01 1.01616010e+00 4.14461110e+00
     2.21166516e+00 6.19560783e-01 1.60890919e-01 6.35295572e+00
     4.07938976e+00 2.80098730e-01 5.15296292e+00 8.92919766e+00
     2.42873523e+00 1.80825836e+00 5.08242260e-01 1.32535865e+00
     3.40171177e+00 7.66517848e-03 2.55061075e+00 5.31618638e+00
     7.44561785e+00 1.47048733e+00 1.01062754e+00 1.91698219e+00
     1.10708480e+00 7.46760069e+00 6.92380850e-01 1.14489736e+00
     8.94523456e-01 3.40142739e-01 1.24821296e+00 5.73913094e+00
     1.73545307e+00 2.16137884e+00 4.59223934e+00 1.70381643e+00
     5.15145971e+00 1.45054647e+00 7.47678879e+00 1.45121154e+00
     2.65779881e-03 2.52267714e+00 5.29617515e+00 1.88943818e+00
     3.11036563e+00 2.34390472e+00 1.21381189e+00 3.28709763e+00
     1.30339814e+00 3.17894772e-01 4.52845861e-01 1.22765384e+01
     3.31079249e+00 6.10160395e-01 1.46990402e+00 1.96710160e+00
     1.36227318e+00 1.24444493e+01 1.01058762e+01 1.49460916e+01
     8.83655520e+00 1.24695344e+00 5.27766055e+00 7.85518949e+00
     2.68058731e+00 4.70316901e-01 2.67798636e+00 1.44489213e+00
     1.98901623e+00 2.43220374e-01 7.87800890e-01 3.06897101e+00
     1.71876556e+00 6.88918039e+00 3.24184938e+00 3.00484711e+00
     5.18438848e+00 2.76985992e+00 6.95642820e+00 1.26383843e-01
     3.90108030e-01 2.13565973e+00 7.43258013e+00 8.46375336e+00
     3.62157120e+00 2.05922663e+00 5.84587008e+00 3.33401622e-01
     2.39457859e+00 3.86005712e+00 4.22820859e-01 9.48705057e-01
     1.28046175e+01 9.09254841e+00 2.99943342e-01 6.13159615e+00
     1.06180052e+00 3.27364077e-01 5.94782625e-01 5.24094453e+00
     1.46136887e+00 7.89325605e+00 4.99601570e+00 1.73208082e-01
     1.08217704e+01 2.63735942e+00 1.11599444e+00 9.33535461e+00
     4.90579383e+00 2.29227454e+00 8.64465264e-01 1.54398432e-02
     5.94670986e-01 6.19475720e+00 6.03519383e-01 1.07824662e+01
     1.22141133e+00 6.91627414e+00 5.87844110e-01 4.71219524e+00
     5.98767993e+00 3.95727031e-01 1.27724641e-01 1.22426730e+00
     3.05641236e-02 1.22335033e+00 2.39854138e+00 2.99765564e+00
     3.19061694e-02 9.76756147e-01 3.44294832e+00 8.37339618e-01
     2.75495961e+00 6.68594272e-02 1.78957691e+00 3.24025831e-01
     3.16644639e-01 3.63734443e+00 1.63390733e+00 2.96675912e+00
     5.18228055e+00 4.83863901e+00 2.45571736e+00 4.38566079e+00
     5.12556871e+00 1.35198559e-01 3.40217376e+00 1.67725518e+00
     1.64868095e+00 1.12209534e+00 4.31418060e+00 7.26664821e-02
     2.91170007e+00 9.22147039e-01 5.68769467e-01 1.95367242e+00
     7.94450896e+00 5.03075635e+00 1.13176464e-01 1.04085325e+01
     4.39958750e-01 3.44727881e+00 4.49337732e-01 4.79284034e+00
     8.25187985e+00 1.50552890e+00 1.34169335e-01 3.44769237e+00
     5.96604770e-01 4.00056537e-02 8.25557297e-01 1.95225211e+00
     2.97448530e+00 3.38924738e+00 1.45833440e+00 9.04323268e+00
     6.11619905e+00 5.75678312e-01 2.40149781e+00 4.67930337e+00
     6.32518475e+00 5.56545948e+00 1.85170184e+00 4.37885773e+00
     1.27217050e+00 5.86165026e+00 1.24953825e+00 7.32721492e+00
     1.80965693e-01 3.12546395e+00 7.37763791e+00 1.08967881e+00
     7.00412518e-01 1.73471427e+00 3.46448635e+00 1.68625250e+00
     5.30077417e-02 3.83044070e+00 1.97301337e+00 1.41383624e+00
     2.34844928e+00 6.64845078e+00 8.10413946e+00 3.80328805e+00
     1.88114043e+00 2.74820940e+00 2.13466495e+00 3.83685786e-01
     1.60440253e+00 5.77563649e+00 4.76124186e-01 3.24320454e+00
     8.10385194e-01 3.46241549e+00 2.75319022e-01 6.37703602e-01
     7.26519089e-01 2.99070741e+00 1.23798453e+00 1.46714322e+00
     9.26486721e-01 1.41816989e+00 1.31723970e-01 3.43765670e+00]
    θ estimado: 3.0148, número de iteraciones: 1
    [1.85312183e+01 1.47084813e+00 2.93833649e+00 4.36917076e+00
     1.04372567e+01 3.32741489e+00 3.73398467e+00 2.00267487e+00
     7.51498394e-01 1.02680762e+00 9.74842313e-01 1.70768220e+00
     2.46434841e-01 1.84059981e+00 6.31999208e-01 2.36951139e-01
     1.29337103e+00 3.43367229e-01 6.28943769e+00 4.14566508e+00
     6.67938747e+00 3.23193691e+00 2.85486249e-01 6.53801252e-01
     9.05227650e-01 2.14679301e+00 1.87699340e+00 3.79272577e+00
     6.38465365e-01 8.28365240e-01 6.41402029e+00 1.94442794e+00
     1.07586700e+00 3.23108484e-01 4.45366501e-01 1.44601196e+00
     4.04670875e-01 1.99719369e+00 1.73215103e+00 3.31271623e+00
     1.04910459e+00 4.46676736e-01 1.00540513e+01 8.43367129e+00
     3.60306002e+00 1.49602422e+00 3.83585015e+00 6.57710027e-01
     1.32190346e+00 3.81690453e+00 9.78587270e+00 2.34325099e+00
     1.91613781e+00 1.24586475e+00 6.61429270e+00 8.97646047e+00
     1.01377658e+01 3.08928562e+00 1.38341010e+00 2.76415627e+00
     5.82635745e+00 4.70217882e-01 9.15408222e-01 4.80062687e+00
     1.03643920e+00 2.36788264e-01 1.28346972e+00 5.09654296e-01
     2.57748373e+00 4.66116620e-01 2.88884012e+00 9.26123857e-02
     6.33438085e-01 8.36169929e+00 1.98822374e+00 1.63261700e+00
     4.56966529e-01 5.05644291e+00 7.19577612e-01 2.36242389e+00
     2.10506854e+00 2.59818814e-02 1.20704430e+01 2.18717217e+00
     2.34344661e+00 1.27827803e+00 2.82080602e+00 6.15143942e-02
     1.17026325e+00 2.26385175e+00 1.46659055e+01 5.10200373e-01
     9.95245121e+00 5.88899686e-01 1.34776073e+00 4.80055150e+00
     1.03012231e+00 3.86319626e+00 2.15602353e+00 1.06838258e+00
     2.04362782e-02 5.83782836e+00 3.26466324e+00 3.22033026e+00
     1.32477928e+00 7.93053310e-01 2.39307952e-01 5.90864342e+00
     7.18273761e+00 3.32065568e+00 2.10648437e+00 2.62552777e+00
     4.67536970e-01 9.29734185e+00 1.03399311e+00 2.78462474e+00
     5.00986818e+00 3.11023214e+00 4.02513110e-01 6.21764750e+00
     8.44184401e-01 1.21004502e+00 2.70489196e+00 7.18214442e-01
     5.77898488e+00 5.03420090e+00 2.01278131e+00 4.86382528e+00
     3.96443900e+00 1.91010411e+00 2.54191693e+00 5.46011704e-01
     6.32043137e+00 2.03913548e+00 5.00335117e-01 1.52034604e+00
     3.57954338e+00 3.79900507e-01 2.18174786e+00 1.52285229e+00
     6.89807361e-01 1.56776065e+00 6.84932179e+00 1.14675064e+01
     1.81714491e+00 2.72317762e-02 6.85567016e+00 9.42794334e-01
     5.13339885e-01 1.58829259e+00 2.10942869e-01 3.50944880e+00
     7.88728789e-01 2.58426292e-01 3.45791567e+00 2.90646210e+00
     1.28218878e+00 6.47871540e-01 5.92496189e-01 2.85202162e-01
     1.36799933e+00 2.12818575e+00 1.67188725e+00 2.14170822e+00
     3.67063115e+00 5.15880694e-02 2.39597288e+00 4.68509952e-01
     2.82324213e+00 5.01080597e+00 7.71920058e-02 7.24850244e+00
     2.10021801e+00 8.66167511e-01 1.41925826e+00 2.62002103e+00
     8.97407559e-02 4.10092147e+00 2.68214526e+00 3.98604893e+00
     8.46312588e+00 3.36880266e+00 4.15222807e+00 1.33568476e-01
     4.12917682e-01 2.96643252e+00 2.45318820e-01 2.36765145e-01
     3.95551811e+00 8.86964869e-01 2.44759129e+00 1.26674003e+00
     5.23736958e+00 1.11257767e+00 2.92899808e+00 7.48369979e-01
     2.00875822e-01 5.19486076e+00 2.07452950e-01 5.54590137e-01
     1.36310254e+00 3.78094109e+00 1.02872955e+01 5.64891861e-02
     1.18427302e+00 9.14864307e+00 4.47273417e-01 4.22300313e+00
     7.94669643e-01 2.28588603e+00 9.78782416e-01 5.83356506e-01
     1.40193175e-01 1.27208442e+00 1.99701099e-01 4.08175576e+00
     2.30786835e+00 1.67300121e+01 1.77440308e+00 1.57948297e+00
     1.81817603e+00 8.44881005e-01 2.22869283e-01 2.81523071e+00
     4.07528281e+00 7.61018522e+00 8.99609841e-01 9.16270528e+00
     8.53289884e-01 1.91646103e+00 4.21990878e+00 3.53755696e-01
     7.84402333e-01 5.98915938e-01 9.76732106e-01 1.03402417e+00
     6.69567112e-01 2.95522948e+00 2.60906080e-01 5.57112146e+00
     2.10212807e+00 1.12880479e+00 2.76808781e+00 1.88976082e-01
     4.98017460e+00 5.88437437e+00 1.94209856e+00 1.98127596e-01
     1.09772675e+01 4.39220600e+00 3.18074354e+00 2.06182985e+00
     8.93596813e+00 1.60528680e+00 5.01787357e-01 7.90954660e-01
     1.23258511e+00 1.37446709e+00 1.02278567e-01 2.00584232e+00
     1.42839458e+00 1.43913286e+01 1.87125530e+00 1.73327833e+00
     1.24353361e+00 7.78696003e+00 1.74323636e+00 2.63641353e+00
     1.27964228e+00 4.87188260e+00 4.52234861e+00 8.86274278e+00
     1.01540912e+00 3.45771442e+00 1.70279426e+00 3.22955586e+00
     1.01252797e-01 4.63851927e+00 2.14814551e+00 3.92592786e+00
     3.22573470e-01 1.18405770e+00 5.79393208e+00 2.81374684e+00
     2.86979774e-01 2.96344280e+00 6.07563754e-01 1.70569603e+00
     2.70972784e+00 1.43421039e+00 1.64407316e+00 1.06840426e+00
     1.75479151e+00 1.92447066e+00 5.47938334e+00 2.33621299e+00
     3.78795936e+00 3.37758663e+00 4.47017499e-01 7.09536527e-01
     1.52524397e-01 7.47579975e-01 5.91490229e+00 4.77638880e+00
     2.27640401e+00 5.22545123e+00 2.05348368e+00 5.04944669e+00
     2.66142383e+00 2.24724742e+00 1.03846753e+00 9.48075522e-02
     2.19008738e+00 2.35761537e-01 9.53983673e+00 5.84156299e-01
     1.88548367e+00 4.68684923e+00 1.13505154e+00 5.17325271e+00
     6.78383111e-02 4.33666058e+00 4.38574253e+00 3.77898912e+00
     3.80792037e-01 8.83320918e-01 3.77080851e+00 5.44041981e+00
     3.83527856e+00 6.67771454e+00 2.73554287e+00 1.17944775e+00
     2.60222668e+00 5.36844383e-01 8.49606206e-01 1.49750888e+01
     3.53451199e+00 1.99443727e+00 1.79987143e+00 2.17853549e+00
     1.16704895e+01 2.14446651e+00 7.41172552e-01 5.07950022e-01
     1.85427704e+00 1.13094824e+01 3.82267109e+00 1.11548849e+00
     5.19921401e-01 3.35476496e-01 8.32740533e-01 1.31599000e+00
     2.51935031e+00 2.61452239e+00 4.53830912e+00 2.53407734e+00
     1.02690602e+00 4.52013456e-01 1.77372593e+00 1.36206542e+00
     7.29388808e+00 2.68284836e+00 1.94708675e+00 3.61359619e+00
     2.63550302e+00 5.53263027e-01 3.72461413e+00 3.78359740e+00
     1.03953715e+00 2.59318251e+00 6.04562410e+00 4.95609779e+00
     5.76909200e-01 9.68875625e-01 3.03689988e+00 4.03383787e-01
     9.43789100e+00 1.05383357e+00 1.76006055e+00 1.34478352e+00
     2.69224922e+00 3.13647352e-01 4.09334690e-01 4.51509374e-01
     1.97167167e+00 3.82045173e-01 6.69200879e-01 4.11623864e+00
     1.90384094e+00 2.55124473e-01 6.00794476e-01 2.24432606e-01
     3.06966580e+00 2.97844262e-01 1.31969389e+00 3.03716832e+00
     1.84389916e+00 1.05923232e+00 9.05791974e-01 8.02669868e+00
     4.45700936e+00 2.74752663e+00 7.65940664e+00 5.86411517e+00
     3.46876707e+00 3.86778182e-01 1.57650610e+00 3.65816222e-01
     4.49989957e+00 5.81619978e-01 3.81963997e+00 1.22395274e+00
     4.31878632e+00 5.22702011e+00 5.70172115e+00 2.65025253e+00
     1.14786179e+00 1.70401096e+00 6.11024457e-01 7.79856321e+00
     2.17051357e+00 6.34010107e-01 2.66112712e+00 2.88233417e+00
     1.82404622e+00 1.66377880e+00 1.85310395e-01 4.62532353e+00
     1.67866231e+00 1.14674614e+00 7.53636964e-01 1.63552975e+00
     8.89820210e+00 1.31171316e+01 2.54807179e+00 1.52585727e+00
     2.82289199e+00 1.10808853e+01 1.16653031e+00 1.49583008e+00
     6.51038071e+00 4.51734618e+00 4.62787395e+00 2.32905973e+00
     1.63896604e+00 2.70146693e-01 1.03944635e+00 2.77246146e+00
     2.88761736e+00 3.35024949e+00 1.52615193e+00 2.37831035e+00
     1.29756577e+00 3.93682339e+00 5.56964459e+00 1.88666490e+00
     1.09924585e+00 9.69992557e-01 1.29311257e+00 4.94262676e+00
     2.86520813e+00 1.45979503e+00 6.61258468e+00 2.86538140e+00
     2.79201632e+00 5.58020826e+00 1.23268231e+00 6.98619123e+00
     1.80866829e+00 2.14555667e+00 5.55381500e+00 2.63468530e-02
     2.75528742e-01 2.99105813e+00 1.46763463e+00 1.95675017e+00
     4.12189030e+00 9.87500181e-01 3.02362759e+00 2.85533684e-01
     5.43483911e+00 1.56900769e+00 6.96372755e+00 6.71792009e+00
     7.03339720e+00 5.09647891e+00 4.23158088e+00 1.91207992e+00
     3.06453664e-01 9.54168815e-01 3.72709553e+00 1.37936040e+00
     3.73403539e+00 1.28880645e+00 3.12574222e+00 5.85583383e-01
     2.34355710e+00 5.66647974e-01 2.75149615e-01 3.91519318e+00
     2.52072126e+00 1.92167117e-01 2.27288967e+00 7.96031587e-01
     1.54224828e+00 6.25401240e+00 2.65857027e+00 1.25737821e+00
     1.05894417e+00 1.32909100e+00 1.02445360e+00 1.93848862e+00
     8.98683632e-01 1.38418785e+00 7.37605947e-01 2.84069634e+00
     1.35431649e+00 2.81984050e+00 1.49357528e+00 6.05998790e+00
     4.10419717e+00 1.05194078e+01 1.40760654e+00 4.83557039e-01
     3.86774611e-01 1.96508159e+00 1.38276736e+00 4.75241046e-01
     2.59049055e+00 1.96002508e+00 6.04523765e+00 4.78410718e+00
     6.04070816e+00 1.60898520e-01 4.69941660e+00 4.22124074e+00
     1.63106799e+00 9.98511535e-01 2.14466972e+00 6.69018084e-02
     1.36641871e+00 2.88378240e-01 8.32131562e-01 2.39811879e+00
     1.25572643e+00 5.04337570e+00 3.18339498e+00 6.32243087e-01
     2.70251304e-01 2.18861525e-01 5.04322683e+00 6.97755227e-01
     1.97303935e-01 9.97614266e-01 5.01259904e+00 2.74064495e-01
     3.45368075e+00 4.92261833e-01 6.52209328e-01 1.61039550e+00
     1.45649610e+00 6.70883693e-01 6.04581326e+00 9.14727011e-01
     2.66932393e+00 1.32538290e+00 7.86945619e+00 1.84955471e+00
     1.05319560e+00 1.36957894e+00 4.60332569e+00 3.93043467e+00
     2.84673374e+00 2.78249030e+00 4.74885964e+00 2.53125844e+00
     8.47983678e-01 8.07415053e+00 6.51460568e-01 4.72348090e+00
     2.29684756e+00 3.59069190e+00 6.36492635e+00 6.02900961e+00
     2.16629876e+00 5.42280962e+00 9.41127835e-01 3.38204305e+00
     1.72592763e+00 2.21902121e+00 2.51644770e+00 1.70384538e+00
     4.57930301e-01 5.20323374e-01 4.57075684e-01 4.72414034e+00
     7.86549778e-01 2.34109942e+00 1.15656141e+00 1.07389043e+00
     1.64418249e+00 3.07195732e+00 5.04053530e+00 1.46944921e+00
     6.05295756e+00 1.62956732e-01 3.01770092e+00 1.98804598e+00
     7.64883859e-02 5.39414140e+00 3.17273819e+00 7.98275504e-01
     1.25435322e+00 3.83550419e+00 1.60717554e+01 2.34489081e+00
     1.94943776e+00 2.17694006e-01 1.06277574e+00 1.18376055e+00
     6.97401959e+00 1.97740891e+00 1.98879038e+00 5.47256537e+00
     3.63483277e-01 7.95203717e+00 6.57287196e+00 8.96962247e+00
     5.52348840e+00 2.82365285e-01 2.00816996e+00 1.25693830e+00
     4.92610730e+00 8.49916276e-01 6.05220618e+00 1.55881487e-01
     7.22917286e+00 8.01304495e-01 1.18267931e+00 5.10608865e+00
     9.68556282e-01 5.29706491e-01 4.86228856e+00 3.31710111e-01
     1.87605509e+00 2.68415975e+00 1.69792009e+00 3.38109732e+00
     3.86290447e+00 2.45069986e+00 2.68877463e+00 8.35836726e+00
     4.17095082e-01 4.92574234e+00 7.06664664e-02 2.73046159e+00
     4.56161469e+00 2.41981347e-01 1.53769948e-01 2.69988114e+00
     9.66725196e+00 2.29510403e+00 7.09862842e+00 9.75208227e-01
     1.17858259e+01 6.91183086e-01 8.48316687e-01 2.27724966e+00
     8.41408164e-01 6.21331252e+00 2.25703665e+00 4.85055721e-01
     4.17545524e+00 4.31840411e+00 1.28205304e+01 1.91711351e+00
     1.40209317e-01 4.57103792e+00 6.23596000e+00 5.61343679e+00
     1.66939977e+00 4.40178104e+00 2.83084289e+00 1.41890218e+00
     5.38200445e+00 3.49767408e-01 4.51583325e-01 2.27908470e+00
     3.36370269e+00 3.29805716e+00 3.64552643e+00 6.86454572e-01
     7.03148444e+00 1.32956830e+01 7.66583445e+00 2.13233459e+00
     3.30491462e+00 7.20341391e+00 6.77918497e-01 5.09423116e+00
     1.29007104e+00 3.93538867e+00 9.81737566e-01 3.17802945e-01
     1.93200693e-01 1.12754513e+00 5.66594478e+00 1.34696840e+00
     6.43905287e+00 5.27131635e+00 1.57404418e+00 7.56445836e-02
     8.42450346e+00 6.06756901e-01 9.84843468e+00 9.00212917e-01
     6.13035437e+00 2.58869216e+00 1.11720265e-01 7.35767298e+00
     1.17706081e+01 5.78269640e-01 7.27137228e-01 1.10363386e+00
     8.50221206e+00 7.01780429e+00 5.19553821e+00 1.15681101e+00
     6.47318714e+00 9.87897478e-01 3.24657839e+00 1.88308364e+00
     1.00366188e+00 2.70372094e+00 8.34425558e+00 4.14430309e+00
     3.75414100e-01 1.90414902e+00 2.57247502e+00 2.40372677e+00
     1.72416001e+00 3.09272385e+00 7.28251252e+00 3.31603122e+00
     6.93900087e+00 1.05867854e+00 1.42939760e+00 2.79657429e+00
     4.07655571e+00 6.58159138e+00 1.23201829e-01 3.81102477e+00
     2.34795527e+00 9.95929891e+00 3.20466540e-01 7.16988497e+00
     1.04771234e+01 1.57583328e-02 5.82062782e+00 6.31787682e-01
     9.80778557e-01 2.96117496e-01 1.00056229e+00 2.88455776e+00
     1.01619370e+01 5.52370397e-01 5.52174595e+00 2.30751466e+00
     1.83049061e-01 4.11367094e-01 4.88368102e-01 4.92979997e+00
     1.52491914e+00 1.81118669e-01 3.44368093e+00 2.79062211e+00
     8.06716702e+00 2.89201994e+00 1.42085032e+00 3.68834779e+00
     3.55116464e+00 3.36121899e-01 8.40574205e+00 2.75804349e+00
     2.73644548e+00 5.25375834e+00 2.65142628e-01 7.08791370e+00
     4.33471215e+00 2.75418735e+00 1.75411129e-01 3.64320681e+00
     5.62901919e+00 1.13415398e+00 3.42201769e+00 1.50676831e+00
     4.39514645e-01 3.84131499e-01 3.80949509e-01 1.34852566e+00]
    θ estimado: 3.0761, número de iteraciones: 57
    [4.58240986e+00 2.27735069e+00 7.21352162e+00 5.21014793e+00
     5.02246181e+00 9.59098136e-01 4.46849372e+00 5.73350016e+00
     4.25071280e-01 9.93118126e-01 8.43943755e+00 1.12179884e-01
     4.04906055e+00 3.86826059e+00 8.19650913e-01 1.17260632e+01
     5.32674756e-01 7.21360964e-02 8.22854627e-02 8.13786589e+00
     4.26480426e+00 6.15607741e+00 3.06857266e+00 3.32684408e+00
     4.72802556e+00 9.19918595e-01 1.77311121e+00 1.81307866e+00
     9.04005250e-01 2.19883356e-02 7.14253477e+00 1.63709862e+00
     3.96232811e+00 2.87898504e+00 2.99006051e+00 3.89409682e-01
     3.05081685e-02 3.37070744e+00 4.31209255e+00 2.83167765e+00
     4.88062403e+00 1.36031185e+00 1.00360717e+00 1.87491202e+00
     3.82317016e+00 7.03476459e-01 5.52060731e+00 3.00304578e+00
     6.38468342e-01 5.30987978e+00 1.64896262e+00 3.48138987e+00
     8.88451739e+00 2.60333789e+00 6.02102745e-02 3.76680733e+00
     8.98455466e-01 1.10978255e+00 3.10663452e+00 1.18845339e+01
     1.23302938e+00 2.04125235e+00 1.06907270e+00 9.23061607e+00
     3.65147115e+00 4.59867697e+00 1.99890480e+00 3.07911827e+00
     6.50397302e+00 5.92546552e+00 2.36280978e+00 3.34948005e+00
     8.36416048e-01 4.97284550e+00 8.94338619e-02 2.09114793e+00
     1.02503415e+00 5.21252015e+00 8.11048674e+00 7.28379334e+00
     6.50365304e-02 6.80948033e-01 4.82640752e+00 4.10406728e+00
     4.35913121e+00 2.08674608e+00 3.71172106e-01 2.43919211e+00
     2.23602638e+00 4.40942176e+00 2.03540829e-01 1.57516947e+00
     4.98763643e+00 7.16144564e-01 3.84141632e+00 1.21742427e+00
     8.97551384e-01 1.74782957e+01 1.88508068e+00 6.13950884e+00
     8.23186142e+00 2.24357343e-01 6.02319681e+00 3.61627938e+00
     3.38187362e+00 1.85990354e+00 7.25500657e+00 1.05437607e+01
     2.14180506e-01 7.94421034e-01 1.81717101e+00 1.14177656e+00
     1.90459240e+00 5.71482014e-01 1.14470959e-01 1.27214924e-02
     6.19588322e-01 2.90772476e-01 7.72790968e+00 6.99118523e-01
     4.69864167e+00 1.56031707e+00 8.81587445e-01 1.92809426e+00
     2.17515307e+00 5.75174184e+00 2.00400567e+00 5.26087865e+00
     1.17312600e+00 8.80746574e-01 6.95069784e+00 6.04362275e+00
     3.28001400e-01 3.26936655e+00 2.68541822e+00 1.58184342e+00
     4.43214616e+00 1.08832926e+01 3.40909469e-01 4.23002352e+00
     5.24726574e+00 4.63480045e-01 2.92508982e-02 1.07238707e+01
     1.16594875e+00 3.27543356e+00 3.06142869e+00 9.25208604e-01
     1.06947032e+00 7.09171313e+00 2.89264768e+00 3.56631125e+00
     1.74195254e+00 2.11761911e+00 2.56776860e-01 1.87898628e+00
     3.09947637e-02 9.49829006e-03 6.97917498e+00 1.12357562e+00
     5.19927412e+00 3.90619156e+00 5.95448073e-02 7.34224552e-01
     1.51236664e+00 4.81573482e-01 3.67600588e+00 1.16055876e+00
     4.02071285e+00 1.27379372e+00 5.28513203e-01 6.13920416e+00
     3.18701268e+00 6.16270391e-01 7.06245688e+00 1.41394830e+01
     2.65700437e-01 1.47567967e+00 8.55270929e-01 3.80265405e+00
     3.05909771e+00 1.89992471e+00 6.47877336e+00 7.56269419e-01
     2.49604802e-01 1.47040942e+00 2.84601170e+00 1.09624875e+01
     4.29852349e+00 9.72087929e-01 5.03348703e+00 3.98007084e-01
     3.75622371e-01 3.83331561e+00 2.98183466e-02 2.89662272e-01
     6.76503025e+00 2.69904084e+00 4.60535241e-01 4.95055038e-01
     3.69895831e-01 7.56860718e-01 4.88061133e+00 7.94019298e+00
     2.71712994e-01 5.35812389e-01 6.00153534e-01 1.20799158e+00
     3.45224919e+00 4.12371849e+00 3.07597119e+00 4.41209838e+00
     1.65002612e+00 1.28414202e+00 3.81684609e-01 2.83567555e-01
     4.42753504e+00 4.97061079e-01 7.64337400e+00 2.65610627e+00
     1.92660273e+00 7.03012251e-01 2.99503267e+00 2.00578958e+00
     1.81572344e-01 4.32151567e+00 1.34818736e+00 3.71531703e-01
     2.33349226e+00 3.57242573e+00 6.65280421e+00 4.00301355e+00
     4.03501510e+00 1.19447405e+00 4.16197779e+00 1.36617999e+00
     3.52581867e+00 2.35827015e+00 5.41203480e-01 1.43380530e+01
     2.72067874e+00 1.05121481e+00 1.30014214e+00 1.98566796e+00
     1.02756903e-01 8.91414482e+00 5.25704556e+00 6.04208563e+00
     4.69184126e+00 6.84599873e-01 3.78959939e+00 2.15367123e+00
     1.73798244e-01 2.99121193e+00 1.48140705e+00 1.50052796e+00
     5.19306639e+00 1.86272054e+00 6.34630852e+00 1.32948864e+00
     2.09987501e+00 1.06287773e+01 7.32037089e-02 2.13863206e+00
     2.95341162e+00 4.17313497e+00 2.21696785e-01 1.46508053e+00
     6.57649314e+00 4.66216021e-01 1.88025006e+00 4.11558793e-02
     7.07331679e-02 1.24390786e+01 9.74712967e-01 2.04590440e+00
     6.74762215e+00 7.27456807e+00 2.33167189e+00 2.93071193e+00
     7.39010412e+00 8.84342592e-01 2.78054742e+00 2.79664206e+00
     3.72032153e+00 1.16064832e+00 4.51245338e-02 4.05726642e-01
     1.79889568e+01 2.28746671e+00 7.80008155e-01 6.29006451e+00
     2.38394770e+00 8.43759212e-01 1.14362060e+01 3.78975522e+00
     3.29881108e-01 4.53590963e+00 3.27993047e+00 4.22965916e-01
     6.09635445e+00 2.73671994e+00 6.67562548e-01 3.15800441e+00
     8.62290646e-01 3.89068186e+00 3.34213112e+00 6.09655942e+00
     7.00911478e-01 1.08539082e+01 9.57316883e-01 1.67174621e+00
     5.78992979e+00 4.33351710e-01 2.42626167e-01 5.59138561e+00
     1.55696478e+00 1.38944511e+00 5.48370620e-01 8.60491942e-01
     1.18927889e+00 6.64096728e+00 3.03179180e+00 3.77250929e+00
     1.55012683e+00 3.91120276e+00 1.58339728e+00 2.55226564e+00
     4.82197740e-01 1.02609672e+01 1.25576351e+00 1.95482478e+00
     3.58481902e+00 9.47748647e-03 3.62074018e-01 4.49581486e+00
     9.78532282e+00 7.54538474e+00 7.19013227e-01 4.52882114e+00
     5.11501484e-02 2.30787633e+00 6.74817570e-01 5.45938705e+00
     3.09777333e+00 7.43193083e+00 1.75002235e+00 1.79091197e+00
     1.14170938e+00 1.13140085e+00 1.35575573e+00 1.48225214e+00
     1.27487978e+00 4.45620206e+00 5.62563673e+00 3.18141452e+00
     9.42938582e-01 2.45174816e+00 2.84254533e+00 1.24608609e+00
     7.90677112e-01 1.61933601e+01 1.04479579e+01 3.67172129e+00
     9.15080623e-01 2.71066848e+00 7.58010645e+00 3.07806234e+00
     1.16146406e+00 3.94893220e+00 1.30894153e-01 2.60984328e+00
     2.29370787e+00 6.66199374e-02 2.89783734e+00 4.90404621e-01
     4.09407519e+00 1.29453857e-01 7.56621135e-01 1.91435660e+00
     2.06688244e+00 1.12020667e+01 7.63523780e-01 1.26911293e+00
     4.34002090e-01 2.28151665e+00 8.97698438e+00 9.62955922e-01
     8.88059733e+00 8.55244698e-01 6.22448545e+00 9.55947137e-01
     7.19672015e-01 2.31759961e+00 3.04571556e+00 2.13616367e+00
     1.71998853e+00 1.45130902e+00 8.88387552e-01 1.30252701e+00
     6.52665883e+00 4.77975842e-01 6.20012162e+00 9.57632837e+00
     2.48050838e+00 1.37873810e+00 2.35839849e+00 4.00166251e-01
     4.02989653e+00 3.82739801e+00 3.06643860e+00 2.18097923e+00
     7.42247976e+00 1.37663706e+00 1.87770310e-02 6.88484619e-01
     2.90216304e+00 2.63771209e+00 1.61944897e+00 2.71511287e+00
     1.49067322e+00 5.57718249e+00 4.29891803e+00 2.95615737e-01
     2.92047570e-01 2.67531553e-02 5.81158450e-01 1.23697966e+00
     2.76401262e+00 1.60302649e-01 2.62821245e-01 2.99905793e+00
     4.85428934e+00 7.02482148e+00 1.82754418e+00 1.58122654e+00
     1.27796601e+00 3.62198562e+00 2.52875021e+00 4.34636510e+00
     7.41724261e+00 9.66798361e+00 4.40641385e-02 3.35970275e+00
     3.53805598e+00 5.21189434e+00 1.04798895e+00 3.04950420e+00
     1.73939754e+00 1.06158739e+00 1.87255205e+00 2.94463883e+00
     1.52555483e+00 1.09975564e+00 1.19571968e+00 1.44240107e+00
     3.00423779e+00 3.00994866e+00 1.94436041e+00 6.28948314e+00
     1.83591012e+00 3.04897588e+00 6.47886596e+00 5.75509175e+00
     2.98062948e+00 5.23744411e+00 9.59390414e+00 3.27046154e+00
     6.36665317e-01 1.10172421e-01 4.68879203e+00 4.73119638e-01
     4.55418275e+00 3.08899050e+00 2.81316207e+00 1.21897681e+01
     3.37004303e-02 9.65549171e-01 7.53213491e+00 7.79137558e-01
     2.49116037e-01 1.74220204e+00 3.63280443e-01 4.74888885e+00
     2.05322192e-01 2.27154828e+00 3.38094662e-01 6.78297029e-01
     4.55013195e+00 1.06334872e+00 3.17635068e-01 2.61735131e+00
     1.34672827e+00 5.71421086e+00 2.29111170e-01 1.15859447e+01
     3.90116862e+00 3.63457592e+00 6.87088472e-01 1.47285060e+00
     2.79384094e+00 3.14151016e+00 3.06577257e+00 9.40897160e-01
     5.41734544e+00 2.41349625e+00 2.23473668e+00 7.45155491e-02
     1.22075067e+00 9.52742859e-01 8.92901156e-01 4.09097299e+00
     2.42432709e+00 2.52444394e+00 1.08990894e-01 3.59694056e-01
     1.03293091e+01 1.56420058e-01 4.38067974e-01 7.65590532e+00
     3.19648653e+00 1.77570043e+00 5.40193542e-01 3.70616332e+00
     1.35274856e+00 2.73386848e+00 1.39995721e+00 9.75432572e-01
     5.18991305e+00 3.76356547e+00 2.11228445e+00 1.71948879e+00
     3.76399546e+00 2.48515481e+00 9.12392143e-01 9.31166072e-01
     8.80469216e+00 3.24550375e+00 1.60667409e-01 3.20678087e+00
     1.65078551e-01 5.95632781e+00 8.07316301e-02 6.56643385e+00
     1.12625182e+00 5.02130147e+00 2.56530346e+00 6.04604599e+00
     4.04890479e-01 3.72286046e+00 4.96374613e+00 3.82895580e+00
     9.96361589e-01 5.07538193e+00 2.94601993e-01 7.04303415e+00
     5.29326641e+00 4.65483884e+00 6.40521302e+00 3.89924785e+00
     9.32671242e+00 1.91655266e+00 1.72611076e+00 4.86096918e-01
     1.08308325e-01 5.80752627e+00 1.57153089e+00 1.66357157e+00
     4.19949679e+00 3.58047884e+00 4.01547100e+00 2.03521628e-01
     8.00186673e+00 5.63251252e-01 1.64584076e+00 4.11205171e-01
     1.32595667e+01 2.58588563e-01 4.93813140e-01 4.86614717e+00
     9.42260803e+00 4.84103172e-03 4.35023871e-01 6.72381369e+00
     7.87650858e-02 9.69135197e-01 1.62342173e-01 8.43033131e-01
     7.10143243e+00 4.62971312e+00 9.43763769e+00 2.38341502e+00
     9.85714602e+00 3.43321753e+00 4.35137643e+00 8.76441003e-01
     4.61917993e+00 1.62557374e+00 3.64257333e+00 1.85347604e+00
     5.12258297e+00 1.41579653e+00 1.35611138e+00 5.15509225e+00
     5.62081881e+00 9.54253470e-01 1.25407836e+00 2.06382428e-01
     1.16191411e+00 2.19557789e+00 1.12970094e-01 9.68227792e-01
     6.74904869e+00 6.05147532e+00 1.31148041e+00 3.59426846e+00
     6.37772517e+00 1.97597566e+00 4.39827443e+00 5.16653353e+00
     8.10920808e-01 8.87695207e+00 1.18034021e+01 1.20354548e+01
     4.12179134e+00 4.19102174e+00 1.92374537e+00 3.76643750e+00
     1.64441772e+01 4.39704673e+00 3.34773095e-01 1.09888999e+00
     1.78210618e+00 1.07973533e+00 3.17465022e+00 2.29972923e+00
     9.22903484e-02 2.57642364e-01 9.27600430e+00 2.01190487e+00
     7.24554722e-01 7.74989114e-01 1.64087243e+00 1.87322394e+00
     4.16789875e-01 3.76982525e+00 4.05102199e+00 1.41218740e+00
     4.01565632e+00 4.36845622e+00 3.81710459e-01 2.17433042e+00
     9.01046124e+00 2.66610161e+00 4.89458931e+00 7.94959065e-02
     3.66480222e-01 4.55425599e+00 1.31593927e-01 4.48927405e+00
     3.08192856e+00 1.69969749e+00 6.62401831e+00 3.37601658e+00
     1.68734481e-01 2.05187525e+00 7.08000411e+00 3.51457237e+00
     1.76276219e+01 2.10237587e+00 4.27547812e+00 2.40113858e+00
     6.93792908e-01 2.06527375e+00 2.63823237e+00 3.71197292e+00
     7.11926758e-01 4.47291250e+00 3.36871862e+00 1.91038344e+00
     4.33544697e+00 9.53399757e-01 5.21080324e+00 1.18855431e+00
     4.48510548e+00 2.00398130e+00 2.14641704e+00 1.29995040e+00
     5.04172963e+00 4.94229232e+00 3.41436760e+00 3.24454129e+00
     2.70675425e+00 7.47944122e-01 5.64002240e+00 3.01831356e+00
     2.66781728e+00 2.78178733e+00 4.97528891e-01 1.32622372e+00
     6.55445434e-01 1.44672167e+01 2.68625845e+00 8.36364218e+00
     2.71296729e+00 2.37421509e+00 2.78893319e+00 7.56902347e-01
     9.00989752e+00 1.00145048e+01 1.60237160e+00 8.02736346e+00
     4.40067874e+00 1.30922056e+00 1.18308591e-01 1.54636799e+00
     8.60776905e-01 7.06856317e-01 1.12184248e+00 1.90529584e+00
     1.58362272e+01 1.60782977e+00 5.32673865e+00 7.11247228e-01
     2.53155233e+00 1.18085152e+00 2.20246890e+00 2.01274819e-01
     6.64134088e-01 4.66756696e+00 8.12987958e-01 1.29039701e+00
     1.16861703e+00 3.31863601e+00 1.48080349e-02 4.29360053e-01
     9.99562192e-02 1.64913300e-02 2.66890390e+00 1.62619882e+00
     5.24997352e+00 3.55725913e-01 6.65550397e-01 3.23706774e-01
     1.54474512e+00 4.74101719e-01 2.25149402e+00 5.11236228e+00
     7.19913161e-01 6.66978222e+00 1.13157231e+00 2.49656897e-02
     1.48562544e-01 2.03685160e+00 5.54286021e+00 1.28466448e+00
     4.81065658e-01 2.39051337e+00 9.38925223e+00 1.26387929e+00
     1.95494482e+00 1.51568391e+00 7.91897611e+00 3.58627307e+00
     2.52571166e+00 2.13291013e+00 7.33363749e+00 2.37983344e+00
     3.93366061e+00 6.66507058e+00 4.27063848e+00 6.77616585e-01
     1.59141020e+00 1.23806546e+00 5.33632005e+00 7.59189434e-01
     2.77636072e+00 9.46888373e+00 1.57755427e+00 1.60612578e-01
     7.11745986e+00 6.26586679e+00 3.81438108e+00 1.25760673e+00
     1.45987782e+00 2.14730793e+00 9.87689998e-01 7.98101212e+00
     2.08404801e+00 4.41543768e+00 1.13532042e+00 9.15299130e-01
     1.53874728e+00 1.17103347e+01 2.43292358e+00 4.68536131e-01
     2.06595350e+00 1.76453241e+00 4.71301531e-01 3.32265272e-01]
    θ estimado: 3.1065, número de iteraciones: 36
    [9.20141045e+00 5.49366876e+00 1.05936053e+00 3.27477489e+00
     1.71870750e-01 9.00504801e+00 2.17087220e+00 5.90224525e+00
     2.71363421e+00 1.18976138e+00 5.26392067e+00 2.71891279e-01
     3.42442528e+00 4.19191567e-01 1.19568538e+01 2.25281949e+00
     1.87335661e+00 1.24099628e+00 4.65407029e+00 4.00538087e+00
     4.50079647e+00 1.66157683e+00 2.96654447e+00 7.40726575e-03
     7.92393141e-01 2.13731599e+00 9.63489498e+00 2.79139262e+00
     8.09311918e-01 9.74191464e-01 6.20849529e+00 2.38732915e+00
     5.33531291e+00 2.56301526e+00 2.37709064e+00 2.86832305e+00
     1.82953551e-01 1.57512194e-01 3.16958506e-01 1.19199705e-01
     3.16460968e-01 3.88797926e+00 8.30162170e-01 1.78477034e+00
     1.14370777e+00 2.86160456e+00 9.71569628e-01 1.09277800e+00
     1.21679153e+00 1.48211498e+00 3.09972781e+00 1.01328141e+01
     2.77907383e+00 1.57075753e+00 1.47951624e+01 2.57360936e-02
     5.63797044e-01 1.92868633e+00 5.23099820e+00 7.44332817e-01
     1.16246266e+00 4.20400719e+00 1.78487847e+00 3.59276693e-02
     4.38792909e+00 8.57865268e+00 1.20366534e+00 1.84092736e+00
     6.13843897e-01 3.68218510e-01 6.95319256e+00 1.39001051e+00
     5.72607080e+00 1.93590989e-01 2.00757904e+00 1.94987516e+00
     2.05601087e+01 3.27837187e+00 6.16834609e+00 1.30479206e+01
     1.12994624e-01 3.88801427e+00 1.06562429e+00 2.09183001e+00
     1.97323661e+00 2.82633615e+00 7.39552681e-01 1.16786373e+00
     9.74977253e-01 3.42547141e+00 9.50517399e-01 1.17003051e+00
     2.32866873e-01 1.81413965e+00 1.69672133e+00 5.10465165e-01
     5.24427036e+00 1.15114592e-01 9.32751331e-01 2.71258263e+00
     4.87322180e-01 3.73989859e+00 3.36784367e-01 7.87633008e-01
     4.55158113e+00 2.38435394e+00 2.56988320e+00 9.75349937e-03
     9.10831424e+00 2.11782407e+00 1.55000341e+00 2.11826952e+00
     1.29870625e+00 4.51456423e+00 6.11986426e+00 8.26546495e-01
     3.22682401e+00 6.85626199e+00 5.56846770e+00 6.29646059e+00
     8.85235434e-01 5.28453307e-01 7.57963638e-01 1.74427472e-01
     1.05285453e+01 2.16005373e+00 1.20082766e+00 2.39828488e-02
     4.25341669e-01 1.45730207e+00 2.76913816e+00 1.02170133e+00
     2.57081608e+00 9.96639965e-01 3.74060912e-01 2.69534138e+00
     6.12687032e-01 5.88624274e+00 3.46979707e+00 9.12725882e+00
     5.82427390e-01 1.55253812e+00 3.51899005e+00 7.28109393e-01
     4.17422402e-01 6.75537341e+00 1.26816944e+00 1.04096384e+00
     1.22059484e+00 2.91703965e+00 2.14591513e+00 2.88545378e+00
     3.67619566e-01 3.54835167e+00 5.26818902e-01 2.50114481e+00
     8.50561957e-01 6.39341411e+00 7.30562344e+00 8.69265054e+00
     9.53833493e-01 2.14047205e+00 1.39349308e+00 8.45622819e-01
     9.09020795e-01 1.72715692e+01 8.10735724e+00 1.80199112e+00
     7.50159615e-01 4.03900037e+00 3.32161116e-01 2.34535860e+00
     4.00444197e+00 6.74389817e-01 2.12152369e-01 4.29639238e+00
     7.76111061e-01 4.84158419e+00 6.54953558e+00 7.66850516e-01
     1.40830269e+00 1.47022844e+00 3.22938363e+00 5.48473228e-01
     2.62349973e+00 1.36598692e+01 3.22916768e+00 1.31691783e-01
     4.83171635e+00 1.53180138e+00 8.76197065e-01 4.27066765e+00
     1.84751157e+00 1.16897240e+00 2.61148030e+00 8.94255463e-01
     1.00378614e+01 2.81692441e+00 8.73962413e-01 4.82959619e+00
     2.88052104e+00 1.20781263e+00 4.44525391e+00 4.93437674e+00
     1.05563697e+00 7.43801631e+00 3.67353425e+00 1.37863832e+00
     7.37943866e+00 8.57999775e-01 7.60673063e+00 2.56823886e+00
     6.41299777e-01 3.18545025e+00 1.03371723e+00 5.35354091e+00
     1.08698190e+01 2.67929892e+00 1.33470621e+00 2.94258266e-01
     8.48080354e-01 5.33518635e+00 4.57240054e+00 9.92961145e-02
     3.96350128e+00 5.20463120e+00 8.96094785e-01 4.43878285e+00
     1.24179236e+01 4.63003197e+00 3.20058383e+00 8.92675288e-01
     4.37487913e-01 4.48745946e+00 1.37268645e+00 4.55262358e-01
     2.23639074e+00 1.92290443e+00 3.54101192e+00 6.01908184e+00
     7.04191149e+00 7.39765863e-01 7.67225691e+00 2.18592319e+00
     5.40237809e+00 9.66683577e+00 8.79156774e+00 1.40440921e+00
     5.48638157e+00 2.98059617e+00 1.63553746e+00 6.82585340e-01
     2.95621321e+00 1.13958324e+00 2.15388281e+00 4.64309330e-01
     9.61990575e-02 6.43634067e+00 2.21331832e+00 2.30982846e+00
     5.43387964e-01 2.99539954e+00 2.30413054e-01 2.99488204e+00
     2.61886207e+00 5.33950779e+00 8.36326939e-01 3.73309395e-01
     3.42116200e-01 1.96526945e+00 1.51157805e+00 1.60991991e+01
     1.35496285e+00 5.36982741e+00 2.67384652e+00 1.13508505e+00
     7.26004594e+00 2.31617948e+00 1.32282260e+00 3.75179112e-01
     4.40834565e+00 1.73774824e+00 6.09241379e-01 3.03368662e+00
     2.89080641e+00 4.94896611e+00 3.59315984e+00 2.35482073e+00
     4.85615825e+00 1.47565943e+00 4.44263898e-01 7.67945668e+00
     8.16896695e-01 2.25071055e+00 2.23764387e+00 1.19558693e+01
     3.16862216e+00 9.85942864e+00 2.95107081e+00 1.14296898e+00
     1.70133143e+01 1.77891418e+00 5.81635069e+00 1.60725112e-01
     1.52729672e+00 3.47014203e+00 1.19317527e+01 3.15997128e-01
     1.23573462e+00 4.30164258e+00 5.44680986e+00 1.76804725e+00
     6.88767556e-01 3.70987800e+00 8.36970627e+00 7.91318059e-01
     4.50412165e-01 1.74192431e+00 3.18939962e+00 3.92220296e-01
     2.70749944e+00 9.15688760e-01 3.26660996e+00 6.00002811e+00
     2.48883609e-01 1.57979076e+00 2.05466413e+00 9.19343568e+00
     4.86108706e+00 9.41619311e-01 4.07463412e-01 4.72052072e+00
     2.40642151e+00 4.59680786e+00 5.17565558e-01 4.75597966e+00
     1.60799675e+00 6.17563462e+00 5.80891330e-01 1.21643106e+00
     5.07010517e+00 8.93613733e+00 2.55556160e+00 9.59881298e+00
     3.53321317e+00 6.66870012e+00 4.68047661e+00 6.69399966e+00
     2.95055604e+00 6.77476814e+00 4.46862882e+00 1.38801508e+00
     2.82085609e+00 1.31121580e+00 5.69555236e+00 4.42402716e+00
     1.62327090e+00 8.62058330e-01 3.67572632e-01 1.25577337e+00
     4.71844671e+00 1.42554042e+00 9.67654159e-01 2.13254707e-02
     2.56730364e+00 2.10400699e+00 5.26281137e+00 9.37259803e+00
     5.41083150e-01 6.25337527e+00 6.45143532e-01 1.08140093e+01
     3.71876290e-01 2.62035417e-02 1.77926536e+00 2.89664095e+00
     1.92255793e+00 1.66103671e+00 8.63072769e-01 1.00754529e+00
     1.52565789e+00 3.28557385e-01 6.62182097e-01 8.73561352e+00
     5.88396476e-01 1.01568053e+01 1.45456226e+00 2.00842464e+00
     5.83833258e-01 1.88955099e-01 1.46903767e-01 4.82076364e-01
     2.79408919e+00 1.31523276e+01 4.60241012e+00 6.45823812e+00
     7.70431680e-01 2.28395757e+00 1.52948922e+00 5.46613802e+00
     1.24813310e+00 5.61795631e-01 3.91887169e+00 2.90386797e+00
     3.07651708e+00 2.63667557e+00 2.22270053e+00 1.86386734e+00
     2.31997756e+00 7.21422038e-01 6.38074689e-01 1.02505593e+00
     1.65700335e+00 8.76664477e-02 9.42404115e-02 3.00486809e+00
     1.47729074e+00 6.46061502e-01 2.51671971e+00 3.57787830e+00
     2.56874422e+00 3.34516589e+00 1.96379100e+00 6.98860249e-01
     2.27520363e+00 1.76679634e+00 1.44066733e+00 8.41935734e+00
     2.28625849e+00 1.45608271e+00 8.84044962e+00 2.49774520e+00
     2.46380700e+00 6.47037579e+00 6.07963124e-01 2.19164581e+00
     3.06675549e-02 2.59076108e-01 3.95164499e-01 8.61334839e-01
     3.12866072e-01 5.22405299e-02 2.03511800e+00 7.97037899e-01
     6.02494502e-01 1.33268662e+00 1.17034848e+00 1.13323122e+00
     1.91854409e-01 3.09720520e+00 4.92183431e-01 2.26551640e+00
     1.54559591e+00 1.11643684e+00 1.00705742e+01 3.41860961e-01
     5.01022974e+00 4.11719075e-02 4.82704970e-01 7.10176130e-01
     1.76454932e-01 8.60237520e+00 1.35510093e+00 7.73241423e-01
     2.78855233e+00 6.93981242e+00 5.94783007e-01 5.26947770e-02
     2.34644165e-01 2.97558716e+00 1.53069020e+00 2.61085125e+00
     4.04770176e-01 2.56372340e+00 5.14443714e-01 2.00849686e-01
     4.15407027e+00 1.66555475e+00 3.13291960e-01 3.08509128e+00
     2.47986094e+00 1.80883200e+00 3.41008317e-01 1.15585837e-01
     1.88820519e+00 3.92514976e+00 5.69348283e-01 5.80424421e+00
     3.70811361e+00 1.19958682e+00 1.19933924e+01 4.97213098e+00
     7.34950491e+00 9.44079310e+00 4.75981723e+00 1.13173916e+00
     1.50595791e+00 9.17615404e-01 7.55076698e+00 2.68032358e-01
     7.62302182e-02 1.32705949e+00 4.16888027e+00 2.92511465e+00
     4.57101486e-01 6.07191703e+00 1.95660124e-01 7.02468204e-01
     5.11599606e+00 6.44329259e+00 3.25223503e+00 1.02325372e+00
     1.65298187e+00 6.85327884e+00 9.39475404e-01 7.33170013e+00
     2.44666762e+00 1.38968441e+00 5.40264945e+00 1.91467515e+00
     8.84257539e-01 4.53580627e+00 5.50779441e+00 6.37996994e-02
     2.19522520e+00 5.21607892e+00 4.27483855e+00 5.55128376e+00
     9.38607414e-01 4.26337608e+00 1.27987296e+00 4.51806034e+00
     5.83963190e-01 9.89743324e-01 5.08913476e-01 4.32839804e+00
     2.35226591e+00 8.18569409e-01 5.34588587e+00 2.35935574e-01
     7.73202737e+00 1.24867825e+00 1.17257897e+00 7.79993405e-01
     2.58058210e+00 7.48208800e-01 1.91548051e+00 3.21531037e-01
     9.23720688e-01 2.71222774e+00 3.66868870e+00 6.90277493e+00
     5.51094680e+00 3.89369962e-01 1.21545636e+00 4.20856685e+00
     6.72604574e-01 2.73460679e+00 3.68560249e-01 2.65023044e-01
     8.67189275e+00 1.43789828e-01 7.27216251e+00 3.74451379e+00
     8.95305476e+00 1.64451479e+00 5.97416963e+00 5.90856681e+00
     3.82911148e+00 3.33219295e+00 1.02950641e+00 1.17944871e+00
     5.07457444e+00 2.82223666e+00 3.15673104e-01 2.98206550e+00
     2.47517437e+00 5.39607866e+00 6.07399882e-02 8.51554041e-02
     4.42149200e-02 2.68215118e+00 6.25337332e+00 2.03016368e+00
     7.66028311e-03 1.18275529e+00 2.31716701e+00 3.13145025e+00
     5.65770266e+00 7.93352370e+00 8.47958841e-01 2.27608292e+00
     2.85816745e+00 5.31443577e-01 8.91275644e-01 9.71924505e-01
     2.79533234e-01 3.69426819e+00 1.41792817e-01 2.15650147e-01
     2.03591302e-01 3.43444063e+00 2.82906649e+00 7.81506317e-01
     7.55745653e-01 8.72411684e-01 3.31274242e+00 5.02855457e+00
     8.86320188e-01 5.85812050e+00 2.01725891e+00 1.27180948e+00
     4.08479112e+00 7.54178981e+00 3.34529543e+00 8.46986753e+00
     1.96085735e+00 7.26396348e-01 8.09991719e-02 7.81715759e+00
     3.12056114e+00 3.46921417e+00 1.14335819e+01 9.20003478e-01
     8.22865096e+00 1.20143664e+00 7.60581580e-01 1.44602815e+00
     2.76167177e+00 1.28642052e+00 1.37000756e-01 1.86452169e+00
     2.26109327e-01 1.89706113e+00 1.36673670e+00 4.47592353e-01
     7.11048059e-01 9.30567215e+00 8.20271172e-01 1.02249821e-01
     2.41115195e+00 3.45941804e+00 7.93801956e+00 7.57061535e+00
     5.81305381e-01 1.54423152e-01 2.24982342e+00 1.25671947e+00
     2.28677532e+00 8.95611737e+00 6.71739470e+00 1.34546592e+00
     6.97633134e+00 9.73355144e-01 3.47420283e+00 3.05734420e+00
     1.85954709e-01 2.40968673e+00 3.36343819e+00 1.91167463e+00
     1.40435699e+00 8.85349697e-01 2.25102820e+00 7.95684744e-01
     2.57134701e+00 3.95960412e-01 9.73445678e-01 1.73981105e+00
     5.83963982e+00 9.32321601e+00 5.34640431e-02 2.12384686e+00
     3.30117148e-01 6.38245560e+00 3.70128887e+00 4.31114565e+00
     7.94137967e+00 1.33780066e+00 5.64072141e-01 3.17961757e+00
     6.07847810e-01 3.22207279e+00 5.87482984e-01 5.45170603e-01
     8.76467437e-01 2.13193432e+00 1.41638360e+00 4.51325785e+00
     8.16399021e-01 4.39340788e-01 6.57719348e-01 1.83973587e+00
     1.00411404e+00 4.86011855e+00 7.94947856e+00 5.16655278e+00
     6.23612448e-01 1.29544792e+00 1.13599863e+00 5.47102295e-01
     6.49989165e+00 1.69116696e+00 3.24544700e+00 9.15880893e-01
     1.36645047e+01 2.11977235e+00 6.22862812e+00 6.42629536e-01
     1.34896308e+00 4.91911705e+00 9.04178150e-01 2.66126653e+00
     2.45155753e+00 1.37277414e+00 4.94029873e-01 2.56325258e+00
     6.17747955e+00 2.74837480e+00 9.14058250e-01 6.10795138e-01
     5.39929498e-01 2.25523718e-01 2.88170749e-01 2.53629478e+00
     1.17457366e+00 3.04602578e+00 2.81821772e-01 5.51472352e+00
     3.37670851e+00 1.08132980e+01 1.38827704e+01 3.49210073e+00
     3.02458422e+00 3.05778957e+00 1.69829810e+00 8.28969522e-02
     1.08865844e+00 7.31398404e+00 1.97137755e+00 2.58159697e+00
     7.63473751e+00 2.86884945e-01 2.61374801e+00 2.74741433e+01
     2.99667320e-01 7.65875392e+00 8.61203613e+00 2.26392194e+00
     2.03849062e+00 3.55651509e-01 5.27653619e+00 7.13218241e-01
     1.36525681e-01 5.38827079e-01 2.46246980e+00 3.20024325e+00
     5.55899950e+00 6.44084709e+00 1.00740951e+01 7.13177020e+00
     8.14108120e+00 5.15087788e+00 6.03368503e-01 1.58992917e+00
     2.35715947e+00 3.93927249e+00 1.31473439e+00 1.39025214e+00
     5.58633011e-02 2.55102759e+00 1.26944663e-01 3.52053423e+00
     4.60302189e+00 3.07026754e-01 3.39292475e+00 5.34946377e+00
     8.64878065e-01 9.57461706e-01 5.91641421e+00 2.15433141e+00
     2.40046297e+00 3.74300431e+00 3.50796789e+00 8.31249753e+00
     2.46576894e+00 2.56868723e+00 4.66949016e+00 3.72004891e+00
     3.12265963e+00 4.99890095e+00 1.28914666e+00 3.79321080e+00
     2.03203457e+00 3.71240646e-01 3.64371426e-01 1.52366025e+00
     1.06791557e+00 6.56178513e-01 1.04658203e+00 1.15683356e+01]
    θ estimado: 2.9121, número de iteraciones: 21
    [1.92693961e+00 1.45581241e-01 8.38618451e-01 9.42213964e+00
     2.94999430e+00 2.06907408e+00 4.82766257e+00 1.03484765e+00
     1.37372327e+00 4.34349781e+00 3.77236144e+00 4.82359852e+00
     6.19836318e-01 5.04296454e+00 1.80513413e+00 1.06616852e+00
     1.88427423e+00 3.21982112e-01 1.51811497e+00 4.15982698e+00
     1.72432386e+00 8.85447424e-01 2.42132213e-01 1.66104741e+00
     2.01558101e+00 1.55464217e+00 2.08451303e-01 1.02185063e+01
     1.01173567e-01 2.22752577e+00 3.54874943e-02 1.19184269e+00
     1.96038710e+00 1.87296321e+00 8.89656378e-01 5.59403667e+00
     7.62895650e+00 5.04560190e+00 4.01405760e-01 5.90418352e-01
     7.91930922e-01 8.10816016e-01 1.01037131e+00 6.43726519e-01
     5.36448609e+00 3.10637816e+00 7.62793927e-01 1.10319960e+01
     2.48882195e-01 1.90347830e-01 5.17080209e-01 5.76297327e-01
     1.48765128e+00 1.76513113e+00 1.82119025e-01 3.54306948e+00
     1.56985130e+00 1.38548407e+00 1.39077040e+00 6.51682414e+00
     1.16514364e+01 7.43678736e-01 3.47226642e+00 6.56483721e-01
     3.43738026e+00 4.16854083e+00 6.03337240e+00 3.65088104e+00
     6.98718475e-01 1.77511200e+00 1.05833588e-01 2.49726030e+00
     2.93157632e+00 1.40185882e+00 2.78261411e-01 7.05974283e+00
     1.09444474e+00 5.38104259e-01 2.01152256e+00 1.97153640e+00
     6.86574589e-01 2.20583845e+00 5.18848480e+00 1.14764644e+00
     2.48915730e+00 1.05331774e+01 1.22884553e+00 4.91456545e-01
     1.62315143e+00 5.36165427e+00 1.50900935e+00 2.00923329e-02
     2.55193894e+00 1.55575212e+00 8.14294923e+00 7.35254267e-01
     9.22328886e-01 1.60573008e+00 3.33840270e-01 3.42793007e+00
     1.10049927e+00 1.67418665e-01 3.26660875e+00 7.08857657e+00
     3.35352763e+00 7.55536235e-01 1.76985444e+00 4.44800766e+00
     2.32679976e+00 4.27494621e+00 3.28614075e+00 8.79910185e-01
     2.37422555e+00 8.49711893e+00 1.78328591e+00 6.02771060e+00
     7.26977290e-01 2.22695501e+00 1.37020826e+00 4.78467245e+00
     8.93048818e+00 4.61724474e+00 8.24593143e+00 5.38377308e+00
     5.32032709e+00 1.09897526e+00 4.77485291e+00 1.33242837e+00
     8.86539304e-01 4.42059084e+00 1.23147946e-01 4.81812499e-01
     8.27005701e-03 6.19435479e+00 6.36194635e+00 7.09590989e+00
     1.69593836e+00 1.58505599e+00 8.03868546e-02 1.53477037e+00
     2.65639825e-02 4.77376124e-01 2.40461221e+00 1.37203438e+00
     2.42680342e+00 1.16956982e+00 3.53992300e+00 5.42126348e+00
     2.21659135e+00 5.31731634e+00 2.37145721e+00 4.32985106e+00
     4.92572952e-01 9.52228386e-03 1.64037860e+00 1.74173624e+00
     2.30252012e+00 6.34209259e+00 8.32775177e-01 9.12365041e-01
     2.83226981e-01 8.71532024e-01 4.73913520e+00 7.94525888e-02
     3.55231463e+00 4.87297895e+00 5.31253897e+00 1.23546984e+00
     4.45272675e-01 1.39283741e+00 8.90049453e-01 2.04538370e+00
     2.77690081e+00 5.18013164e+00 9.21491232e+00 7.15720444e+00
     1.01296220e+01 1.68103296e+00 1.38356344e+00 6.07758676e+00
     1.80739502e+00 5.89495931e+00 5.74965025e+00 2.22907107e-01
     4.75097344e+00 1.08197298e+00 4.98986630e+00 4.56709674e+00
     3.47072058e+00 7.37946255e-01 7.23692669e+00 7.88503129e+00
     2.95868917e+00 2.86584266e-01 1.17998863e+00 1.86893564e+00
     2.95808465e+00 3.08377376e-01 2.29273980e+00 3.32943921e+00
     7.17581927e+00 1.07817162e+00 1.29261558e+01 5.26249087e-01
     6.86053433e+00 8.46302079e+00 6.31090994e-01 1.57939730e+00
     2.31571748e-01 1.28508589e+00 2.27680112e+00 1.26844295e+00
     2.04929799e-01 2.93128730e+00 2.10773496e+00 7.08333342e+00
     8.14101532e-01 1.78870594e+00 4.58221929e-01 4.35954244e-01
     1.35055623e+01 3.94841457e+00 1.86590473e-01 2.34408516e-01
     9.17698983e-02 1.55708569e-01 2.81987958e+00 5.09994657e+00
     8.96243347e-01 2.61881270e+00 3.95110432e+00 2.78806859e-01
     2.04685501e+00 3.32315284e+00 6.77174651e+00 9.85094873e+00
     2.48121600e-01 1.76203424e+00 1.42885846e-01 9.04781007e-01
     1.49082894e+00 5.49235920e+00 3.25514670e+00 1.15297890e+00
     4.86152018e-01 6.29426133e-01 8.63856608e+00 7.92592945e+00
     2.88891091e+00 1.47167639e-02 1.31630735e+00 9.66296703e-02
     1.91232204e+00 9.05869755e-01 7.47789377e+00 2.73438302e+00
     1.19844207e-01 2.75471732e+00 5.28632112e+00 2.31813993e-01
     8.01889216e+00 3.08865065e+00 1.49314259e+00 1.62048161e+00
     3.99719784e+00 1.43261953e+00 1.43270332e+00 5.85905262e+00
     3.95727405e-01 2.36714178e+00 1.10099137e-02 3.18401494e+00
     2.94018855e-01 2.94487754e+00 4.03398416e-01 2.23507868e+00
     4.02079403e+00 4.37993289e+00 5.48046825e+00 7.59601736e-01
     3.97721241e+00 2.09687697e+00 2.83136571e+00 6.06063930e+00
     1.02078693e+00 1.72860041e-01 5.93327989e-01 3.44258026e+00
     4.39601625e+00 5.74128153e+00 6.91585652e-01 4.03132800e+00
     5.13406033e+00 3.88071970e+00 8.85778201e-01 3.83755434e-01
     1.93086860e+00 8.25311083e+00 9.48320713e+00 7.03120724e+00
     4.28993065e+00 4.37811115e+00 1.00557902e+00 2.50565567e+00
     5.39901000e+00 1.58777461e+00 2.46104006e+00 1.61534520e+00
     4.64576278e-01 9.96786034e-01 4.67455642e-01 1.06628191e+00
     1.60504605e+00 4.32228455e-01 6.54749966e+00 6.10660500e-01
     1.14746867e+00 5.89843051e+00 3.87500850e+00 1.15140279e+00
     3.37098690e+00 6.11948396e-01 1.73687785e+00 1.20457553e+00
     5.39653471e+00 2.94288252e+00 5.85056457e+00 1.06792316e+00
     1.74419719e+00 1.98652937e+00 2.27361779e+00 6.43996098e-01
     1.67473871e+00 3.54173685e+00 8.00472278e-01 3.24004184e+00
     6.77330382e+00 4.90203694e+00 2.07830743e+00 1.02295241e+00
     1.50698422e+01 5.93213513e+00 6.83717778e-04 6.89491534e-02
     7.66011535e-01 8.53097109e-01 4.31228074e+00 1.32010962e+00
     8.21893357e-01 2.99447701e+00 7.69250058e-01 1.20502750e+00
     1.66987494e+00 2.76952419e-01 1.35308357e+00 7.08761694e+00
     1.19047263e+00 6.76314932e-01 2.36569230e+00 1.79457412e+00
     3.09542930e+00 3.11324171e+00 2.79452021e+00 4.44861745e+00
     4.78679084e+00 1.62827360e-02 5.73267744e+00 8.62904407e-01
     1.81559254e+00 4.58559999e+00 3.55624452e+00 1.26566913e+00
     3.39620257e+00 1.67338870e+00 8.24173971e-01 5.94085830e+00
     9.35092257e-01 3.78073431e+00 2.10962706e+00 8.26499236e-01
     1.21802934e+00 1.46006987e+00 8.57998553e+00 5.63539335e-01
     9.32999216e+00 1.99825151e+00 7.45066544e+00 1.72335952e+00
     6.50731450e+00 2.47776030e+00 1.24939092e+00 8.46177924e-01
     1.59023349e+00 6.34723889e+00 1.34430320e-01 1.71523146e+00
     3.23696742e+00 3.62942858e-01 1.11679633e+00 1.13034298e+00
     3.43388001e+00 8.35336771e-01 1.11299405e+00 3.65497165e+00
     2.65340807e+00 9.07680823e-01 9.67514880e-01 6.35023882e-01
     7.03005472e+00 1.26480558e+00 1.53459995e+00 2.93711779e+00
     5.33644399e-01 2.03350366e+00 9.40125998e-01 4.34060333e+00
     1.82260645e+00 5.35764178e+00 6.78526610e+00 5.63185353e+00
     3.25313789e+00 4.85578882e+00 1.73644192e+00 2.79801914e+00
     4.23083045e-01 2.05333123e+00 2.82782984e+00 1.39461458e+00
     2.61317299e+00 3.97640491e+00 1.21299217e+00 1.21223157e+00
     6.56125360e+00 2.09346767e-02 2.24731745e+00 1.06529321e+00
     2.11703701e+00 6.43718780e+00 2.23691129e+00 2.36237243e+00
     1.28598412e+00 5.29695323e+00 6.47253118e+00 7.65754845e+00
     2.53537525e+00 1.09449992e+00 3.21852271e-01 5.50423658e+00
     2.20298234e+00 4.00841076e+00 7.60261181e+00 2.93871090e+00
     8.83557559e-01 8.37026754e-01 1.80916803e+00 4.72514272e+00
     7.47239828e-01 8.36083530e-01 9.21898902e+00 3.15335975e-01
     3.85396949e+00 1.03592464e+01 5.29843811e-01 2.55044714e+00
     6.41259732e+00 2.11270291e+00 7.84304861e+00 4.84778243e+00
     5.80516975e-01 3.70633324e+00 3.77507148e+00 3.93477802e-01
     2.79261956e+00 4.54078193e+00 5.20429035e-01 1.59558480e+00
     9.59518584e+00 1.63761657e-01 1.18136085e+00 1.45208539e+00
     1.45701817e+00 1.89174570e+00 1.51491667e+00 6.67796525e+00
     1.19304014e+00 1.21021807e-02 3.41180577e+00 3.07888246e+00
     1.68081034e+00 1.33433147e+00 3.42557818e+00 3.08320406e+00
     4.36346308e-01 8.55283232e+00 2.03473085e+00 3.64776295e+00
     5.38182245e-01 1.73342138e+01 1.47199625e+00 4.05201582e+00
     1.67771184e+00 1.14914342e+01 1.17502971e-02 1.08196019e+00
     2.07679677e-01 4.51053255e+00 2.35482999e+00 5.83460229e+00
     2.41538523e+00 8.97101818e+00 1.66021433e+00 3.00312744e-01
     3.15229530e-01 1.76421067e+00 6.79950295e+00 2.88446107e+00
     5.51105510e-01 1.93043515e+00 5.92619284e+00 1.70941723e-01
     7.23646880e+00 7.39650648e-01 5.93093022e-01 3.05970466e+00
     3.98211649e-01 2.50258612e+00 3.57998121e-02 4.01362141e+00
     4.29250597e+00 1.66793919e+00 1.96054613e+00 3.17929538e+00
     2.48165805e+00 3.37641991e+00 8.75245171e+00 1.07613808e+01
     3.24525706e+00 2.88099938e+00 1.09222932e+00 1.87030542e+00
     3.36454862e-01 3.05096318e+00 1.41303057e+00 1.36147583e-01
     2.19148072e+00 1.03819802e+00 2.09411559e+00 5.10598110e+00
     2.90869764e+00 6.56391139e-01 3.23973853e-01 1.45663301e+00
     1.07494369e+01 5.10002296e+00 4.03633368e+00 2.24231958e+00
     2.20366032e+00 4.70801625e+00 4.11561505e-01 5.22051310e+00
     4.55741147e-01 8.94017590e+00 5.78872722e+00 3.68482156e+00
     1.20438120e+01 9.94641321e-02 6.29786165e-01 3.97149515e+00
     1.90777125e+00 1.02867595e+01 4.81133214e+00 3.72142218e+00
     2.20833362e+00 9.64632334e+00 1.07288681e+00 1.10408722e+00
     1.69805148e+00 1.12646214e+01 2.08152700e+00 6.97529408e+00
     1.90497807e+00 6.53953544e-01 2.39157641e+00 7.41298228e+00
     1.34202245e+00 2.89146169e+00 2.83039421e+00 4.25468869e+00
     4.22334496e-01 4.49748486e+00 2.63848769e+00 2.05014321e+00
     1.51713880e+01 3.77385946e+00 3.79077461e-01 3.07323759e+00
     7.96280535e+00 4.85090815e-01 3.07964476e+00 2.30056717e+00
     1.13146696e+00 3.14406699e-02 2.43709651e+00 5.88817497e-01
     5.30610655e-01 2.01878320e+00 4.82570428e-01 2.28459873e+00
     1.87070782e+00 6.28343548e+00 2.37500513e+00 3.11168614e+00
     7.55577885e-01 1.06138050e+00 7.36192435e-02 2.65302438e-01
     1.26414665e+00 8.66948654e-01 9.76704313e+00 3.02868608e-01
     1.95950160e+00 1.08245096e-01 1.25846336e+00 1.17461688e+00
     1.06514229e+01 3.46673978e+00 1.17771321e-01 2.30958745e+00
     2.71011263e+00 1.96969711e+00 9.12762958e-01 1.91369170e+00
     1.11593587e+01 6.90610018e+00 3.73222314e+00 4.77086906e+00
     1.60791598e+00 5.70651316e+00 5.77969468e-01 6.75168533e-02
     5.97064429e+00 1.65376240e+00 4.48212201e-01 7.79833106e-02
     5.79402085e+00 4.57424987e+00 5.85513313e+00 1.29583079e+00
     2.01632401e-03 3.58746702e+00 1.55480129e+00 8.14478637e-01
     1.43336297e+00 6.40124765e-01 3.39324718e+00 2.06115677e+00
     4.27735878e+00 5.97568113e-01 1.23474298e+00 7.63569646e-01
     6.79127780e+00 6.69168852e-01 2.26411352e+00 9.69820635e+00
     6.07374071e-01 2.17534555e+00 3.79997134e-01 2.57244402e+00
     6.22606208e-01 4.19199896e+00 5.81272044e-01 5.11091784e-01
     2.02456356e+00 7.53146283e-01 9.16925186e+00 3.11870197e+00
     1.01649766e+01 1.05905053e+01 1.12713343e+00 4.62331541e+00
     2.10995037e+00 1.42479492e+00 5.71066544e+00 5.80461962e+00
     3.51778598e+00 9.60208884e-01 1.66880189e+00 1.70960544e+00
     6.60473784e+00 4.58220061e-01 1.81880527e+00 2.19578913e+00
     5.12254354e+00 2.08396966e+00 2.75476069e+00 4.64290907e+00
     2.91178466e+00 4.04862447e+00 3.17327756e+00 1.54291794e+00
     1.62743783e+00 4.06053454e+00 5.21467191e+00 6.88802948e+00
     2.33879453e+00 4.40840234e+00 1.39635261e+00 1.14739627e+00
     4.97970269e+00 4.79872860e-01 1.84070085e+00 1.91660730e+00
     1.57835437e+00 2.11442418e-01 4.48672509e+00 6.89485578e-01
     1.95756427e+00 1.76269289e-01 1.33599946e+01 4.70416080e-01
     2.95561320e-02 2.10410040e+00 3.08802530e+00 4.69294573e-01
     1.74435202e+01 3.74208797e+00 2.86346427e-01 2.16326934e+00
     8.27960472e+00 2.00556513e+00 5.83776728e-01 1.18363424e+00
     5.19713041e+00 3.04162826e+00 9.19755175e-01 3.50962599e-01
     3.60766185e+00 6.96669353e+00 5.06053977e-01 9.60829760e-01
     1.18404662e+01 4.43927249e+00 8.84458555e+00 8.53679034e-01
     3.03413515e+00 3.65772731e+00 2.03110277e+00 5.43801280e+00
     1.35478157e+00 2.42260392e-01 4.50228273e+00 4.15820980e+00
     1.99045611e+00 1.38139088e+00 3.14825476e-01 5.66789012e-01
     2.34681644e-02 1.90170192e+00 1.09045030e-01 1.28847909e+00
     4.52009934e+00 1.90179098e+00 9.03076475e-01 5.65012248e-01
     5.36258980e+00 4.08740116e+00 5.63326267e+00 5.20591706e-01
     1.85695228e+00 1.20100042e+00 7.32825124e+00 3.43860173e+00
     1.35578901e-01 1.66937206e+01 8.99861570e-01 5.61305398e+00
     4.15209169e+00 2.43159920e+00 2.65678731e+00 8.45424303e+00
     9.81789661e+00 8.19012561e+00 1.20710632e-01 7.74962574e-01
     1.26582572e+00 9.56170352e-01 6.31613173e+00 8.71595058e-01
     8.00481640e-02 5.67001333e+00 9.12219532e-02 2.99429094e+00
     3.70993587e+00 3.45231161e+00 1.22730942e+00 7.59122343e+00
     6.97205228e-01 1.09633801e+00 6.23729013e+00 3.22116938e+00]
    θ estimado: 3.0411, número de iteraciones: 14
    [1.76398172e+00 1.28779180e+00 7.18032355e+00 6.08703128e+00
     1.10996223e+01 1.70794519e+00 2.22851395e+00 3.09285099e-01
     1.20421459e+00 9.90737736e+00 1.00079185e+00 6.87606599e-03
     2.86237885e+00 5.50110112e-01 5.90250876e-02 1.67147847e+00
     1.18307944e-01 1.33058592e+00 1.85102654e+00 1.36295687e+01
     1.75383313e+00 3.55535429e+00 4.17659098e+00 8.50587558e+00
     7.36492702e+00 7.45465196e+00 1.06524063e+00 3.07694683e+00
     1.98548866e+00 1.44855654e+00 8.50145184e-01 7.82869703e-01
     1.44245839e+00 1.02697526e+01 2.38418103e+00 2.11847648e+00
     2.12641712e+00 1.52221722e+00 2.86683503e+00 7.06416250e+00
     2.21563516e+00 9.56343026e+00 6.19668064e-01 2.13998747e+00
     5.25043754e-01 1.38654237e+00 2.31376185e+00 6.31221684e-01
     8.42883044e-01 5.04061914e-01 2.76237555e+00 1.27044849e+00
     2.49570083e+00 4.23073401e-02 2.95438341e+00 2.10056333e+00
     6.01507540e+00 2.31200356e+00 4.08036696e+00 2.70285884e+00
     3.89635799e-01 2.58890129e+00 2.00172258e+00 2.93207512e-01
     1.29076486e+00 2.07302514e-01 4.12536917e+00 3.50976854e+00
     1.96836030e+00 4.72926507e-01 3.24925964e+00 2.59551885e-01
     1.33624660e+00 8.80367964e-01 3.91260409e+00 3.64716824e+00
     2.24077358e+00 7.20413059e+00 1.36515152e+00 2.15271441e+00
     1.39418990e+00 3.21336471e-01 3.59049996e+00 1.03747715e+00
     2.66030028e-01 3.91472988e+00 1.13540991e+00 1.72741180e+00
     4.23343066e+00 6.40692640e+00 5.93937695e-01 6.58343306e-01
     6.10742101e+00 4.93525413e+00 5.52041324e+00 8.50386887e-01
     2.87664632e+00 7.52432014e+00 6.03019086e+00 7.11479852e+00
     1.80977440e-01 3.38471501e+00 2.96507034e-01 5.31798497e+00
     3.65129353e-01 9.54509049e-01 1.19615813e-01 3.09240236e+00
     7.19253998e+00 5.07829283e-01 5.50648268e+00 7.71594909e+00
     1.14902027e+00 1.62201050e+00 2.61887639e+00 8.14512305e-01
     9.34599234e-02 9.08511075e-01 5.24300132e-01 1.87015497e+00
     5.63458955e+00 1.79832560e+00 1.26774077e+00 2.38770523e-01
     2.04996361e+00 5.54044676e-01 6.24218339e+00 1.46911716e+00
     3.95021122e+00 3.42143445e+00 1.52992034e+00 4.36480127e+00
     6.19361626e+00 3.22790005e+00 3.08098599e+00 1.77521414e+00
     2.04042080e+01 3.62799084e+00 1.54553092e+00 7.92202830e+00
     3.48992640e+00 9.58035560e-02 8.37860826e+00 2.24450420e+00
     1.21639755e+00 2.19129447e+00 1.20444892e+00 4.96624024e+00
     1.75442542e+00 3.13026668e-01 5.30002389e+00 6.03611477e+00
     1.34738373e+00 1.34462855e+01 7.86806290e-01 1.25716377e+00
     1.24678680e+00 7.39218003e-01 6.54576083e-01 1.07291803e-01
     2.92093234e+00 5.38177667e+00 3.39860021e+00 2.10736894e+00
     1.29693944e+01 1.90599615e+00 1.88396360e+00 2.42268538e+00
     1.54990189e+00 5.40204294e+00 1.44290667e+01 2.85546449e+00
     1.08090325e+00 1.34918956e+00 1.64197152e+00 5.26722255e+00
     2.16818133e+00 1.64372921e+00 3.56892843e+00 1.99651890e+00
     5.15730242e+00 4.71070857e+00 3.19231330e-02 1.22632210e+00
     1.12103805e+00 1.98180861e+00 9.16239511e-01 7.81628493e+00
     9.06677031e+00 8.63422188e-01 2.15315889e+00 1.36584335e+01
     1.29419873e+00 3.08228460e+00 1.10737477e+00 1.71143899e+00
     1.06808967e+01 2.47206460e-02 9.90777987e-01 1.36965013e-01
     7.42838523e+00 3.35851754e+00 7.68806120e+00 8.26605973e+00
     9.29168812e+00 6.30395845e+00 2.27821612e+00 9.11785197e+00
     5.14514160e+00 5.64315212e-01 5.18634154e+00 4.66301754e+00
     1.03865427e+00 1.64965053e+00 4.51786962e+00 6.18643433e-01
     3.32648940e+00 3.05914823e+00 1.00163914e+00 2.44736825e+00
     2.53362368e+00 2.03894210e+00 2.50106048e+00 2.75323436e+00
     1.08675129e+00 1.66392500e+00 6.24947890e-01 5.98572668e+00
     4.14008521e+00 8.00346865e+00 2.59563817e+00 2.67078546e+00
     6.36560478e+00 1.02870056e+00 9.67217239e-02 4.38766824e+00
     4.89065536e+00 4.45190007e-01 1.80459794e+00 8.31934720e+00
     9.05857375e-02 3.96509571e+00 1.87926365e+00 4.73811423e+00
     3.99350223e-02 4.12403389e-01 2.57828614e+00 3.45249810e-01
     9.72653399e-01 9.16591009e-01 5.90609324e+00 1.17948285e+00
     3.78546633e+00 1.36003729e+00 5.82673744e+00 4.85989337e-01
     4.88850006e+00 6.12310847e-01 3.14836995e-01 5.64634846e-03
     3.82450501e-02 1.55704461e+00 4.41640071e-01 7.73554596e-01
     7.88266232e-01 9.97344573e+00 5.49844015e+00 1.19477543e+01
     3.21969821e+00 5.67757227e+00 3.89998926e+00 1.39726440e+00
     5.07449246e-01 6.33413250e-01 5.48312685e+00 4.82541439e-01
     1.79806829e+00 8.38736215e-01 3.69070555e+00 4.53495209e-01
     5.12083857e+00 6.45248117e+00 6.20732462e-01 1.56089715e+01
     2.73450803e+00 3.12742798e-01 8.31556869e-02 1.89545691e+00
     9.21901017e-01 6.70892616e+00 5.97197055e+00 2.33808225e+00
     3.27194490e-01 4.64931200e-01 2.39982176e+00 1.25202010e-02
     1.40790996e+00 5.32414891e+00 2.36177480e+00 4.98138838e+00
     1.21235213e-01 5.11670916e+00 2.32534451e+00 1.62577648e+00
     3.29281298e-01 6.67435594e+00 1.90727699e+00 7.05156074e+00
     7.10701864e+00 3.30090407e-01 2.62310506e+00 9.24591311e-01
     4.40533817e+00 9.60281618e-01 6.19197579e-01 5.68660289e-01
     3.09238844e+00 1.38203035e+00 2.34141625e+00 6.07501518e-01
     5.16867091e-01 1.33962452e+00 1.11513237e+01 1.79653372e+00
     4.87451133e+00 4.41200759e+00 4.44141307e+00 1.05607206e+00
     3.49767989e+00 3.22770366e+00 5.53251485e+00 4.11024207e-01
     6.74294204e+00 3.58889204e-01 8.14005089e-01 1.34261495e+00
     2.51714694e+00 1.99960335e+00 7.77796971e-01 3.27086874e+00
     1.05641856e+00 5.77506392e+00 5.70074609e+00 4.29533621e-01
     5.48315791e-01 9.26746410e-01 2.02145777e+00 2.60335970e+00
     2.63633115e+00 9.57150885e-01 2.41506202e-01 2.15107470e+00
     2.61070523e+00 3.34889426e-01 5.21806668e-01 1.70144006e+00
     2.48807144e+00 8.07499866e-01 1.13066180e+00 5.56238028e-01
     6.42466330e+00 5.71717899e+00 1.60098832e+00 4.00254916e-01
     2.68454723e+00 1.63876854e+00 6.38565785e-02 3.41482680e+00
     2.96035114e-01 1.28005682e+00 6.43112263e+00 5.67460121e+00
     7.63784675e-01 4.24645066e-01 4.11925412e+00 7.00991341e+00
     2.41628583e-01 5.15935916e-01 3.06420406e+00 8.37191155e+00
     2.06408163e+00 1.05398548e+01 5.06363098e+00 3.90092674e+00
     6.40173289e+00 5.87267199e+00 1.61683022e+00 4.82812190e-01
     4.22925995e-01 8.83922984e-02 2.02232572e+00 4.03538281e-01
     2.46282858e+00 1.29563719e-01 4.53696028e+00 4.47015723e+00
     4.07723445e+00 1.50199690e+00 1.29810097e+00 3.44241060e-01
     5.12952687e+00 5.49321785e+00 3.90758509e+00 1.19364733e+00
     2.73901598e+00 4.24030854e-01 6.95234162e+00 9.21846533e-01
     2.59256864e+00 3.05653737e+00 6.33932591e+00 2.27241583e+00
     1.66508757e+00 6.00584191e+00 1.69560189e+00 6.16260493e+00
     2.27367060e+00 3.02951305e+00 9.35205885e+00 4.49801274e+00
     5.12572951e-01 1.25089445e+00 6.20087544e-01 8.69317266e-02
     4.58731259e+00 2.04177837e-01 6.44988643e-01 9.27826564e-01
     6.72671662e-01 1.11670380e+00 6.85456672e-01 6.63702585e-02
     2.93844576e+00 1.25548490e+00 8.36811199e+00 1.31133498e+00
     9.53272741e-01 2.23111519e+00 5.48414192e-01 2.07661490e+00
     1.13991953e+01 9.50451397e-01 8.61070527e-01 2.46636001e+00
     1.44601999e+00 2.76461788e+00 1.09707774e+01 6.22222068e-01
     9.42076392e-03 1.12111835e+00 9.47728396e+00 3.19801419e+00
     3.88067229e+00 3.28365188e+00 1.04007929e+00 1.70921158e-01
     2.06300752e+00 6.13116271e+00 9.12507119e-01 2.16310309e+00
     2.70030316e+00 4.25952997e+00 6.02739610e+00 3.38742492e+00
     3.83239647e+00 5.64119586e+00 5.99784130e+00 7.30030007e+00
     5.47602343e-01 1.71554088e+00 2.82358752e+00 3.50403927e-01
     9.49580879e-01 6.80714531e+00 7.50141709e-02 8.08123012e-03
     1.55564144e+00 8.86722400e-01 2.04972593e+00 1.06459685e+00
     7.07810298e-01 2.43833817e+00 1.16620332e+00 3.82747801e+00
     5.63231399e+00 1.85531605e+00 1.51301847e+01 7.09312946e+00
     3.42368195e-01 9.34398418e+00 6.46878350e+00 1.98308868e+00
     1.23083285e+00 2.87431235e+00 2.41114303e+00 5.41338842e+00
     2.71142728e+00 7.35342081e+00 9.08245691e-01 3.41472317e+00
     5.16845316e+00 3.39482808e+00 2.89115691e-01 3.14552043e+00
     3.41279726e+00 1.80246078e+00 1.83247959e+00 4.18753617e-01
     1.31329625e+00 1.57604845e+01 1.18782665e+00 2.19811513e+00
     2.32367708e+00 4.39148612e+00 1.66252896e-02 6.64858208e+00
     6.27714572e+00 2.85162900e-01 3.98207871e+00 1.85949695e+01
     1.78622359e+00 5.96351049e+00 2.04688765e+00 1.41838760e+01
     1.40246256e+00 4.23010917e-01 2.11179816e+00 7.53835838e-01
     3.89608848e+00 2.86303491e-01 2.44553905e+00 2.94356639e+00
     5.04676532e+00 1.36830677e+00 1.13272122e+00 7.83499624e+00
     5.86915840e+00 1.75739793e+00 5.94922748e+00 1.30902474e-01
     9.47832918e-01 4.28769929e+00 1.61642586e+00 1.94161872e+00
     5.29168325e-01 1.56023563e+00 2.98667742e+00 5.60401302e-01
     1.00572822e+00 7.46820223e+00 3.37956675e+00 7.65725111e-01
     2.56139729e+00 5.16288273e+00 2.46218998e+00 7.00031832e-03
     2.37809275e+00 3.71516083e+00 2.89508247e-01 1.61064268e+00
     1.51668856e+00 2.09983260e+00 5.15829953e+00 7.66752998e+00
     8.55497853e+00 8.90553049e-02 5.09147814e+00 3.58771844e+00
     1.81156674e+00 3.13024116e+00 1.37716387e+00 1.56845185e+01
     1.23343064e+00 9.87414705e+00 4.93100442e+00 3.86364510e+00
     4.97244161e+00 1.32556327e-01 2.06039054e+00 8.69208047e+00
     8.36548380e+00 6.25512111e-01 1.18432023e+01 4.00967942e+00
     3.27978715e+00 3.10296680e+00 1.36575519e-01 1.46985449e+00
     1.06912609e+01 1.05366258e+00 8.33796484e+00 4.24765219e+00
     2.94425732e+00 1.27121859e+00 7.38031631e-01 7.38643372e+00
     4.53585147e+00 4.52532985e+00 1.77453717e+01 2.20209747e+00
     4.41595456e-01 2.05667391e+00 3.65897161e-01 3.70710678e+00
     1.82589550e+00 3.03183002e+00 2.50863952e+00 2.80708835e+00
     2.04890565e-01 1.30972607e+00 9.47435017e-01 2.04203419e+00
     7.49359389e+00 1.86995647e+00 1.99922628e+00 4.93327228e+00
     6.40156883e+00 1.08870333e+00 2.61107049e+00 1.30452523e+00
     2.51097292e+00 3.84242333e-01 2.88527890e+00 1.23003178e+00
     1.80203598e+00 1.41005117e+00 3.07660715e+00 1.25747341e+00
     6.56252906e+00 3.83797114e+00 3.71444637e+00 2.93940510e+00
     3.27579346e+00 7.80201357e+00 9.10819337e+00 3.94545972e+00
     8.64124049e+00 3.76840481e-01 8.42269136e+00 1.75702951e+00
     2.36991056e+00 2.34467294e+00 4.88220310e+00 1.49870154e+00
     3.37756684e+00 2.07046289e+00 2.49797083e+00 5.11995217e-01
     5.91130317e-01 3.82663984e+00 9.91357733e+00 1.04131840e+00
     1.55747932e+00 2.10435784e-01 4.15855561e+00 4.54007332e-01
     2.01609716e+00 1.98783786e-01 2.85175668e-01 2.79729981e+00
     1.25686883e+00 3.50110083e-01 4.75969655e-01 9.34438005e-02
     2.55799695e+00 1.24748697e+01 3.50130692e+00 1.27069962e-01
     5.47592625e-02 2.54224960e-01 5.18785817e-01 4.97436218e-01
     1.84924804e+00 1.79909679e+00 1.80713175e-03 9.41908859e+00
     7.04128079e-01 6.63828518e-01 2.76113028e+00 1.16390137e+00
     3.93365063e-01 6.35642973e-01 2.29925645e+00 1.57980207e+00
     1.33572613e+00 1.84639037e-01 1.11225612e-01 2.00037140e+00
     3.80446881e+00 1.28046817e+00 1.43571333e+00 3.93226598e+00
     5.04164557e-01 5.84700944e+00 2.23193543e+00 6.61687817e+00
     1.56005306e+00 6.94230811e+00 5.81447441e+00 3.92073358e-02
     1.99979495e+00 1.06383991e+00 3.83836117e+00 4.92304775e+00
     6.94838743e-01 2.08648446e-01 5.97564006e+00 1.32389786e+00
     1.71790172e+00 9.99740228e-01 6.75863341e-01 3.57147973e+00
     2.36739381e-01 4.48387743e+00 3.03508413e-01 1.69490852e+00
     6.04461963e-01 5.27256485e+00 4.12831377e-02 8.67582048e+00
     2.59401432e+00 5.60023593e+00 5.48759217e+00 2.67072176e+00
     1.34462712e+01 3.22082385e-01 4.54964661e-01 4.96603368e+00
     1.41691787e+00 2.00086283e-01 3.84800255e+00 2.29160218e+00
     6.47255564e-01 2.94220456e+00 4.24417143e+00 3.37618468e+00
     1.60284385e-01 3.61644657e+00 9.47149527e-01 5.10149834e-01
     1.36307541e+00 1.32313500e+00 5.10381300e+00 4.72138566e-01
     4.73828092e+00 2.71237999e+00 6.33951761e+00 2.42253545e+00
     1.68102289e+00 9.40293999e+00 7.86299625e+00 3.01989751e+00
     1.79413639e+00 6.03359336e+00 2.36258057e+00 2.55739111e+00
     3.67095924e+00 9.59356225e-01 4.22140266e+00 3.39460915e-01
     3.48791075e+00 2.75388898e+00 5.48229046e+00 7.11314008e+00
     4.17453297e+00 2.25122605e-01 6.35772351e+00 2.57496770e+00
     2.11828957e-01 3.43049703e+00 4.00858951e-01 5.40907210e+00
     2.17979573e+00 1.22693528e+00 5.32989115e-02 2.87369410e+00
     3.81883792e-01 3.36018837e+00 3.75832564e+00 9.85855399e-01
     6.46208355e+00 8.36657740e-01 5.57257763e-01 8.24848525e+00
     1.15632108e+00 2.97348846e+00 1.43223257e+00 4.93631717e+00
     8.71477210e-01 9.03453705e-01 3.67772382e+00 2.95560711e+00
     2.26617642e+00 1.52812616e+00 3.29653773e+00 1.45012280e+00
     2.50629554e+00 1.09074509e+01 2.07554242e-01 7.25759346e+00]
    θ estimado: 3.0684, número de iteraciones: 8
    [2.34491868e+00 1.50714347e+01 2.08571259e+00 1.15891619e+00
     2.54493943e+00 5.57075332e+00 1.16404009e+01 2.52767127e-01
     1.27815588e+00 3.17581110e-02 2.76880158e+00 6.38289547e-01
     4.92568154e-01 1.86910774e+00 1.13382582e+00 2.37655547e+00
     1.34832123e+00 2.03029900e+00 2.93835852e+00 2.14418672e+00
     2.19834236e+00 5.71736212e+00 1.85233638e-02 4.86355533e+00
     1.64613985e+00 2.16320118e+00 2.37486949e+00 1.76159401e+00
     3.12028380e+00 9.91495823e-01 4.74462669e+00 2.01035596e+00
     7.94964881e-01 1.97223980e+01 8.78432387e+00 1.72275976e+00
     4.42287775e+00 1.13381178e+00 5.17601653e+00 7.86833125e+00
     3.50950978e+00 7.48552044e-01 6.78413722e+00 6.25953104e-01
     1.81900801e+00 3.93095987e+00 7.33075698e-01 1.22406524e+00
     2.40935905e+00 3.28078751e+00 2.32104731e+00 3.37810347e+00
     1.02827928e+00 5.72358794e+00 9.08615919e-01 8.48162749e+00
     2.17145467e+00 1.69407089e+00 2.54617776e+00 4.32203264e+00
     3.30382160e+00 3.73992257e+00 1.34728857e+00 2.22091759e+00
     8.01249561e-01 1.82380394e+00 2.24109679e+00 3.78445014e+00
     2.51912409e+00 4.57107490e+00 9.28338902e-02 1.05150211e+00
     4.46295437e+00 2.99901869e+00 2.49268837e+00 2.77139412e+00
     2.46811128e+00 6.19805434e-02 1.63438758e+00 2.60338236e+00
     2.88321843e+00 2.30660996e+00 6.70420326e+00 1.89671197e+00
     4.74860622e+00 5.90005845e-01 5.84159759e-01 5.34503556e+00
     5.44876262e+00 6.24135478e+00 2.86857740e+00 1.84837735e+00
     2.50091905e+00 1.32962856e+01 8.57014431e+00 1.90638699e-01
     1.78607173e+00 5.67698368e+00 4.94218724e+00 6.14963258e+00
     5.06350943e+00 3.10658431e+00 6.81411883e+00 1.80622616e+00
     5.95545969e+00 9.37843265e-02 1.28207509e-01 9.70404542e+00
     1.09168222e+00 1.44486024e+00 4.03244518e+00 1.04569128e+00
     7.37802779e+00 9.32888841e-01 2.94769091e+00 9.95246000e-01
     1.56498295e+00 3.05224156e+00 1.67141783e+00 5.55755065e-01
     2.33320574e+00 2.43074399e+00 5.64247997e+00 3.08714209e+00
     1.27790419e+00 1.82664478e+00 6.90964682e-01 1.75175739e+00
     1.90325012e-01 2.25160990e+00 2.81266237e-01 4.42137032e+00
     1.82609568e+00 2.71657614e+00 2.84399676e-01 1.30075107e+00
     2.91646384e-01 7.58094786e-01 1.34381062e+00 2.31404153e+00
     2.56403747e+00 2.50262501e+00 1.48794647e+00 1.53577558e+00
     1.25009836e+00 2.22683578e+00 2.28321535e+00 4.19005858e+00
     1.93425990e+00 3.56936085e-01 5.00435048e+00 6.36609117e+00
     5.34627166e+00 2.77741741e+00 3.68086983e+00 7.07334354e+00
     5.77873991e+00 2.38458065e-01 2.89529824e+00 4.17288916e+00
     1.90605800e-01 3.71818134e+00 3.16155457e+00 1.72223277e+00
     7.94663826e+00 5.26881263e-01 6.86196897e+00 3.46222046e-01
     3.89583877e-01 3.76386986e+00 1.85520090e-01 9.77297786e-01
     2.77001257e+00 4.17480470e-01 3.65807259e-03 6.48962797e-01
     7.33332101e+00 1.50046879e+00 4.14436128e+00 1.74101841e-01
     6.31554190e+00 1.26140085e+01 1.26241192e+00 8.81330173e-01
     1.58532982e-01 7.57126659e-01 1.25450160e+00 4.74455969e+00
     6.12917124e+00 8.20533390e+00 8.85806317e-01 1.63172830e+00
     5.43543832e+00 1.23118708e+00 2.87871456e+00 1.11842165e+01
     4.12799514e+00 1.11635521e+01 8.21237328e+00 3.83309652e+00
     5.62041779e+00 1.47756920e-01 2.15076818e+00 3.03763759e+00
     4.61432045e-01 8.01222628e+00 2.39960815e+00 3.02243491e-01
     5.75510139e+00 5.20155401e+00 5.83069122e-01 6.02047433e-01
     9.52103864e-01 1.22928279e+00 8.35833549e+00 2.87230990e+00
     7.08357968e+00 4.09793031e+00 5.10243334e+00 1.87648717e+00
     5.11379274e+00 1.14225901e+00 4.18662998e-02 2.62399144e+00
     2.45038317e-01 5.41743453e+00 9.14435391e-01 1.11915623e+00
     1.99731586e-01 2.35276267e+00 3.35556982e+00 7.32393086e+00
     2.96684076e+00 1.44525574e+00 1.51567721e+00 7.09276082e-01
     6.35735564e-02 5.61483713e-01 1.70309872e+00 2.21749674e+01
     1.56003548e-01 2.19456762e-01 3.62869287e+00 1.12416270e+00
     2.22222935e+00 3.08494963e+00 7.24641002e+00 4.73958184e+00
     5.93090114e+00 3.53340353e+00 7.07002193e+00 2.18655063e+00
     2.99998411e-01 3.83025069e+00 1.15500072e+01 4.66062919e+00
     5.79322901e+00 2.95343947e+00 1.61473667e+00 6.89666725e+00
     2.65407909e+00 5.46663131e+00 8.18833171e+00 1.69008769e-01
     1.56498078e+00 2.70301702e+00 1.14717559e+00 7.21953300e+00
     2.97999190e+00 4.36752493e+00 3.15982755e-02 1.59272006e+00
     8.41482128e+00 2.58774061e-01 1.71139994e+00 9.86533654e-01
     7.76524922e+00 2.87381141e+00 2.88760365e+00 2.32629505e+00
     1.83210029e-01 4.73772853e-01 1.50211103e+00 2.70551898e-01
     7.87415531e-01 3.30766777e+00 6.16249939e+00 4.67263675e+00
     7.74459431e-01 4.70277089e+00 4.42068944e+00 4.80957138e+00
     3.72499122e+00 9.51373155e-01 6.91638102e+00 7.23047284e-01
     5.11496448e+00 1.35885200e-01 6.90606962e-01 1.55653714e-02
     1.39991747e+00 7.01629873e-01 3.67516796e+00 2.13141233e-01
     7.19162386e-01 7.99616071e-01 3.07800022e+00 3.52870953e+00
     1.62796253e-01 1.80148949e+01 2.95511462e+00 7.75134192e-01
     1.46358623e-02 2.19308084e+00 7.00899511e-01 1.73945253e-01
     1.90461017e+00 3.31637786e+00 3.73031588e+00 2.98659840e+00
     4.57217281e-02 1.16949628e+00 1.99756492e+00 3.55958102e+00
     3.90406075e+00 6.86765290e+00 1.94871946e-03 7.16214673e-01
     5.22645740e+00 3.94770055e+00 3.43953031e+00 6.86611254e+00
     3.46693720e+00 1.18301108e+00 5.21921178e+00 3.78134920e+00
     1.86071053e+00 1.10403939e+00 7.45167625e+00 3.71817166e+00
     1.62758027e+00 7.44463076e-01 3.55311449e-01 1.09619192e+01
     7.30587454e-01 5.82302814e-01 4.77112837e+00 6.64563878e-01
     2.14567177e+00 2.38310106e+00 1.71432606e+00 4.94562660e-01
     8.02889232e+00 3.47904095e+00 1.28347441e-01 1.16584299e+01
     6.98470538e-01 9.72512018e-01 3.27509188e-01 5.67807468e-01
     3.63163344e+00 3.47057588e-01 7.87530976e+00 3.97355147e+00
     4.50468671e+00 1.11926473e+01 1.88578021e-01 2.63945736e+00
     5.98820189e+00 2.00000707e-01 6.65354568e+00 3.83821726e+00
     3.43955668e+00 2.16635213e+00 1.07218089e+00 7.08243905e-01
     1.84644407e+00 1.39036390e+00 2.45670399e+00 2.67753566e+00
     5.74777134e-01 3.30954979e+00 5.43088929e+00 2.91827232e-01
     1.12512203e+01 4.78791105e+00 8.71415041e+00 3.48831661e+00
     5.94292288e-01 2.28336310e+01 5.42804368e+00 4.86030734e+00
     1.60717761e+00 3.98241866e+00 4.00784237e+00 5.55767834e+00
     5.40751765e+00 7.30810527e+00 1.76962272e+01 2.85669567e+00
     5.84906053e+00 5.64617170e+00 1.31099229e+00 9.56920078e-01
     1.83373481e+00 2.79022771e+00 1.03766012e+00 9.75051339e-01
     1.53770862e+00 2.41472724e-01 2.88634512e+00 4.96573825e+00
     1.96380098e+00 1.57335409e+00 6.43507057e-01 8.29526078e+00
     7.15603852e+00 1.82706723e+00 3.58519639e+00 9.04821658e-01
     1.99524009e+00 3.49476319e+00 4.52658084e+00 1.24988065e+00
     1.08805358e-01 4.95777507e-01 5.00592911e+00 2.73067075e+00
     1.05243452e+00 4.53495483e-01 1.34304197e+00 1.92771307e-01
     5.92054924e+00 1.91321214e+00 2.37397386e+00 9.67288978e-01
     3.33527454e+00 1.56967109e+00 5.85860997e+00 7.67714902e+00
     2.75725996e+00 1.07568211e+00 1.23619711e-01 6.60621622e+00
     1.56963885e+00 2.48341178e+00 5.60205973e+00 6.04016537e+00
     3.45981181e+00 1.62589402e-01 9.46162148e+00 3.07923784e+00
     4.88174301e-01 6.58648224e-01 6.65535236e+00 1.56626101e+00
     1.72019500e+00 2.48262423e-01 5.82459293e-01 8.04056598e-01
     1.66798006e+00 3.80647336e+00 3.00163429e+00 3.54537455e+00
     5.84764217e-01 3.37555178e+00 3.31382690e+00 2.10156495e+00
     7.18094155e-01 4.06330178e-01 1.35266390e+00 5.08574823e+00
     3.09149317e+00 2.89408546e+00 5.54244108e-01 1.46679947e+00
     5.32013976e-01 1.64295160e+00 2.60016221e+00 6.08139085e+00
     2.33562406e+00 3.32826918e-01 5.16131653e+00 1.02807680e+00
     1.83650522e+00 1.92925462e-01 5.06506677e-01 4.37732745e-02
     4.16161248e-01 1.93290686e+00 2.31170175e+00 4.86991316e+00
     8.46004683e+00 8.13656229e-01 2.88711483e+00 9.86896137e-01
     6.00632872e-02 5.69100145e-01 1.94800992e+00 1.28723873e-01
     1.76240741e+00 7.15577265e+00 6.12833725e+00 4.20450711e+00
     7.41753009e-01 5.34438473e+00 2.53971306e+00 3.86062526e+00
     5.54622642e-01 4.35018754e+00 1.39357570e+00 1.78652634e-01
     3.99968594e+00 2.96363012e+00 4.24713491e+00 3.58938877e+00
     2.23941310e+00 2.07237450e-01 2.27314523e+00 6.04979688e+00
     4.00216964e-01 7.48867008e-01 5.17280415e+00 1.50807617e+00
     1.91398319e+00 5.03656388e-01 1.03844294e+01 2.43381137e+00
     1.85665097e+00 4.49390982e-02 1.55265964e+00 1.59858960e+00
     2.02848541e+00 2.65373210e+00 8.56203322e-01 2.17187048e-02
     1.56527733e+00 5.91567418e+00 4.46212892e-01 2.24925291e+00
     3.29292721e+00 8.16156567e+00 6.29006499e-01 1.18751344e+01
     5.42029948e-01 1.19308575e+00 2.81192255e+00 1.05697376e+01
     8.31907010e+00 7.83109699e+00 5.38040387e-01 2.34916609e+00
     2.19331347e+00 8.98350511e-01 7.45148170e+00 2.32351699e-01
     1.21493467e+00 2.68962387e-01 2.42112378e+00 2.01653684e+00
     3.21415512e+00 1.81423781e+00 9.85996667e-01 5.90781595e-01
     9.87059401e-01 1.09346964e+01 5.35355316e-01 3.30771697e-01
     6.08658219e-01 4.08842423e-01 9.54689535e+00 3.63990869e+00
     1.52501670e+00 2.10774516e+00 7.67773847e+00 5.70017904e-01
     7.59158251e-01 2.81040684e+00 2.11802726e+00 5.16179559e+00
     1.19528719e+00 1.95810854e+00 2.13058621e+00 3.73841573e+00
     2.59427540e+00 6.98181584e-01 2.00046882e-01 1.90363276e+00
     1.10718308e+01 2.32717415e+00 2.53669142e+00 2.75427204e-01
     4.09535835e+00 6.00331167e-02 6.94374221e-01 8.50789205e-01
     9.76657232e-01 6.54151194e-01 3.68093454e-01 3.37971901e+00
     4.08964268e-02 1.06054938e+00 2.48620253e-01 8.40193400e-01
     2.62244544e+00 3.27235492e+00 1.24529491e+00 4.98897270e-01
     5.23500388e+00 9.38718530e-01 1.05961142e+00 6.73811559e+00
     1.55511643e+00 2.87659707e+00 1.98272107e+00 1.98845583e+00
     8.29934427e+00 3.83553724e-01 8.96181146e-01 1.54938512e-01
     5.88629274e+00 1.30341349e+00 6.96828200e+00 1.14971407e+01
     1.03860087e+01 6.29497560e+00 7.38893809e+00 2.82967586e+00
     1.41316466e-01 2.83801685e+00 1.37674439e+00 4.57454321e-01
     5.59010342e-01 4.49463162e+00 3.82408069e-01 1.70448255e-01
     2.11370833e+00 4.96976803e-01 9.37918153e+00 3.09501139e+01
     6.77481854e-01 1.81722298e+00 3.48210176e+00 2.17163752e+00
     1.46069306e+00 3.31038673e+00 4.59128249e+00 1.41131660e+00
     1.72516651e+00 1.11730826e+01 9.98138046e+00 1.70722144e-01
     6.55197354e-01 7.31247759e-01 5.21468219e+00 1.50464854e+00
     4.84978394e+00 3.73390391e+00 2.37660875e+00 7.82553719e+00
     1.47654816e+00 7.23768579e-01 1.09717028e+01 1.62106538e+00
     6.70599221e-01 5.49898385e+00 1.63486188e+00 1.18447628e+00
     8.97421119e+00 8.52446234e+00 1.83661446e+00 6.56596218e-01
     3.10686090e+00 1.49280659e+01 2.02169165e+00 1.05444319e+00
     4.58989340e-01 7.29962195e+00 1.03425229e+00 3.51271484e-01
     5.47364982e-01 7.24022180e+00 1.93003339e+00 1.81549225e+00
     8.27283247e-01 1.11010986e+00 3.76752913e-02 9.81219975e+00
     9.05020260e-01 6.67480438e+00 9.44688959e-01 2.72254129e+00
     2.83938366e+00 5.87689389e+00 6.30449649e+00 4.54427693e+00
     1.08595910e+01 1.48997561e+00 3.16341468e-01 2.36342937e-01
     7.25660832e-01 8.81152349e+00 1.47223945e+00 1.10613904e+00
     3.54905765e-03 3.89604721e-01 2.52414918e-01 4.24298179e+00
     7.81915756e+00 4.97011218e+00 3.20449388e-01 2.09063264e+00
     4.04409911e+00 4.04282361e-01 2.93029612e+00 6.96703232e+00
     4.45371045e-01 2.11495768e+00 4.83211522e-01 2.75550880e+00
     3.48721139e+00 1.31654424e+01 5.21334542e-01 3.44903442e+00
     1.89299484e-02 2.79589278e+00 1.25534692e+00 2.38262771e+01
     3.11044909e+00 1.15606620e+00 1.69690049e+00 3.48301741e+00
     4.21959960e-01 6.66928722e-01 8.40477144e-02 1.17977485e+00
     1.16668402e+00 8.18774839e-01 2.81628824e+00 2.11858396e+00
     7.81022140e+00 2.39789742e-01 1.93666176e-01 1.07655227e+00
     7.67598868e-01 1.48986134e+00 6.12011607e-02 5.02498193e+00
     1.08859209e+01 4.27505999e-01 2.34350753e-01 2.39960057e-01
     1.06993583e+00 8.71152414e+00 7.94103847e-01 3.44607516e-01
     1.94671993e+00 1.06687231e+00 9.12312081e-01 1.93973310e+00
     7.73145513e+00 6.27006543e+00 4.13602426e+00 2.22358691e-01
     2.57234636e+00 5.79441615e+00 7.24552606e+00 1.66208229e+01
     4.39013208e-01 2.48063347e+00 7.11469829e-01 5.36612722e+00
     4.43746741e+00 5.63260199e+00 6.31587406e+00 3.69988203e+00
     1.83741449e+00 1.25921768e+00 4.36120997e+00 4.03812440e+00
     9.41699023e-01 6.45462405e+00 5.01571080e+00 7.35414972e+00
     1.02344277e+00 6.61446909e-01 1.05693510e+01 5.44534301e-01
     2.22943069e+00 1.58612826e+00 8.80309765e-01 1.37402174e+00
     1.32865350e+00 3.17110102e+00 6.76759959e+00 4.27015659e+00
     8.52305340e+00 1.31031353e+00 1.55145999e+00 1.12622864e+01]
    θ estimado: 3.1134, número de iteraciones: 5
    [8.77002768e-01 5.36625372e+00 6.77583190e+00 7.71342508e-01
     1.40235493e+00 6.60099624e+00 9.13217663e-01 7.37104161e+00
     4.83622616e+00 4.60140218e-01 9.29606239e+00 8.57017500e+00
     1.76715218e-01 3.60978815e+00 4.82208708e-01 5.46232709e+00
     1.00522475e+00 6.67658215e-01 4.02618844e+00 4.51478541e-01
     4.70205866e-01 2.66917269e+00 2.54750108e+00 9.66560414e-01
     2.93668218e+00 1.71620859e+00 1.80261636e+00 8.64969276e-02
     3.38553023e+00 9.74480004e-02 1.97656704e+00 4.24125980e+00
     2.29094149e+00 1.12393560e+00 2.55597332e+00 6.87606407e+00
     4.00985648e-01 4.24925090e+00 3.69381666e+00 3.28254593e+00
     6.89513489e-01 9.63690098e-01 4.39832246e+00 6.57257947e+00
     1.71967128e+00 2.46468443e-01 3.74524722e-01 7.26291880e+00
     3.97131881e-01 3.18422942e+00 2.26646796e-01 7.52343372e+00
     5.04463951e+00 7.65140168e-01 8.25879632e+00 2.98703996e+00
     2.69397971e+00 4.69903034e+00 1.13064613e+00 1.27110187e+00
     2.07146210e+00 5.86216395e+00 3.28028948e+00 3.27007344e+00
     1.97952079e+00 1.53651545e-01 6.53661551e+00 1.22314157e+00
     4.88711037e-01 4.92181002e+00 1.80191399e+00 5.95590323e-01
     9.84021164e-02 2.84788379e+00 8.41223610e+00 6.92610643e+00
     1.52556253e+01 4.96217818e+00 1.45922897e+00 2.42488754e+00
     1.35263165e+00 3.78501651e+00 4.89630812e+00 2.51358036e+00
     5.21374468e-01 5.54577093e+00 8.12287955e+00 1.15573839e+00
     1.22191372e+00 2.95648423e-01 2.05129542e+00 3.82224318e+00
     7.64320754e+00 3.68302181e+00 8.12472274e+00 2.18661618e+00
     8.03122619e+00 3.12518788e+00 1.38953252e+00 6.19170818e+00
     1.37488723e+00 4.10699422e+00 2.40599645e+00 6.03657324e-01
     2.70439787e+00 1.93534025e+00 3.66692529e+00 2.32904722e-01
     6.88716132e+00 1.78462480e+01 8.07705632e-01 2.48842923e+00
     1.02002174e+00 2.80238693e+00 7.39180072e+00 3.03944272e-01
     2.85920464e+00 8.44079927e-01 2.27527388e-01 1.01558932e+00
     7.48119474e-01 4.92856388e+00 1.17305685e+00 2.45162549e+00
     4.54621970e+00 1.50941237e+00 2.31879495e+00 8.80670843e-01
     7.44703828e-01 6.55543435e-01 2.95356642e+00 6.86322077e+00
     4.34278106e-01 2.82029810e+00 2.47205138e-01 2.96343732e+00
     3.23386492e+00 9.81549431e-01 5.64880958e+00 5.70112614e+00
     1.17449895e+01 4.84939632e-01 1.09933509e+00 8.95920794e+00
     5.79370615e-01 3.95940178e+00 2.06512234e+00 1.65727926e+00
     1.76274863e+00 3.98642404e+00 5.78032610e-01 2.11852988e-01
     2.14911979e+00 1.44956185e+00 2.53903069e+00 4.85412185e+00
     3.39926298e+00 9.46744702e-01 6.98733100e-01 2.33603996e+00
     1.07695973e+00 1.94032077e-01 9.75095195e-01 1.16359548e+00
     4.95790139e+00 3.02876876e+00 6.21478942e-01 5.48047010e+00
     1.87005174e+00 6.61146817e-01 1.11192808e+00 2.59119177e+00
     5.39362418e-01 5.76308004e+00 3.11672884e-01 4.29018677e+00
     1.35683033e+00 3.13903608e+00 1.04308110e+00 4.61822560e-02
     1.75908146e+00 4.76554732e-01 3.30180661e+00 4.56039254e+00
     1.83755407e+00 2.01892406e+00 3.22305708e-01 4.02777166e+00
     4.93542055e+00 2.65810150e-01 4.45476627e+00 2.06749990e+00
     5.10283703e-01 8.84772795e-01 1.00799565e+01 2.31388173e+00
     2.40921533e-01 1.08991303e+01 7.11034794e-01 4.63573929e+00
     2.42032027e+00 4.13513482e-01 2.71677853e+00 6.67530784e-02
     2.10230712e+01 5.83408520e-01 3.65901867e+00 2.22145746e+00
     9.16917413e+00 1.41094582e+00 1.17680067e+00 7.21621368e-01
     3.32100816e+00 6.47167442e+00 5.81846089e+00 2.53844141e+00
     2.62300972e+00 4.35993289e+00 7.84100253e+00 1.94999028e+00
     1.11005503e+00 3.40754755e+00 8.09280472e+00 5.88114555e+00
     3.21533438e+00 2.22274417e+00 1.12681986e-01 4.04046998e+00
     4.51928764e+00 6.28132540e-01 2.16677757e-01 2.71376902e+00
     6.39574587e+00 2.60111887e-01 1.61941771e-01 6.79717489e-01
     1.14658705e+00 9.22585804e+00 2.50344198e+00 2.54241194e+00
     2.40179835e-01 3.92401314e+00 4.43359699e-02 1.05754720e-01
     3.34124571e+00 4.05848295e+00 2.59055875e+00 1.22199905e+00
     7.27215621e+00 7.19189923e-01 2.73440537e+00 2.67848043e+00
     6.34238843e-01 1.10820806e+00 6.70729678e+00 1.48483476e+00
     2.05590141e+00 4.10371730e+00 4.41890165e+00 3.54570957e+00
     9.23619588e-01 1.71585408e-01 1.09599858e-01 4.03129913e+00
     2.98491950e-01 6.28386412e-01 4.33069684e-02 9.68981863e-01
     5.71642202e-01 4.00753455e+00 4.07234919e+00 2.83910333e+00
     1.14663614e+00 2.22172180e+00 6.73604098e+00 2.05126155e+00
     7.72907322e-01 3.94017078e+00 1.18079841e+00 4.66835784e-01
     4.82838445e+00 1.10713587e+01 1.39009076e+00 3.23265328e+00
     4.77199646e-01 2.91927018e+00 3.18258214e+00 9.22685152e+00
     4.56701330e+00 1.03468052e-01 2.80679981e+00 7.21213622e+00
     8.88665085e+00 4.11164726e-01 5.36914829e+00 6.95304507e+00
     5.69047711e+00 2.62848504e+00 2.24453627e+00 1.14768631e+00
     8.83024130e+00 4.56878302e+00 5.35557842e+00 4.74732369e+00
     2.23219038e+00 6.45334311e+00 3.33037433e+00 4.44878124e+00
     7.22563068e-01 1.29286388e+00 1.92385167e+00 1.06729073e+00
     3.27037737e+00 6.19660942e-01 1.24045699e+01 1.06354049e+00
     2.60339815e+00 3.04855514e-02 2.21080457e+00 5.02910132e-01
     2.04760848e-01 4.90597385e+00 9.62166229e-01 1.84858650e+00
     2.01929544e+00 5.16187643e+00 1.10209562e+00 4.20116412e-01
     2.77924515e+00 1.41738789e+00 3.77157940e+00 8.05305285e+00
     6.47353563e+00 4.93553156e-01 6.26095625e-01 5.83088633e+00
     5.82798197e+00 1.15943727e+01 7.15800143e+00 1.01606342e+00
     6.89892569e+00 8.10816226e+00 3.77821897e+00 1.06560468e+00
     7.66106087e-01 1.94240276e+00 3.46980568e+00 5.77487363e-01
     3.10236536e+00 7.17589050e+00 5.44853723e+00 3.34462531e-02
     9.33317129e-02 5.39421765e-01 2.71239045e+00 6.15489326e-01
     5.89137750e-01 1.56628931e+00 3.04472706e-01 9.81687888e-01
     2.60551115e+00 1.49179979e+00 9.14877751e-01 6.04300385e+00
     3.63106859e-01 4.21284771e+00 8.52957955e+00 1.34042210e+01
     1.78555452e+00 1.03809412e+01 1.24660727e+00 1.33553116e+01
     7.25731808e-03 1.20283380e+01 2.33703265e+00 7.99752770e-01
     6.77739729e-01 3.64148846e+00 2.36562560e+00 1.02637175e+00
     3.44870159e+00 6.03052411e-01 1.80431927e+00 6.17440955e-01
     1.36433244e+00 2.59917937e-01 5.07934321e-01 5.99093782e-01
     4.48821786e-01 2.71771744e+00 1.18221105e+00 1.48849098e-01
     4.63441556e-02 4.15045205e+00 9.94977807e+00 8.02801158e-01
     3.63854768e+00 1.72914097e+00 1.69767193e+01 1.98315954e+00
     2.36567887e+00 2.62017973e+00 3.68284569e+00 5.85539868e+00
     3.24029443e+00 7.99416198e-01 2.42192396e+00 4.85590925e+00
     9.19102896e-01 3.05959897e+00 1.65608947e+00 1.87470355e-01
     1.43105476e+00 5.16433275e-01 3.35331137e+00 1.33378462e+00
     1.85302052e+00 1.06496391e+01 1.62401887e+00 7.94902775e-02
     1.92898813e+00 5.22077204e-01 1.15601681e-01 1.02288247e+01
     2.90678122e+00 3.12835730e-01 1.15716608e+00 3.83556456e+00
     4.37396346e+00 4.09428745e+00 3.80769127e+00 1.32050503e+00
     1.85162042e+01 2.06151192e+00 8.76650349e-01 1.56265229e+00
     2.06187524e+00 4.59954310e+00 8.75875512e+00 5.56734168e-01
     2.65050089e+00 2.77786826e+00 7.24204652e-01 1.10309869e+00
     8.46529723e+00 6.35045456e-01 4.22957620e+00 6.24214618e+00
     4.01411423e+00 5.23498951e+00 2.06668214e+00 5.39895889e+00
     6.31142770e-01 9.43677299e-01 4.14021164e+00 2.37132998e+00
     1.69540853e+00 1.85889612e+00 3.25637065e+00 2.69940924e+00
     6.10466456e-01 8.20412549e-01 2.49568795e+00 2.74112542e+00
     2.47298889e-02 3.42347969e+00 5.46980271e+00 1.26793732e+00
     3.99210798e-02 5.56226820e+00 1.05105994e+00 8.47927380e+00
     8.37895441e-01 7.77254061e+00 2.96386188e+00 2.33370397e+00
     1.21314349e+00 1.65356765e+00 5.54569975e+00 7.07047667e+00
     6.07015828e+00 1.53249906e+00 5.03467646e-01 3.77266203e+00
     1.18459614e+01 1.54925032e+00 7.71669250e+00 8.85040928e-01
     5.13023620e-01 9.62222195e+00 8.06035955e+00 2.24480617e+00
     5.73318372e-01 4.92219822e+00 5.05037774e+00 3.31536927e+00
     1.43780170e+00 7.37888913e+00 3.12284837e+00 4.89918921e+00
     9.89130311e-02 2.96782852e+00 2.85492490e+00 3.27119295e-01
     3.26124262e-01 8.84449515e+00 9.87877587e-01 5.28124081e-01
     1.85984901e+00 2.18928346e+00 1.60415410e+00 5.67322842e+00
     1.04045848e-01 3.08748956e+00 8.85472072e-01 8.38393685e-01
     2.68336379e+00 1.79099311e-01 1.12463250e+00 6.68719841e+00
     2.16125767e+00 2.54362092e+00 2.91825298e+00 1.51703335e+00
     9.49514261e+00 3.46917949e+00 1.99827638e+00 5.75096807e+00
     4.52356814e+00 8.13874208e-01 1.74935167e+00 1.10483696e+00
     6.77164537e+00 1.76968570e+00 1.64099479e+00 6.19517146e+00
     7.98472265e+00 2.77298513e+00 3.43749933e+00 5.43989934e+00
     3.00428329e-01 8.97226782e-02 1.38385193e-01 9.52863307e-02
     8.69977828e+00 9.01301692e+00 6.80873930e-01 1.62562332e+00
     1.21562134e+00 9.72252785e-01 2.77294896e+00 1.23919644e+00
     3.84357340e+00 4.60004708e+00 1.04986316e+00 9.60871890e-01
     4.91588667e+00 1.21039783e-02 9.71911334e-01 2.63431066e-01
     2.76294961e+00 3.70681877e+00 2.47914494e-01 5.14565981e-01
     7.65390241e-02 8.11869274e-01 2.19521695e+00 2.46281702e-01
     9.93213736e-01 3.06169801e+00 1.59435641e+00 2.75076267e-02
     1.32840236e+00 1.79072893e-01 8.59756330e+00 2.05367587e+00
     2.39524239e+00 4.80153487e+00 7.59550352e+00 1.96991205e+00
     2.10525135e+00 3.16577235e+00 2.20383815e+00 2.40711301e+00
     1.72801101e+00 2.82848852e+00 3.02772499e+00 5.50982319e+00
     4.46767378e+00 4.62793035e-01 7.15457861e-01 5.59360224e+00
     1.91667469e-01 3.62915444e+00 1.07249512e+01 1.72776133e+00
     1.92885461e+00 2.57736661e-01 1.35175438e+00 2.62544680e+00
     4.85256745e+00 3.15570421e+00 2.09247332e+00 3.43239027e+00
     6.00611663e-01 4.14907372e-01 4.08836318e+00 4.90012849e+00
     3.75203398e+00 1.00555254e+01 3.11804240e+00 2.71078207e+00
     5.17810592e+00 4.04606333e+00 3.80902987e+00 1.15780673e+00
     1.08995839e+00 3.36255660e+00 7.78734171e+00 4.77826717e-01
     2.99402180e+00 2.02700715e+00 5.12425425e+00 8.91460030e-01
     1.29480591e+00 4.69148399e+00 4.76116580e-01 5.54712073e+00
     4.62408966e+00 4.57295574e+00 2.15711292e+00 4.39497273e-01
     1.76670058e+00 2.35159889e+00 2.43988851e+00 1.02782281e+01
     2.05621408e+00 3.22510610e+00 6.87914894e-01 3.92793557e+00
     8.76030196e-01 1.32854966e+00 5.42330547e+00 1.11790378e+01
     3.73256317e+00 3.37951990e-01 2.24840121e-01 2.65493354e+00
     2.65698679e+00 2.24868168e+00 2.08626708e+00 3.89434355e-01
     9.73429925e-01 3.76313549e+00 9.34300191e+00 1.22339855e-01
     1.74457417e-01 7.36186057e-01 1.50572622e+00 1.12170137e+01
     8.20356205e-01 3.65460794e+00 1.52410836e+00 1.22518137e+00
     5.19461371e+00 3.02674040e+00 4.67760464e-01 9.17359757e+00
     9.21879160e+00 3.26602702e+00 6.56012320e-02 5.99117405e-01
     5.59711960e-01 4.57314902e-01 1.12535161e+00 2.14830216e+00
     3.55894457e+00 1.68487179e+00 1.21491714e+00 5.61786689e+00
     8.75980424e+00 2.69662952e+00 4.05848219e+00 1.56558245e+00
     5.54805076e+00 6.85639862e-02 5.11516763e+00 2.38423903e+00
     1.10741536e+00 1.75307591e+00 4.04284029e+00 7.31480330e-01
     1.43022293e+01 4.08072766e+00 5.74593002e+00 1.07306146e+00
     2.59776808e+00 3.83677727e+00 3.68246078e-01 6.19174328e+00
     5.25745249e+00 6.60932736e-01 2.16602365e-02 3.88866709e+00
     4.34525713e+00 2.00266289e+00 1.79567651e+00 3.10892311e-01
     2.89425068e+00 3.46715123e+00 1.23247542e+00 5.94036055e-01
     6.83078292e-01 1.47308447e+00 4.83746547e-01 8.02011107e-01
     9.89754890e-01 2.05430351e+00 2.68360058e+00 3.50465903e-01
     3.81939796e+00 9.83855976e-01 2.90573965e+00 3.77526863e-02
     1.84981478e-01 3.98392226e+00 1.10342301e+01 1.99616221e+00
     2.39166618e+00 4.02424048e-01 3.89516392e+00 1.50935878e+00
     3.26303797e+00 1.25945526e+00 3.81936773e-01 7.67982234e+00
     1.84359642e-01 4.03127597e+00 5.60656663e+00 9.47921635e+00
     1.02990254e+01 2.00707381e+00 8.65800974e+00 3.88817150e+00
     1.44866156e+00 1.17138967e+00 1.01618372e+00 3.31570936e-01
     6.40053761e+00 1.20673568e+00 4.34615318e+00 3.55742017e+00
     7.41859744e+00 2.71715804e+00 3.42969007e+00 9.55894911e+00
     7.84171177e+00 5.63726448e+00 1.09824900e+01 7.29135827e+00
     6.24843269e+00 3.39670612e+00 1.29308486e+00 4.49777471e+00
     1.28743120e+01 5.37347913e-02 5.82902970e+00 1.71740598e+00
     1.23938147e+01 7.32405819e+00 1.11527058e+00 2.25056664e+00
     4.48628308e+00 3.07602430e+00 5.14516804e-01 4.60542488e+00
     3.94173967e+00 2.57626290e-01 4.07362307e-01 2.38764823e+00
     1.57684503e+00 2.13814621e+00 8.00093759e+00 6.27873684e-01
     1.68086308e+00 2.65581422e-01 1.29632184e+00 2.99406404e+00
     9.79409517e-01 6.26387160e-01 1.65613410e+00 4.08209180e+00
     7.84676749e-01 1.36142709e+00 2.56236343e+00 9.18802378e-01
     8.92836934e+00 6.29890913e-01 3.60916702e-01 2.29182315e+00]
    θ estimado: 3.1033, número de iteraciones: 1
    [2.02844978e+00 2.12813946e+00 1.05094538e+00 2.67908578e-02
     1.70101047e+00 3.75958851e+00 1.71749969e+00 6.20352430e-01
     1.06727909e+00 3.81186228e+00 2.70095895e+00 6.01531829e+00
     2.69002186e+00 2.45512171e-01 1.92491122e-02 7.76544264e-01
     4.42823193e+00 5.16568091e-01 1.58531058e+00 4.75146895e-02
     1.41409731e+00 8.95669835e+00 1.49206989e+00 7.41679225e+00
     2.47521555e+00 3.67232446e-01 4.61039798e+00 1.55885392e+00
     2.84908426e+00 8.32667678e-01 1.40607782e-01 6.90005408e-01
     2.25503361e+00 2.03802659e+00 1.26972662e+00 7.74286441e+00
     1.28465836e+00 6.07347028e-01 4.02956855e+00 5.88076109e+00
     5.00392339e-01 6.09414959e+00 4.11229582e-01 2.46170425e+00
     3.05412739e+00 2.72030672e+00 1.30234042e+00 8.73548765e+00
     1.89907548e+00 1.38710672e+00 3.64666953e+00 2.08027978e+00
     9.34665999e-01 9.74269605e-01 2.90791640e+00 5.48625830e-01
     7.13952742e+00 1.29589343e-01 4.37579824e-01 6.01925832e+00
     4.87872591e+00 4.78768406e+00 5.41759941e+00 2.80566185e+00
     5.86312611e+00 5.93168125e+00 2.87008091e+00 2.12758127e+00
     1.69346069e+00 3.35897901e-01 1.48204987e+00 5.08006621e+00
     7.72192790e+00 4.57559208e-01 1.60205153e-01 8.87992697e-01
     1.01889568e+00 2.48629710e+00 7.69990503e-01 7.00539904e-01
     1.99432562e+00 4.90989190e-01 1.16398001e+00 9.60671445e-01
     1.38206298e+00 2.99961651e+00 5.54468025e-01 5.24401598e+00
     6.16188376e+00 8.12769477e-01 6.35430121e+00 1.26032131e+00
     3.13440392e-01 7.24667878e-01 1.13829388e+00 3.81878238e+00
     4.65272320e+00 2.78057698e+00 9.60238967e-01 3.52121323e+00
     1.11286007e+00 2.79060762e-01 5.28931505e-01 1.80763830e+00
     4.87507730e+00 6.60881990e+00 3.50464331e+00 2.32621299e+00
     1.78830783e+01 4.56644024e+00 4.02822306e+00 2.28665249e+00
     7.55522976e+00 6.31930966e+00 1.47864739e+01 8.78246966e+00
     1.72788168e+00 1.16533574e+00 2.18891961e+00 1.13762361e+01
     3.89584053e+00 1.76842024e-01 1.66602175e+00 3.00948557e-01
     3.32857136e+00 7.83551578e+00 1.47216972e+00 4.07478762e+00
     6.48353348e+00 2.03466550e+00 2.23073247e+00 1.01164470e-01
     1.79178170e+00 2.95713464e+00 9.96940375e-02 9.21386715e-01
     2.04652700e+00 1.24219753e+00 3.01369631e-02 3.21190777e-01
     2.67238280e+00 6.36577207e-01 2.10659762e+00 2.26901865e+00
     1.43069661e+00 4.29059244e+00 2.01373544e+00 4.59583899e-01
     2.38917954e-01 6.02908335e+00 2.10593220e+00 4.02398960e+00
     6.09965443e-01 3.92750068e+00 2.81579186e+00 7.45522951e-01
     5.03776881e-01 5.91848509e+00 2.74035067e+00 2.75302562e+00
     3.60696120e+00 7.90370970e-01 5.25200275e-01 1.81717541e+00
     1.09331582e+00 9.16745655e+00 6.90517672e-01 2.43959814e+00
     2.19720143e-02 1.67892674e+00 7.80624597e-01 7.84792046e-01
     1.46618814e+00 6.64485624e-01 2.14923779e+00 1.01840798e+01
     1.74682491e+00 6.26043052e-01 7.30004875e+00 2.91373128e+00
     2.35540700e+00 2.05983455e+00 1.31030815e+00 7.51864832e+00
     6.61775554e+00 5.75521317e+00 3.86611755e-01 5.07939951e-01
     9.04119023e+00 2.58365347e+00 1.23956371e+00 1.83312353e-01
     2.68667990e+00 5.45895067e+00 4.06284502e+00 6.10128664e+00
     2.75440286e+00 1.60812045e+00 5.28360269e+00 1.04354959e+01
     2.24415970e+00 3.72102893e-02 6.89326470e+00 5.19107719e-01
     5.60063001e-01 9.46045322e+00 3.57565585e-01 2.24094517e+00
     3.91851940e+00 3.82710763e+00 1.05359814e+01 4.05411763e-01
     3.73807480e+00 2.24007830e+00 1.06881286e+00 8.68962528e-01
     2.62263639e-01 5.71704271e-01 5.29794806e+00 1.13528186e+00
     2.29365132e+00 5.56461136e+00 2.76857105e+00 2.60504791e+00
     5.38147855e-01 1.18663145e+00 3.86478523e-01 3.39641254e+00
     1.44505546e+00 7.48729255e+00 4.55466366e+00 1.15341338e+00
     9.91318482e-01 2.00485737e+00 8.41252521e-01 1.80074831e+00
     2.92651745e+00 2.50832345e+00 6.27881318e-01 9.36805882e-01
     3.21524204e+00 8.69955328e-02 2.09566948e+00 7.67479978e-03
     5.04989764e+00 6.20997411e+00 4.93923140e+00 1.27807369e+00
     4.70620549e+00 5.23884225e+00 4.36749682e-01 8.53925091e+00
     1.18581333e-01 4.20326202e-01 8.31877942e-03 1.67429417e+01
     1.40946294e+00 3.74901997e+00 4.73814380e+00 2.99594330e+00
     6.31123563e+00 1.16248261e+00 2.16775540e+00 3.46526591e-01
     3.57399138e-01 6.07432386e+00 3.56712094e+00 7.20819200e+00
     4.79399767e+00 3.12481367e-01 1.10099132e+00 1.08575331e+01
     6.47494742e+00 9.92268942e-01 1.44183415e+01 4.85103072e+00
     9.40757171e+00 4.39191668e+00 2.16394531e+00 2.25644766e+00
     7.03393974e-02 2.04388743e+00 3.83385923e+00 1.31159162e-01
     7.76588513e-01 2.43044480e+00 4.51281926e-01 5.11515347e-01
     5.59151948e+00 1.38157537e+00 2.06290929e+00 7.61160834e-02
     1.76877667e-01 9.65588203e-01 5.00680737e+00 2.74537205e+00
     2.23521079e+00 6.87790398e-01 6.18738274e+00 3.24174683e+00
     9.43362879e-02 3.00213431e-01 1.26851321e-01 4.93470169e+00
     6.85006855e+00 1.13597316e-01 1.33564545e+00 8.32286208e-01
     9.03659771e-01 1.35781486e-01 2.49126219e+00 3.29283553e+00
     2.36662019e+00 4.92225843e+00 6.18576272e-01 2.54925979e+00
     2.82821453e+00 6.76947478e+00 4.29136648e+00 5.37977013e+00
     2.37047951e+01 6.17205996e-01 1.93233783e+00 1.56582873e+00
     4.48084597e+00 1.96481212e+00 1.71123693e+00 1.29209960e+00
     3.27085387e+00 1.73343935e+00 1.36314336e-01 4.54466440e+00
     2.62583851e+00 2.93268463e+00 2.42858884e+00 3.80118542e-01
     1.55524378e+00 4.17483599e+00 1.69756389e+00 2.53903639e+00
     1.36623190e+01 1.90105769e+00 3.81189072e+00 2.32143389e+00
     7.74479320e-01 5.67651744e-01 1.78062728e+00 2.85220827e+00
     5.91658252e-01 1.12275928e+00 1.57171815e+00 2.96852298e-01
     1.91781025e+00 7.78891328e+00 3.76824957e+00 4.49298572e+00
     7.95128617e+00 1.43105973e+00 5.14206658e+00 2.47611704e+00
     2.70167434e+00 1.39903195e+00 4.85530903e+00 4.99035173e+00
     2.27616160e+00 6.75483288e+00 5.51365363e+00 3.75458273e+00
     7.66289466e+00 3.55003619e+00 4.19461919e-01 8.26887651e-01
     1.36100803e+01 9.93633965e-01 2.03415592e+00 7.60421676e+00
     1.61493540e+00 7.29306984e+00 4.60030448e+00 3.47326373e-01
     3.40022766e+00 1.44335189e+00 2.73077266e+00 1.75883032e+00
     7.38809001e+00 1.48558728e+00 3.42029844e+00 6.75695729e+00
     5.90384042e+00 1.94107621e+00 1.52619222e-01 9.01733132e+00
     5.86058620e+00 1.18575929e+00 4.18201796e-01 3.56822640e+00
     9.20794486e-01 6.70733987e+00 2.75255152e+00 3.74861186e+00
     5.53771111e-01 2.90850324e-01 6.89494756e-01 4.19149552e+00
     2.65323298e+00 4.08970924e-01 8.49917555e-02 6.16069483e-01
     4.05033263e+00 4.91726499e+00 1.25547237e+01 6.58867303e+00
     2.75578115e+00 2.71825032e+00 5.27503980e-01 4.55387656e+00
     1.10148895e+00 2.57397685e+00 1.14226952e+01 3.42305333e+00
     4.41727932e-01 4.55921291e+00 4.52332799e+00 5.06049166e+00
     3.05752927e+00 4.36751534e+00 2.44387836e+00 1.01478118e+00
     1.73331373e+00 1.74053378e-02 7.18801464e+00 4.41444136e+00
     1.81158719e+00 2.25504593e+00 1.90351357e+00 8.52013207e-01
     2.02911241e+00 6.63366832e-02 5.87605501e+00 3.07006486e+00
     2.69043791e+00 1.44245989e+00 4.32536844e+00 1.44700767e+00
     2.44046798e-01 3.24787280e+00 1.24718828e+01 1.68693017e+00
     5.99546525e+00 5.07805847e+00 6.67879575e+00 4.48010099e+00
     2.08244921e-01 1.28290350e+00 8.13487384e-02 1.06953331e-01
     3.22091776e-01 3.50444377e+00 1.76873200e+00 2.18694887e+00
     3.22304991e+00 2.29307105e+00 1.92662938e+00 2.80862241e+00
     3.67157061e-01 7.44947345e-01 8.47799071e-01 4.01610104e+00
     5.32472674e-01 1.47908849e+01 2.90728544e+00 1.13042195e+00
     3.30654232e+00 3.08057619e+00 2.20267143e-01 2.13247483e+00
     7.30850395e-02 6.01505880e-01 6.97081870e+00 9.46570660e+00
     2.72307379e+00 6.69373659e+00 4.29731833e-01 2.22633584e+00
     3.63941307e-02 2.84422182e+00 1.32547415e+00 7.42333646e+00
     3.93107377e-01 1.27246705e+00 6.26247823e+00 1.84084735e+00
     6.71504492e+00 5.58454360e-01 4.86640623e-01 5.11852518e-01
     1.03319835e+01 1.81801508e+00 1.85562277e+00 7.19908302e+00
     3.91735561e+00 6.45576781e+00 1.07559310e+01 6.42420268e+00
     1.79095945e-01 4.01161646e+00 3.57503703e+00 5.45039763e+00
     5.20009402e+00 6.85668655e+00 2.84085704e+00 8.21120180e-01
     1.15655394e+01 4.10685731e+00 5.56346667e+00 6.39844085e+00
     2.81773700e+00 1.42550120e+01 2.99869442e+00 7.65411357e+00
     1.05437553e+00 5.64729132e+00 4.92762576e-01 1.29110434e+00
     5.08088339e-01 1.08172099e+00 3.30435197e+00 5.61375764e+00
     5.12843276e-01 2.45281408e-01 2.16887476e+00 1.90578051e+00
     6.33405255e+00 5.66981506e+00 1.40694251e+00 1.04379251e+00
     2.00757988e+00 9.75895715e-03 3.45895763e-01 2.59583553e+00
     4.66006480e+00 3.01266442e-01 8.32061403e+00 7.03852813e-01
     1.89821113e+00 1.31483000e+00 6.74848145e+00 5.13248531e-01
     9.55463210e-01 9.60041288e-01 5.00206509e-01 8.11349782e+00
     7.15141966e-01 2.67899201e+00 9.72312786e-01 7.69086398e-01
     1.24334438e+00 9.85628479e-01 7.56956039e-02 2.42389739e+00
     4.58601933e-01 9.09529283e+00 1.44091133e+01 9.21399346e-01
     4.66545906e+00 7.06386590e+00 2.44494090e+00 1.37261362e+00
     8.11014209e-01 1.69107320e+00 2.64432881e+00 8.35939959e+00
     5.49470224e+00 3.43127641e+00 6.07293109e-01 1.12776057e+00
     5.73545121e+00 2.02366877e+00 2.26243354e+00 3.18337496e+00
     4.31934580e+00 6.43238989e-01 6.25218691e-01 3.00745039e+00
     5.21077923e+00 9.66898021e-01 3.36325429e+00 1.22006649e+00
     2.50066156e-01 2.46160207e+00 1.09108735e+00 2.34648486e+00
     2.01410244e+00 4.17777270e-01 4.65854305e+00 7.29893452e-01
     1.06913236e+00 4.82757186e+00 1.43141468e-01 1.19772671e+00
     2.00720457e+00 3.88251095e-01 2.86006831e+00 7.15739820e-01
     5.46295922e+00 1.17481293e-01 2.06622840e+00 5.58737379e-01
     5.07286572e+00 1.32613605e+00 1.68939016e+00 2.30661079e+00
     1.53676044e-02 2.56944691e+00 1.27641490e+00 1.18965849e+01
     5.77306086e-02 8.73058297e+00 1.06334723e+00 3.61731799e+00
     1.08574196e+00 9.02423394e-01 7.97740802e-01 1.68991692e+00
     1.43634557e+01 1.38868481e+00 4.01805246e+00 5.09142861e-02
     5.81819262e-01 5.47134180e+00 7.62714386e-01 8.60650902e-01
     1.62000410e+00 8.52517840e-01 3.13025601e+00 4.77798621e+00
     2.35947752e+00 5.49577266e+00 3.99576529e+00 2.43551271e+00
     3.26172139e+00 1.07527929e+00 9.86730903e-01 3.43350212e+00
     3.22048484e+00 1.51385771e+00 1.44960496e+00 2.75928797e+00
     8.05083622e+00 3.35855732e+00 1.07949554e+01 7.48665071e+00
     1.13300372e+01 1.09927016e+00 3.79411632e+00 6.91732760e-01
     5.28337576e+00 2.57052958e+00 2.41880736e+00 1.20511780e-01
     2.05144663e+00 6.05500421e+00 3.45722410e+00 3.03235259e+00
     1.55598073e+00 1.01768304e+01 2.13467033e+00 4.08786767e+00
     7.18800835e+00 2.21016393e+00 4.30574070e+00 8.66496678e-01
     3.02601529e-01 2.55729283e+00 3.35985360e+00 1.19027959e+01
     1.51640097e+01 7.33401071e+00 2.31675023e+00 1.65494622e+00
     3.33302188e+00 1.16986201e+00 2.07892397e+00 5.72674103e+00
     4.93698157e+00 1.40506055e-01 1.85129434e+00 9.32330493e+00
     2.79889876e+00 1.48351269e+00 4.00827486e+00 2.86664171e-01
     5.00762361e-01 4.85452972e+00 2.99617287e+00 1.83885840e+00
     5.38947315e+00 5.39740921e+00 1.56034361e+00 4.38477428e-01
     6.40747755e-01 4.26269627e-01 3.71545373e+00 4.65056950e+00
     8.89167260e-01 5.72226977e-01 1.34974428e+00 8.85422030e-01
     4.40896865e+00 2.28057531e+00 6.06767018e+00 6.56831415e-01
     6.09600328e-01 2.61805130e+00 1.26478013e+00 1.94438110e+00
     6.95477287e-01 1.07684403e+00 2.54900922e-01 4.32485592e-01
     4.53558145e+00 5.26389677e+00 1.39066986e+00 2.51259310e-01
     6.03314434e-01 1.46212232e+00 1.58004177e+00 2.32770238e+00
     1.35860216e+00 3.50407102e+00 8.73296642e-01 1.70028170e+00
     6.33676744e-01 1.50280675e+00 9.04617620e-01 2.16854923e+00
     2.28658657e-01 2.03069641e+00 3.15881875e-01 5.99444101e+00
     8.00861108e+00 3.89111190e-02 7.38391632e-01 3.87883670e-02
     1.99354725e+00 9.79543377e-01 7.62038988e-01 3.43844996e+00
     1.11699152e+00 2.68395722e+00 5.72265720e+00 2.73537278e+00
     7.80463485e+00 3.44929193e+00 6.00682645e-01 2.57460935e+00
     2.80083939e+00 2.71763066e-01 6.23649409e+00 1.90693367e+00
     5.12524917e-01 1.34226380e+00 2.86168685e-01 9.05343349e+00
     7.91441861e-01 1.04884581e+00 7.44342664e-01 3.87121438e+00
     1.58500841e+00 1.04926906e+01 1.77999896e+00 1.42330187e+00
     1.64266585e+00 2.48168635e+00 9.39147790e-01 6.34230237e-03
     7.06144046e+00 3.74357421e+00 1.42769656e+00 1.85105375e+00
     3.07983971e-01 5.54634037e+00 1.40627561e+01 5.69310986e+00
     2.81723952e+00 7.50801015e+00 1.02138921e+00 2.31585472e+00
     2.27715381e+00 1.13280738e+00 3.98482930e+00 1.14792068e+00
     2.05554015e+00 7.87614448e+00 1.41888382e+00 3.50580515e+00
     7.39368363e+00 2.42694528e+00 1.99711467e+00 1.45060615e+00]
    θ estimado: 3.0434, número de iteraciones: 1
    [4.34624738e-01 8.43932091e+00 3.08479867e-01 1.67018851e+00
     1.85322176e+00 2.96194859e+00 7.04818373e+00 4.19559180e+00
     1.02528378e+00 9.44921303e-03 3.98370838e+00 1.30215997e+01
     2.73741961e+00 4.03214411e+00 1.18040054e+00 8.46479920e-01
     3.04441367e+00 6.83430986e+00 3.21307322e+00 1.77535061e+00
     2.23178868e+00 1.57821420e+00 1.87150330e+00 9.34838621e+00
     2.53595424e+00 2.15780350e+00 2.10497047e+00 6.92452764e+00
     2.28907821e+01 3.30635449e+00 4.48434387e+00 8.13654481e-02
     1.32951190e+00 1.32153149e+00 5.63237782e+00 1.90309745e+00
     1.05183040e+01 1.04578644e+00 2.88800384e+00 1.33314038e+00
     2.34010582e+00 1.74703514e+00 1.30293219e+00 4.01406271e-01
     1.61302604e+00 4.30035100e+00 1.95089703e+00 3.64227082e+00
     5.04654328e+00 1.81254743e+00 6.12215359e+00 1.51193993e+00
     1.39694615e+00 8.63310694e+00 1.17889607e+00 5.38639371e+00
     1.90771296e-01 3.43589798e+00 9.21036779e+00 2.84234445e+00
     4.83882741e-01 1.88204487e+00 5.30809436e+00 1.66569005e+00
     2.27008742e+00 1.21385319e+01 3.81969853e+00 4.37923768e+00
     2.39835752e-01 6.26636766e+00 3.68044148e+00 8.81192129e+00
     8.04937964e+00 4.21843851e+00 2.57742440e+00 3.79036274e-02
     5.93356041e-01 5.44242605e+00 4.36696449e+00 4.89288998e+00
     6.95286038e+00 4.45138001e+00 2.29365161e+00 4.22254399e-01
     1.48568476e+00 1.06232782e+01 1.38163400e+00 3.55482697e+00
     7.56624527e-01 2.68229008e+00 6.97956880e-01 1.14328749e+00
     1.81613312e+00 1.09153279e+01 4.25866980e+00 3.32797947e+00
     5.43447707e-01 8.09963373e+00 1.07085398e+01 5.80978300e-01
     3.91156689e+00 9.02384022e-01 6.16101974e+00 1.13349883e+00
     8.71453113e-01 1.79150158e+00 4.47742005e+00 1.73823136e+00
     2.05015926e+00 4.76278498e+00 5.55098979e+00 1.52172499e-02
     4.20073370e+00 2.02567047e+00 1.50582730e+00 2.29949632e+00
     4.30642625e+00 2.35490219e+00 2.99605441e+00 7.65934483e-01
     2.96952887e+00 4.27752493e+00 1.23857539e+00 2.20916882e+00
     1.18823191e+01 1.15330144e+00 3.35114378e+00 2.51737565e+00
     2.88380812e+00 9.87426177e-01 3.51779488e+00 8.65955953e-02
     1.91412213e+00 2.52916197e+00 4.64106998e+00 1.03303827e-01
     1.08715725e+01 8.46924068e-01 2.83627184e-01 2.70016657e-01
     5.85645899e+00 1.39588325e+00 3.49907348e+00 7.57474136e-01
     1.08185992e+01 2.60407501e+00 5.49938085e+00 2.17004458e+00
     7.51536408e-01 9.78444151e-01 1.06198967e-01 9.83099193e-01
     2.79111584e+00 3.12081116e-02 1.19458092e+00 1.33803134e+00
     1.58011476e+00 8.49760135e+00 2.40659084e-01 2.54132882e-01
     9.72243096e-01 2.36890028e+00 2.20198176e+00 1.63139820e+00
     1.05171336e+00 2.48569571e+00 1.38441409e+00 1.23936091e+00
     8.25891478e+00 8.94746786e+00 3.90431588e+00 2.24257492e+00
     5.87483951e-01 3.52339749e-01 5.67763635e+00 1.75419098e+00
     1.65841034e+00 1.07133041e+01 5.76967027e-01 6.43667769e-02
     1.08529837e+00 1.35340695e+00 1.39686423e+00 7.92696648e-01
     1.47965449e+00 6.06376535e+00 2.45700298e+00 5.69090812e+00
     1.02850940e+01 3.75487524e+00 3.78428077e-01 1.51629809e+00
     1.76843108e+00 4.35336075e+00 5.96541942e+00 3.98526061e+00
     3.23545400e+00 1.27003318e+00 5.67235771e+00 4.18378772e-01
     1.35436911e+00 5.98682093e-01 8.95142324e-01 6.79847631e+00
     1.35738793e+01 1.45956206e+00 3.74424537e+00 4.29385002e+00
     9.06769106e-01 9.67331806e-01 1.12077132e+00 5.77332291e+00
     9.10178706e+00 1.98014007e+00 3.22507980e+00 9.41995974e-01
     2.72028850e+00 5.28714909e+00 6.65948659e+00 1.28056112e+00
     1.76169436e+00 1.84705418e+00 6.47738139e+00 6.91799751e+00
     3.56792680e+00 1.74056644e+00 5.14599453e-01 8.41319953e+00
     9.37242181e+00 7.10001803e+00 2.87768875e-01 3.93774987e+00
     6.14666102e+00 1.81193868e+00 8.00053591e-01 2.63562595e-01
     2.52211156e+00 6.73600820e+00 2.70832783e-01 2.67536222e+00
     2.34445573e+00 5.09105656e+00 7.41114715e+00 2.41046406e+00
     1.05563143e+00 2.75906184e-01 6.79160724e-01 4.37920066e+00
     5.55255640e+00 3.22255964e+00 9.62238651e+00 1.45548179e+00
     3.48251124e+00 2.92247048e-01 1.13708292e+01 1.06749103e-01
     7.85952690e+00 2.14175243e+00 8.18150484e-01 1.50567198e+00
     2.16931409e+00 1.63624028e+00 2.20011485e+00 1.64134762e+00
     5.04384880e+00 2.97579290e+00 2.66117939e+00 2.65556950e+00
     1.71309987e+00 4.34489994e-01 8.30341193e-01 7.93022679e-01
     7.33170711e+00 2.19373601e-01 1.34513258e+00 1.71295065e+00
     5.07371805e+00 3.88358244e+00 3.19030075e+00 3.53081385e-02
     2.57006612e+00 3.90752962e+00 1.78108029e+00 4.73895596e-01
     2.95797603e-01 1.01689227e+00 1.83306604e+00 1.83022846e+00
     1.82078354e+00 7.94271813e+00 3.73836623e+00 7.93559533e-01
     4.30298347e+00 3.37123998e-01 6.76220301e+00 4.63046186e+00
     2.59327831e-01 4.25009634e+00 5.53043510e-01 2.88977856e+00
     6.36958508e+00 2.54016875e-02 4.24484916e-01 4.98899525e+00
     1.09840748e+01 9.10816645e+00 1.99868506e+00 1.53307861e+00
     6.21685726e-01 5.83335423e-02 3.85062469e+00 3.28625093e+00
     7.74360395e-01 1.08014628e+00 4.45157737e-01 7.34618494e-01
     4.17615080e+00 1.33840244e+00 9.17080885e-01 4.96689244e-02
     7.96106833e-03 4.87551967e+00 1.35088628e+01 2.02190763e-01
     2.49713444e+00 1.95811681e-01 9.45827895e-01 9.63386380e-01
     1.97299383e+00 3.91950510e-01 1.70775221e+00 1.20651666e+00
     3.46928441e+00 2.90758175e+00 2.49021561e+00 4.60218024e+00
     4.22606446e+00 2.67431677e+00 3.50329360e+00 5.78860988e+00
     1.18937932e+00 8.03184161e-01 7.35457997e-01 1.41689344e+00
     2.40997549e+00 2.47293422e+00 1.52422034e+00 2.03736390e+00
     1.21624884e+00 3.75037188e+00 3.82097618e+00 1.42318436e-02
     1.81280190e+00 7.64612143e+00 2.65073877e+00 3.83728386e+00
     6.28851352e+00 1.54988526e+00 1.98573443e+00 4.38313477e+00
     1.30357774e-01 6.03958629e-01 3.30530519e+00 1.00379627e+00
     6.55104346e+00 5.46928413e+00 1.22801491e+00 2.44192899e+00
     2.02636361e+00 3.36585290e+00 4.55916031e+00 3.74214936e+00
     6.50545552e-01 1.43230199e-01 2.95672667e+00 1.35961769e+00
     2.07696221e+00 4.53119046e+00 9.57428730e-01 9.57823506e-01
     1.41484460e+01 7.24086345e-02 3.38326721e+00 4.11993835e+00
     1.29608733e+00 1.07328630e+00 1.59778122e+00 2.91704442e+00
     4.29318831e-01 9.51618996e-01 4.62824210e-01 4.20334535e+00
     1.52960374e+00 3.02601219e-01 7.02183128e+00 1.68586531e+00
     2.04284970e+00 6.94461616e+00 5.93643995e+00 1.60403956e+00
     1.80139942e-01 1.19913275e-01 4.89484421e-01 5.26772834e-01
     5.65591097e-01 3.35129404e+00 1.43726536e+00 3.61992320e+00
     3.39656353e+00 5.77936683e-01 4.23040905e+00 4.74281180e+00
     7.48231027e+00 8.90413702e-01 9.17094258e-01 1.15000582e+00
     8.18450012e+00 8.56617559e-01 3.21010768e+00 9.18787500e+00
     6.48328953e+00 9.85556952e-01 1.16413836e+00 5.36950348e+00
     4.72366462e+00 6.02224099e+00 1.31767772e+00 3.07341350e+00
     1.01842741e+00 5.09298751e+00 8.67077472e-01 2.01510583e+00
     4.17201809e-01 2.32780669e+00 5.27958912e-01 2.08624501e+00
     1.33360655e+00 3.26878029e+00 5.77613389e-01 3.92589165e+00
     3.94989106e+00 3.41642302e-01 2.97553379e+00 6.55871994e-02
     8.63714279e-01 2.08078288e+00 5.39636038e-01 9.01262306e-01
     2.62402017e+00 1.55647915e-01 2.82055031e+00 9.27774860e+00
     5.02807902e+00 5.03566664e+00 1.33821310e+01 4.53962044e+00
     9.61931029e+00 1.35444610e+00 7.89227152e-01 5.71709552e+00
     1.12630481e+00 2.19310182e+00 6.69887893e+00 1.37355998e+00
     2.76642631e+00 1.04067222e+00 2.59245349e+00 4.08558495e-01
     7.48112845e+00 3.62766987e-02 8.14591871e-01 2.49366879e+01
     2.43980204e+00 2.17836320e+00 8.25493266e-02 2.43157565e+00
     3.67740612e+00 1.59917442e+00 5.49416884e+00 6.12033626e+00
     1.75253949e+00 5.57737577e+00 8.13809675e-01 1.12093409e+01
     2.99823126e+00 7.59286343e+00 2.00772415e+00 2.52272687e+00
     4.62086871e+00 3.79217705e+00 1.02780912e+00 9.44285335e-01
     1.86032949e+00 3.49181695e+00 5.90864836e+00 3.30423733e+00
     1.40995581e+00 6.57633948e+00 1.37120502e+00 4.75766554e+00
     8.32579380e-01 4.19131339e+00 2.64651471e-01 1.28974055e+00
     5.35316745e+00 1.66310378e+00 4.47013823e-01 1.90562077e+00
     5.31671794e+00 3.64510910e+00 2.94894787e-01 3.71232711e+00
     8.59464446e-01 3.72634541e+00 6.01277180e-01 5.49919079e+00
     8.83889915e+00 1.47922147e+00 3.03048322e+00 1.85207059e+00
     2.89086802e-01 2.25696000e+00 2.94204108e-01 2.14547585e+00
     1.89192350e+00 2.89346381e+00 7.98633060e+00 5.73735280e+00
     2.57709539e+00 1.28088274e+00 1.85149372e+00 1.32999220e+00
     1.75117943e+00 5.02592786e+00 3.27713508e+00 3.88405378e+00
     4.63334839e+00 5.88398520e-01 4.32307353e+00 5.25745324e-01
     1.51973368e-01 2.54125332e+00 7.11076497e-01 2.49559979e+00
     2.99640423e+00 6.83782143e-01 2.70337060e+00 3.94155071e-01
     9.02306165e-01 1.45016558e+00 2.74425904e+00 2.22142572e+00
     6.58375062e+00 3.64758131e+00 1.28756080e+00 2.89182424e-01
     3.86887754e+00 2.46933586e+00 1.10047254e+00 7.79306716e+00
     6.48622537e+00 5.90427499e+00 2.56361696e+00 5.29920824e+00
     2.21822570e+00 1.11711829e+00 6.61615812e-01 4.32564199e+00
     1.14985265e-01 2.05742663e+00 2.17873932e+00 6.52258842e+00
     3.70909539e+00 1.60212032e+00 1.95824775e+00 9.63622691e-02
     6.67473459e+00 4.97117628e-01 8.33476144e+00 1.13418865e+00
     3.90538257e+00 1.62490495e+00 4.61241223e+00 4.11515257e+00
     3.36090988e+00 7.81907233e-01 7.99161176e-01 2.96382459e+00
     1.22137850e+00 7.95522113e+00 1.11720137e+00 1.06552176e+01
     3.94637892e+00 3.46745683e-01 2.69020442e+00 1.01999511e+00
     5.16298882e+00 9.14223525e+00 4.56879897e+00 4.17427944e+00
     3.45509677e+00 5.21866152e+00 6.56275969e+00 6.44065344e+00
     5.68104728e-01 5.85369351e+00 6.14282328e-01 7.74080644e-01
     1.40995409e+00 2.18182449e+00 4.49132401e-01 6.76771665e-01
     1.82796866e+00 2.15055854e+00 7.96660911e+00 2.31135337e+00
     3.92660945e+00 2.31620384e-01 1.24878391e+00 1.26327273e+00
     3.28190883e+00 4.78627398e-01 3.86640350e-01 4.83044012e+00
     1.72543398e+00 4.52819373e+00 3.52745271e+00 2.87998926e+00
     4.10515901e+00 2.29171876e-01 2.00951954e+00 8.95013621e+00
     3.12359832e-01 2.47890554e+00 1.10598646e+00 2.63150989e+00
     1.01807905e+00 9.00972981e+00 1.56638651e+00 1.37990602e+00
     9.43333818e-02 2.62011408e+00 3.80408627e+00 2.91317358e+00
     8.16672654e-01 9.95687970e+00 3.17299084e+00 1.91681414e+00
     1.97522967e+00 2.11906819e-01 1.82815711e+00 2.48269041e-01
     1.56692499e+00 8.06316631e-01 7.52626335e-01 2.37026980e+00
     3.64657183e-01 2.41626452e+00 5.15619951e+00 4.82233245e+00
     6.63294823e+00 1.15326182e+00 6.00981965e-01 3.99847902e+00
     4.04293467e+00 1.23530646e+00 1.01011868e+00 1.93887666e+00
     5.14720290e+00 4.35442962e-01 7.56245078e+00 3.65135766e-01
     4.40653103e-02 2.04767817e+00 3.71152348e+00 3.43034960e+00
     4.79697095e-01 9.90491479e-01 2.59056033e+00 2.45053597e-01
     3.75581130e+00 1.10038206e+00 5.35362928e+00 1.80185090e+00
     6.54841731e+00 5.44346784e+00 2.26158268e+00 3.43677694e+00
     3.50680396e+00 6.31133364e-01 6.79310604e-01 2.11877700e+00
     5.73098721e+00 5.97871401e+00 9.22523902e+00 1.63939028e+00
     9.61033514e+00 2.04152797e+00 7.40521100e-01 4.94064954e+00
     1.96637109e+00 4.77071938e+00 5.12587015e-02 4.87685035e+00
     9.23598491e-01 1.10383043e+00 1.49207008e+00 2.33407315e+00
     1.85573916e+00 2.39717282e-02 4.87822944e-01 1.31139694e+01
     4.65730289e+00 3.38184396e+00 8.25630252e-02 7.96037369e+00
     8.13857423e+00 7.74948343e-01 1.99318394e+00 5.81424805e+00
     3.88335543e-02 2.08008340e+00 9.00469722e+00 7.36095640e-01
     5.92048405e-01 5.06775940e+00 9.30393241e-01 7.92437483e+00
     4.12151763e+00 2.18030132e+00 2.47210485e-01 1.57459647e+00
     1.43147595e+00 1.37907924e+01 1.10753044e+01 4.91974217e-01
     1.17516484e+00 7.01157514e+00 1.47141688e+00 7.57744137e-01
     3.64218062e+00 3.92325062e+00 4.40062276e+00 3.01852623e-01
     5.47731682e-01 1.51313160e+00 3.67057902e+00 1.31284975e+01
     2.39314250e+01 2.23499294e+00 2.31911985e+00 1.32271037e+00
     1.90638895e+00 1.55708458e+00 1.02606612e+00 4.10029472e+00
     4.50745463e+00 4.29253186e+00 4.39017631e-01 2.48020675e+00
     6.34474040e-01 4.35506828e-01 7.51443486e+00 4.38668789e+00
     2.20736188e-01 7.64758473e-02 6.27853552e+00 5.61593213e+00
     1.12363532e+00 3.27300774e-01 7.78561124e-01 1.33513953e+00
     6.40893928e+00 1.21089724e-01 2.46414354e+00 8.68228094e-01
     8.08287564e-01 5.83027023e-01 3.38175857e+00 3.35623895e+00
     5.06484773e+00 9.52783831e+00 1.02751849e+00 3.02088179e+00
     2.67010527e+00 6.09729062e+00 1.27227445e-01 3.15741480e+00
     5.39312491e-01 2.60587353e+00 8.09499747e-01 1.22301444e+00
     1.62447494e+00 6.82930026e+00 2.84987553e-01 4.95111085e+00
     1.68846031e+00 4.10996994e+00 2.73934866e+00 1.81192387e-01]
    θ estimado: 3.0994, número de iteraciones: 1
    [6.13285854e-01 1.84196068e+00 1.35876766e+00 2.26851061e+00
     2.70252481e+00 3.21218672e+00 1.43698303e-01 1.70287217e+00
     4.53905125e+00 5.60329380e+00 2.53970946e+00 2.17135121e+00
     1.60114443e+00 7.08826086e+00 1.54308632e+00 4.95907138e+00
     1.61533874e+00 1.21116755e+00 8.74673048e+00 3.17938029e+00
     1.37432552e+00 1.46845865e+00 5.49853495e+00 1.14186304e+01
     4.87620658e+00 3.92900632e+00 4.27750062e-01 6.51422973e-01
     2.64480334e+00 1.02800450e+00 1.23910122e+00 9.35563230e-02
     4.49441166e+00 1.81741044e-01 4.13273299e+00 1.08472185e+00
     8.72343309e-01 3.82573743e+00 9.13870328e-01 5.87045718e+00
     3.40681269e+00 1.78759221e+00 1.90433220e+00 2.93470607e+00
     2.08105720e+00 9.60535152e+00 6.77930765e+00 1.33094055e+00
     1.01256158e+01 1.14916488e+00 2.36608975e+00 5.65569973e+00
     1.68593883e+00 1.23039507e+00 1.62901997e+00 4.32782392e-01
     2.28214253e-01 9.05723830e-01 5.22696486e-02 4.50333144e-01
     1.26974072e+01 8.55466993e-02 3.58390834e-01 4.39362750e+00
     2.52415625e+00 2.18421175e+00 2.20108786e+00 5.78952251e+00
     1.54107938e+00 8.10231717e-01 1.16452217e+00 3.16448593e+00
     7.76595474e-02 5.03323922e-01 1.54162562e+00 2.25993612e-01
     1.27760084e+00 3.88244848e-01 1.94230605e+00 7.13399598e-01
     2.01753581e+00 1.55600751e-01 3.26468170e+00 3.29825262e-01
     7.10292583e+00 1.57424486e+01 2.86968148e+00 6.49111120e+00
     2.05309839e+00 2.40023694e+00 6.17712702e+00 5.68355576e-01
     1.05114508e+00 2.25889501e+00 1.65260198e+00 1.93105337e-02
     3.41586488e+00 1.44533675e-01 4.03984596e+00 5.20324360e+00
     2.14546714e+00 1.79062611e-01 2.33987607e+00 3.04566037e+00
     6.15222666e+00 3.23861513e+00 5.03259291e+00 1.44327256e+00
     4.47037251e+00 4.92966630e+00 1.91066962e+00 2.41730948e+00
     5.73690711e+00 3.42698249e-01 7.41490765e-01 4.24593000e+00
     9.67545398e-01 7.55127108e-01 5.76332992e-01 4.74559103e+00
     3.10335691e+00 9.37356425e-01 2.13774632e+00 1.36034939e+00
     1.73956407e+00 1.14414040e+00 1.77514913e+00 3.62982234e-01
     2.20546943e+00 1.77691446e+00 2.34928152e+00 3.01767799e+00
     5.14797998e-03 5.14301624e-01 2.86357961e+00 1.23859245e-01
     1.11202249e-01 1.92343900e+00 7.14678182e-01 4.09143901e+00
     2.46226297e+00 8.37811139e-01 2.01785841e+00 4.33356598e+00
     1.33600084e+00 2.47623121e+00 2.23241529e+00 4.06538337e+00
     3.40933345e+00 3.78735007e+00 3.40270779e+00 1.43293259e+00
     5.00658291e+00 5.44557680e+00 6.16186377e-01 4.52729682e-01
     3.76409317e+00 1.84790810e+00 1.07764920e-01 1.58498496e+00
     2.42853852e+00 7.04743433e-01 5.35948023e+00 1.34741830e+00
     7.08189158e-01 3.07692595e-01 2.68548894e+00 1.08675273e+01
     3.21331300e+00 2.51147845e+00 3.57049931e+00 2.05001191e-01
     1.35734992e+00 4.14897261e+00 1.08627453e+01 1.63582827e+00
     6.88535471e-01 9.44950915e+00 6.26560013e-02 1.07728980e+00
     2.76467997e+00 5.31008317e+00 1.46491701e+00 1.93312596e+00
     5.29666608e-01 3.85729485e+00 1.16493963e+00 2.67630766e-01
     6.95787211e-02 5.38799812e+00 1.15929111e-01 3.01318071e+00
     3.43754412e+00 2.91054870e-01 4.43477954e+00 6.78389952e+00
     4.55171324e-01 4.46778663e-01 1.16571233e+01 2.36913508e+00
     1.65917974e+00 8.98179478e-01 3.85867411e+00 9.16592068e+00
     8.01827413e+00 2.72188633e+00 5.00062345e+00 3.24357231e-01
     2.81075488e-01 8.46135933e-01 2.76827529e+00 3.18270631e-01
     5.58314973e-01 1.10840565e+00 1.66817594e+00 4.47908749e+00
     1.59468293e+00 4.96725650e+00 2.55267530e+00 2.07030771e+00
     1.00484747e+01 4.86411660e+00 7.82979554e-01 3.26408862e+00
     6.12812090e-02 2.53689775e-01 5.77444400e-01 2.57838991e+00
     2.00521298e-01 1.17288263e+00 4.98976677e+00 1.22702050e+01
     5.96932587e+00 7.33629997e-01 3.82329449e+00 2.16620734e+00
     4.95977222e-01 5.66545585e-01 1.79967034e+00 2.32322185e+00
     4.55820015e+00 2.82275410e+00 2.27672734e+00 5.28278024e+00
     1.13540821e+00 4.99840132e+00 1.00048795e+00 1.49173170e+00
     3.45314746e+00 3.16615913e+00 4.94613677e-01 5.01445910e-01
     3.62702826e-01 2.29140912e+00 8.82555177e-02 2.13411266e+00
     7.92549579e-01 9.33172942e-01 1.84454680e+00 7.95250342e-01
     1.53977110e+00 8.25050563e-02 1.22247447e+00 7.26820126e-02
     4.98147121e+00 1.11672940e+00 3.16132084e+00 3.33569125e+00
     2.06971984e+00 7.40804503e+00 3.26714120e-01 1.77048040e+00
     6.56751195e-01 8.19490453e+00 4.08399095e+00 4.47223147e+00
     6.35750238e+00 2.99897209e+00 3.38051849e+00 5.88109678e+00
     9.42908688e-01 2.23111057e+00 4.41021713e+00 2.30191550e-01
     1.24358537e+00 1.04850676e+00 1.83535166e-01 4.65112837e+00
     1.30968466e+00 2.78218296e+00 2.29380512e+00 1.72190765e+00
     2.92656140e+00 4.62661155e+00 1.95122050e+00 9.03501297e-01
     6.93879154e+00 1.27584616e+00 3.58370470e+00 5.60045727e+00
     2.82083260e+00 8.45853818e-01 4.47980375e+00 2.96330400e+00
     8.40032716e+00 3.40355551e-01 1.36222803e+00 1.34527878e+00
     1.72864477e+00 2.63581371e+00 3.30504559e+00 2.64440822e+00
     3.17167629e+00 3.27787726e+00 1.12370927e-01 1.37684011e+01
     2.18161991e-01 1.85686100e+00 1.91960323e+00 1.22045547e+00
     5.55792912e-01 7.54474793e+00 2.66019003e+00 1.15088310e+01
     4.39142044e-01 1.24536838e+00 6.76950049e+00 2.56302108e+00
     1.22072363e+00 4.75251538e+00 8.36767610e-02 3.73694510e-01
     5.60221967e-01 2.77534781e+00 8.41749434e-01 1.20069902e+00
     4.39886182e+00 2.74684307e+00 3.49394286e+00 1.96070405e+00
     1.81333330e+00 3.83692903e+00 2.85288241e+00 6.80366235e+00
     1.24892728e+00 4.94707629e-01 4.51328296e+00 6.21910618e-01
     5.10424464e-02 2.09468011e+00 7.57376862e+00 1.93471497e-01
     5.57424359e+00 2.31948928e+00 5.52204653e+00 7.84344162e+00
     3.35297760e-01 1.04802053e+01 1.22347206e+01 1.20564195e+00
     6.37864372e+00 1.90858051e+00 1.01154132e+01 4.31612408e+00
     2.10514710e+00 1.16674461e+00 1.81836428e+00 2.34719014e+00
     2.53762398e+00 1.10208966e+00 4.83585359e+00 6.29907227e-02
     8.90816896e+00 2.10204464e-01 8.25641760e-01 8.98476546e+00
     1.07020099e+00 7.06768246e-01 2.71123697e+00 1.83258229e+00
     5.33926289e+00 9.13423880e-01 6.48475569e+00 3.65092070e-01
     2.10579494e+00 3.52599978e+00 4.23082496e+00 2.83330679e-01
     8.26279602e-02 1.65553894e+00 2.11128496e+00 5.96463689e-01
     3.93902034e+00 1.76202604e-01 1.54407493e-01 9.23164364e-01
     3.56995892e+00 7.04838910e+00 7.61013521e-01 5.30296720e+00
     1.43452285e+00 1.16543363e+00 2.63918365e+00 1.37839766e+00
     6.00355307e-01 2.94419022e-01 1.57155686e+00 7.80046725e-01
     4.06715573e+00 7.62000380e+00 1.05987949e+01 4.45311107e-01
     3.05620506e+00 1.43041002e+01 2.80929914e-01 1.02502172e+00
     3.54184121e+00 1.87609116e+00 4.62283104e+00 1.76305659e+00
     5.72432079e+00 2.60293563e+00 8.30052410e-02 1.81743782e+00
     3.68424624e+00 1.53854178e+00 4.35078727e-01 1.57045557e+00
     3.31699445e+00 1.24836934e+00 2.38971189e+00 8.39717383e+00
     1.92627278e-01 4.07603197e+00 1.47204959e-01 7.78716216e+00
     2.65559367e+00 1.97500409e+00 1.69604984e+00 7.45380910e+00
     2.80093300e+00 8.30663098e+00 1.80675112e+00 5.10396150e+00
     1.27562633e+00 3.11567311e+00 5.61305508e+00 1.14746812e+01
     2.26203538e-01 6.67345045e-01 1.22542304e+00 2.09663059e+00
     2.25459068e+00 2.18873260e+00 4.01989668e+00 3.38799664e-01
     7.28554428e-01 1.71974808e+00 1.79287376e+00 1.91612986e+00
     2.20442939e+00 4.10000508e+00 1.81113955e+00 2.39085354e+00
     1.94785071e+00 8.18065968e-01 1.34173827e+00 1.53302525e+00
     3.02031590e-01 5.72525222e-01 1.95272606e+00 7.04938834e+00
     1.04129811e+00 5.60052179e+00 2.44292268e+00 2.47224128e+00
     1.10737305e+00 3.71029340e-01 1.13106896e+00 1.24948438e+00
     1.46919166e+00 1.24636307e+00 3.56604300e-01 4.37982781e+00
     5.01363567e-01 6.73611801e+00 6.08047623e+00 1.38273743e+00
     1.47284189e+00 2.42022162e+00 4.99149251e-01 5.38711672e+00
     9.32516723e+00 2.00153820e+00 3.68274705e+00 3.64536728e+00
     2.28995072e+00 5.76133198e+00 6.62658438e+00 1.27118272e-01
     2.33539969e-01 8.61147293e-01 3.19367115e+00 3.79519750e+00
     4.13295143e+00 3.52625316e+00 6.07123609e+00 1.15959069e+00
     2.43943567e+00 8.82999334e-01 5.50586361e-01 1.02466179e+00
     3.16232489e+00 2.41851411e+00 2.36890501e+00 6.17124784e+00
     1.49412961e+00 3.22099511e-01 2.61093768e+00 4.64229949e+00
     3.46238746e+00 2.71588434e+00 2.03328177e+00 9.99445521e-01
     8.40104716e+00 1.64341909e-01 7.24900013e-02 7.76495836e+00
     3.98836936e+00 5.72031701e-01 2.15422112e+00 2.04474775e+00
     1.79252369e+00 1.55271033e+00 3.94704916e+00 1.66504373e+00
     7.45194064e-01 4.89398825e+00 5.79165651e+00 4.87289284e+00
     1.05700333e+01 1.42183456e+00 1.81160650e-01 7.54578705e+00
     6.77643496e+00 4.37568035e+00 1.11325985e+00 5.81205026e-02
     5.89919801e+00 1.07092054e+00 6.62232281e+00 7.06087415e-01
     7.94024154e-01 6.46095763e-01 5.74452113e+00 6.80052718e-01
     6.05686995e-01 6.33106207e+00 3.46170889e+00 3.78197239e-01
     3.43569699e+00 9.29089411e-01 3.83091281e-01 7.80862817e+00
     6.04491673e-01 8.38512168e+00 2.92176746e+00 3.12144934e-01
     4.94484718e+00 1.19445489e+00 3.19545994e+00 2.04290600e+00
     8.00464181e+00 1.40499577e+00 9.70409312e-01 2.01447059e+00
     2.95342436e+00 3.31888222e+00 6.08391597e+00 1.67641913e+00
     8.33035931e-01 6.48584074e-01 1.28860677e+00 6.47915979e-01
     1.07634638e+00 1.11228866e+01 8.55421104e-01 1.22995339e+00
     5.79371415e-01 3.12489708e+00 3.17999781e+00 3.05268621e+00
     1.10883638e+01 2.31294469e+00 5.51111852e+00 1.36344558e+01
     3.48328287e+00 1.22930546e+00 2.29881875e+00 3.12113746e+00
     5.66049623e+00 8.86408079e-01 1.46262779e+00 1.85269860e+00
     8.38856276e-01 1.73600615e+00 5.59177656e+00 6.36825740e+00
     3.42623913e+00 1.19164828e+00 8.82616841e-01 3.32428649e+00
     5.30523462e+00 1.29093915e+01 2.85349494e-01 8.06487517e-01
     1.28861042e+00 3.64227874e+00 6.85542557e+00 8.19462181e+00
     2.35285419e+00 1.30792327e+00 4.19629682e-01 1.95558367e+00
     3.62182525e-01 2.77023886e+00 8.85267328e-01 7.71691489e+00
     1.70635241e+00 3.08032839e+00 1.16944846e+00 2.84672965e+00
     5.02154250e+00 2.35563766e+00 1.17613415e+00 3.09798292e+01
     1.93788415e-01 4.94364170e+00 7.35212828e+00 3.63590689e+00
     2.79651601e+00 1.22769204e+00 3.20944783e+00 3.29281638e+00
     1.26336484e+00 8.79239594e+00 2.97216351e+00 4.95810414e-01
     1.28363258e+00 1.11245679e+00 1.75782072e-01 3.60412642e+00
     1.15350230e+00 6.16830897e-01 1.28719447e+00 3.00122723e+00
     6.28926262e-01 1.09098025e+01 1.58306866e+00 1.79941584e+00
     1.09070600e+01 7.48381132e-01 4.20563800e+00 6.20206538e+00
     1.89422256e+00 8.26335842e+00 1.05967282e+01 7.26350458e-01
     4.50783866e-01 3.72534139e+00 2.60099109e-01 3.90535485e+00
     2.10108260e+00 3.50093505e+00 1.32479127e+01 7.05698611e+00
     6.61286145e+00 4.25902348e-01 3.88444836e+00 1.13555887e+00
     1.89418163e+00 2.88376563e+00 1.28249397e+00 4.71721420e+00
     4.00393001e-01 2.52213292e-02 8.49458435e+00 1.88880302e+00
     9.65133967e-02 1.17087787e+01 7.53353090e+00 8.10945545e+00
     9.49814520e-02 1.85676490e-01 5.21910468e-01 2.69428294e+00
     1.75444862e+00 8.98241908e-01 1.27500217e+00 1.10644024e+00
     9.82223822e-01 8.84344812e-01 4.22206180e+00 2.20383215e-01
     3.38881842e+00 2.04527663e+00 1.73573452e+00 1.08911799e+00
     2.36837621e+00 6.10974304e+00 1.07354257e+00 1.25125285e+01
     3.40903566e+00 1.39894063e+00 2.23651103e+00 3.99604053e+00
     3.11314806e-02 6.50641469e-01 2.83027470e+00 5.06644613e-01
     5.53770140e+00 3.62015196e+00 3.67967306e+00 2.83677128e+00
     5.12787710e+00 1.15495917e+00 3.05617863e+00 5.36239311e+00
     3.19627268e-01 1.50426782e+00 2.80270329e+00 1.57575763e+00
     1.69297310e+00 3.47068654e+00 4.25357249e+00 1.59547767e+00
     6.75075540e+00 3.77481245e+00 5.73767825e-01 2.45937463e-01
     1.80044465e+00 4.27732016e-01 5.29727367e-01 3.24237692e+00
     1.02907487e-01 4.43007178e+00 4.90358105e-01 1.61831032e-01
     2.74115655e+00 1.62014534e+00 1.26641710e+00 2.41060232e+00
     1.25274389e+00 3.51623592e+00 1.45545946e+00 1.33420057e+00
     7.56131194e-01 2.12437862e+00 2.14975514e+00 6.32707883e+00
     2.40779451e+00 3.02485875e+00 1.74730441e+00 1.10774693e+00
     6.37398847e+00 4.26526903e+00 3.20369433e+00 2.87884107e+00
     5.74186368e-01 8.82457977e-01 2.37682393e+00 8.76362073e-01
     2.61848090e+00 4.51933182e+00 4.56046520e+00 2.68029534e+00
     1.15057556e+00 9.40311354e-02 2.22560579e+00 5.24666817e-01
     4.95930724e+00 2.40732438e+00 1.82856416e+00 6.10913459e-01
     6.01110664e+00 3.36639137e-01 2.07961994e+00 1.65603641e+00
     6.20103262e+00 5.29289192e+00 4.36040481e-01 5.62621427e-01
     8.38427122e-01 8.81433114e+00 6.89951210e-01 7.94181368e-01
     4.16547332e+00 4.01162361e+00 1.46764266e+00 5.51966374e+00]
    θ estimado: 2.8981, número de iteraciones: 1
    [3.93552715 0.25142286 8.83768591 ... 0.77591953 6.20672819 0.39429469]
    θ estimado: 2.9506, número de iteraciones: 55
    [ 3.78810595  2.50157617  0.15316288 ...  0.64669561  4.73346402
     10.73981884]
    θ estimado: 3.2099, número de iteraciones: 37
    [3.1627792  3.89367688 0.47819266 ... 1.53435481 0.95860456 0.44763044]
    θ estimado: 2.9626, número de iteraciones: 21
    [2.54067571 1.26117191 1.80370385 ... 1.47762588 3.36106849 0.50204537]
    θ estimado: 3.0000, número de iteraciones: 13
    [2.36015396 0.80111258 4.41926726 ... 3.19111717 8.07756814 0.57738601]
    θ estimado: 2.8533, número de iteraciones: 8
    [ 1.40581074  0.75615387  4.41498931 ...  4.67639703  2.44110786
     12.36757893]
    θ estimado: 2.9696, número de iteraciones: 6
    [4.27419064e-02 6.31017605e-01 1.58434801e+01 ... 5.42836779e+00
     2.58973973e+00 7.28700315e-03]
    θ estimado: 2.9223, número de iteraciones: 3
    [ 6.01105     1.69045512  1.06304591 ... 12.81989052  0.8107153
      3.17829244]
    θ estimado: 3.0077, número de iteraciones: 1
    [0.64578956 0.72502265 1.61563339 ... 0.75052149 0.56797187 2.86610933]
    θ estimado: 2.9844, número de iteraciones: 1
    [ 0.69817189  5.18826071 13.28683797 ...  2.38914696  3.62123341
      0.55279808]
    θ estimado: 2.9353, número de iteraciones: 1
    [ 5.41331677  8.70592986  0.66480725 ... 11.06960736  6.18804709
      1.47935313]
    θ estimado: 2.9697, número de iteraciones: 55
    [4.89927953 0.89270664 6.90901678 ... 2.06019311 5.88006245 2.31925294]
    θ estimado: 2.9664, número de iteraciones: 34
    [2.06730242 1.11412068 3.92597316 ... 2.99103246 0.73333866 2.41871159]
    θ estimado: 3.0440, número de iteraciones: 22
    [1.21785296 0.29901282 2.25315467 ... 4.44429237 2.52097946 1.23313282]
    θ estimado: 2.9248, número de iteraciones: 13
    [4.2104005  0.93834152 1.18964144 ... 0.28551631 4.31388351 0.1000749 ]
    θ estimado: 3.0054, número de iteraciones: 8
    [0.82237874 1.13142891 1.25450984 ... 0.45127017 0.30405545 2.77398493]
    θ estimado: 2.9884, número de iteraciones: 5
    [ 1.20865958  2.47757543 10.23067315 ...  0.09029303  0.59136684
      4.79464943]
    θ estimado: 3.0348, número de iteraciones: 4
    [4.47096661 4.4846884  2.8460066  ... 1.79558232 5.4429561  1.5496372 ]
    θ estimado: 3.0214, número de iteraciones: 1
    [1.04734543 3.56826615 4.4749105  ... 7.48616626 3.35325876 2.08898881]
    θ estimado: 3.0422, número de iteraciones: 1
    [1.90900749 4.17331924 0.79858653 ... 0.22350875 1.2848515  2.87384287]
    θ estimado: 2.9665, número de iteraciones: 1





    Text(0.5, 1.0, '$\\hat{\\theta}$ vs valor de a')




    
![png](Tarea4_files/Tarea4_10_2.png)
    



```python
min(x)
```




    3.0




```python

```




    3200




```python

```
