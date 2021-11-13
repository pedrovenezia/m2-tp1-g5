import pandas as pd 
import numpy as np 

class Calculadora_TL():
    '''clase para calcular tl de diversos materiales con diferentes métodoss
    :input
        data_path = str. Data path
        t = espesor
        l1 = float, alto. mts
        l2 = float, largo. mts'''

    def __init__(self, data_path, t=None, l1=None, l2=None):
        self.data_path = data_path
        # Constantes
        self.c = 343     # Velocidad del sonido en el aire
        self.rho_0 = 1.18    # Densidad del aire
        self.load_data()  
        self.t = t
        self.l1 = l1
        self.l2 = l2
        self.f = np.array([20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,
     500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,
     6300,8000,10000,12500,16000,20000])

    def load_data(self):
        '''Carga de datos'''
        excel = pd.read_excel('TABLA MATERIALES TP1.xlsx', header=None).drop(columns = [0, 1])[1::] #slide indexing 
        new_header = ['material', 'rho', 'E', 'nint', 'sigma']
        excel = excel[1:].reset_index(drop=True) #take the data less the header row
        excel.columns = new_header #set the header row as the df header
        self.data = excel

    def parametros(self,rho,E,sigma):
        m = self.t*rho # Masa superficial
        B = E * self.t ** 3 / (12 * (1-sigma ** 2))
        fc = self.c ** 2 / (2 * np.pi) * np.sqrt(m/B)    # Frecuencia crítica
        fd = E / (2 * np.pi * rho) * np.sqrt(m/B)   # Frecuencia de densidad
        return m, B, fc, fd

    #vectorizar 
    def ley_masa(self, m, nint, fc, fd):
        R = np.zeros(len(self.f))    # Vector de pérdidas por transmisión
        for i in range(len(self.f)):
            if (self.f[i] < fc) or (self.f[i] > fd):
                R[i] = 20*np.log10(m*self.f[i]) - 47
            elif self.f[i] > fc and self.f[i] < fd:
                n = nint + m / (485 * np.sqrt(self.f[i]))    # Factor de pérdidas total
                R[i] = 20*np.log10(m*self.f[i]) - 10*np.log10(np.pi / (4*n)) - 10*np.log10(fc/(self.f[i]-fc)) - 47               
        return R

# SHARP 
    def sharp(self,m,nint,fc):
        R = np.zeros(len(self.f))    
        for i in range(len(self.f)):                    
            if self.f[i]<(0.5*fc): 
                R[i]= 10*np.log10(1+(((np.pi*m*self.f[i])/(self.rho_0*self.c))**2))-5.5;                       
            elif self.f[i]>=fc:
                ntotal=(nint)+(m/(485*np.sqrt(self.f[i])))
                R22 = 10*np.log10(1+(((np.pi*m*self.f[i])/(self.rho_0*self.c))**2))+10*np.log10((2*ntotal*self.f[i])/(np.pi*fc))
                R21 = (10*np.log10(1+((np.pi*m*self.f[i])/(self.rho_0*self.c))**2))-(5.5)
                R[i]= min(R21,R22); #Toma el minimo entre R21 y R22            
            elif (0.5*fc)<=self.f[i] and self.f[i]<fc:
                ntotal = (nint)+(m/(485*np.sqrt(self.f[i]))); 
                Rx = 10*np.log10(1+((np.pi*m*self.f[i])/(self.rho_0*self.c))**2) + 10*np.log10((2*ntotal*self.f[i])/(np.pi*fc))
                Ry = 10*np.log10(1+((np.pi*m*self.f[i])/(self.rho_0*self.c))**2) - 5.5
                R[i] =(((self.f[i]-0.5*fc)/(fc-0.5*fc))*(Rx-Ry))+Ry;           
        return R


# ISO 12354-1 OK!!!!
    def ISO(self,l1,l2, fc, m, nint):
        n = nint + m / (485*np.sqrt(self.f)) # Factor de pérdidas total
        # Factor de radiación de ondas forzadas
        k = 2*np.pi*self.f / self.c   # Nro de onda
        h = 5*l2/(2*np.pi*l1) - 1/(4*np.pi*l1*l2*k**2)
        A = -0.964 - (0.5 + l2/(np.pi*l1))* np.log(l2/l1) + h
        sigma_f = 0.5*(np.log(k*np.sqrt(l1*l2)) - A)
        for i in range(len(sigma_f)):
          if sigma_f[i] > 2:
            sigma_f[i] = 2
        # Factor de radiación de ondas libres
        sigma_1 = 1/(np.sqrt(abs(1-fc/self.f)))
        sigma_2 = 4*l1*l2*(self.f/self.c)**2
        sigma_3 = np.sqrt(2*np.pi*self.f*(l1+l2)/(16*self.c))
        # Modo de resonancia de placa 1,1
        f11 = self.c**2/(4*fc)*(1/l1**2 + 1/l2**2)
        tau =np.zeros(len(self.f))
        if f11 <= 0.5*fc:
            for i in range(len(self.f)):
                if self.f[i] >= fc:
                    sigma = sigma_1[i]
                    if sigma > 2: sigma = 2
                    tau[i] = (2*self.rho_0*self.c/(2*np.pi*self.f[i]*m))**2 * np.pi*fc*sigma**2/(2*self.f[i]*n[i])
                if self.f[i] < fc:
                    lamda = np.sqrt(self.f[i]/fc)
                if self.f[i] > 0.5*fc:
                    d2 = 0
                else:
                    d2 = 8*self.c**2*(1-2*lamda**2) / (fc**2*np.pi**4*l1*l2*lamda*np.sqrt(1-lamda**2))
                d1 = ((1-lamda**2)*np.log((1+lamda)/(1-lamda))+ 2*lamda) / (4*np.pi**2*(1-lamda**2)**1.5)
                sigma = 2*(l1+l2)*self.c*d1/(l1*l2*fc)+d2
                if (self.f[i] < f11) and (sigma > sigma_2[i]):
                    sigma = sigma_2[i]
                if sigma > 2: sigma = 2
                tau[i] = (2*self.rho_0*self.c/(2*np.pi*self.f[i]*m))**2 * (2*sigma_f[i] + (l1+l2)**2*np.sqrt(fc/self.f[i])*sigma**2/((l1**2+l2**2)*n[i])) 
        else:
            for i in range(len(self.f)):
                sigma = sigma_3[i]
                if (self.f[i] < fc) and (sigma_2[i] < sigma):
                    sigma = sigma_2[i]
                if sigma > 2: sigma = 2
                tau[i] = (2*self.rho_0*self.c/(2*np.pi*self.f[i]*m))**2 * (2*sigma_f[i] + (l1+l2)**2*np.sqrt(fc/self.f[i])*sigma**2/((l1**2+l2**2)*n[i]))
                if (self.f[i] >= fc) and (sigma_1[i] < sigma):
                    sigma = sigma_1[i]
                if sigma > 2: sigma = 2
                tau[i] = (2*self.rho_0*self.c/(2*np.pi*self.f[i]*m))**2 * np.pi*fc*sigma**2/(2*self.f[i]*n[i])
        R = -10*np.log10(abs(tau))
        return R

            ###DAVY (ok)
    def davy(self,fc,m,nint,rho,E,sigma):
        averages = 3 # % promedio definido por Davy
        dB = 0.236
        octave = 3
        R = np.zeros(len(self.f))
        #Avsingle_leaf = 0
        for i in range(len(self.f)):
            n = nint + (m/(485*np.sqrt(self.f[i])))
            ratio = self.f[i]/fc
            limit = 2 ** (1/(2*octave))
            if (ratio<1/limit) or (ratio>limit):
                TLost = self.Single_leaf_Davy(self.f[i],rho,E,sigma,self.t,n,self.l2,self.l1)
            else:
                Avsingle_leaf = 0
                for j in range(1,averages+1):
                  factor = 2**((2*j-1-averages)/(2*averages*octave))
                  aux = 10**(-self.Single_leaf_Davy(self.f[i]*factor,rho,E,sigma,self.t,n,self.l2,self.l1)/10)
                  Avsingle_leaf += aux
                TLost = -10*np.log10(Avsingle_leaf/averages)
            R[i] = TLost
        return R


    def Single_leaf_Davy(self, frequency, density, Young, Poisson, thickness, lossfactor, length, width):
        cos21Max = 0.9 #Ángulo limite definido en el trabajo de Davy
        surface_density = density * thickness
        critical_frequency = np.sqrt(12 * density * (1 - Poisson ** 2) / Young) * self.c ** 2 / (2 * thickness * np.pi)
        normal = self.rho_0 * self.c / (np.pi * frequency * surface_density)
        normal2 = normal * normal
        e = 2 * length * width / (length + width)
        cos2l = self.c / (2 * np.pi * frequency * e)
        if cos2l > cos21Max:
            cos2l = cos21Max
        tau1 = normal2 * np.log((normal2 + 1) / (normal2 + cos2l)) #Con logaritmo en base e (ln)
        ratio = frequency / critical_frequency
        r = 1 - 1 / ratio
        if r < 0:
            r = 0
        G = np.sqrt(r)
        rad = self.Sigma(G, frequency, length, width)
        rad2 = rad * rad
        netatotal = lossfactor + rad * normal
        z = 2 / netatotal
        y = np.arctan(z) - np.arctan(z * (1 - ratio))
        tau2 = normal2 * rad2 * y / (netatotal * 2 * ratio)
        tau2 = tau2 * self.shear(frequency, density, Young, Poisson, thickness)
        if frequency < critical_frequency:
            tau = tau1 + tau2
        else:
            tau = tau2 
        single_leaf = -10 * np.log10(tau)
        return single_leaf
        
    def Sigma(self, G, freq, width, length):
        w = 1.3
        beta = 0.234
        n = 2
        S = length * width
        U = 2 * (length + width)
        twoa = 4 * S / U
        k = 2 * np.pi * freq / self.c
        f = w * np.sqrt(np.pi / (k * twoa))
        if f > 1:
            f = 1
        h = 1 / (np.sqrt(k * twoa / np.pi) * 2 / 3 - beta)
        q = 2 * np.pi / (k * k * S)
        qn = q ** n
        
        if G < f:
            alpha = h / f - 1
            xn = (h - alpha * G) ** n
        else:
            xn = G ** n
        rad = (xn + qn)**(-1 / n)
        return rad

    def shear(self, frequency, density, Young, Poisson, thickness):
        omega = 2 * np.pi * frequency
        chi = (1 + Poisson) / (0.87 + 1.12 * Poisson)
        chi = chi * chi
        X = thickness * thickness / 12
        QP = Young / (1 - Poisson * Poisson)
        C = -omega * omega
        B = C * (1 + 2 * chi / (1 - Poisson)) * X
        A = X * QP / density
        kbcor2 = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
        kb2 = np.sqrt(-C / A)
        G = Young / (2 * (1 + Poisson))
        kT2 = -C * density * chi / G
        kL2 = -C * density / QP
        kS2 = kT2 + kL2
        ASI = 1 + X * (kbcor2 * kT2 / kL2 - kT2)
        ASI = ASI * ASI
        BSI = 1 - X * kT2 + kbcor2 * kS2 / (kb2 * kb2)
        CSI = np.sqrt(1 - X * kT2 + kS2 * kS2 / (4 * kb2 * kb2))
        out = ASI / (BSI * CSI)
        return out

    def _llama_metodo(self, metodo):
        if metodo=='sharp':
            return self.sharp 
        if metodo=='ley1':
            return self.ley_masa
        if metodo=='ISO':
            return self.ISO
        if metodo=='davy':
            return self.davy
    

    def calcular_r(self, material, metodo):
        """Fórmula para calcular r con los métodos definidos
            :inputs 
                material: str, nombre del material
                metodo: str,list nombre del metodo pueden ser varios en ese caso deben ingresarse
                los strings dentro de una lista 
            :outputs
                R = np.array. Array con r calculadas para métodos ingresado"""
        results = {}
        nint = self.data[self.data.material == material].nint.values[0]
        rho = self.data[self.data.material == material].rho.values[0]
        sigma = self.data[self.data.material == material].sigma.values[0]
        E = self.data[self.data.material == material].E.values[0]
        m, B, fc, fd = self.parametros(rho,E,sigma)
        for x in metodo:
            if x == 'ley1':
                funcion = self._llama_metodo(x)
                results[x] = funcion(m, nint, fc, fd)
            if x == 'sharp':
                funcion = self._llama_metodo(x)
                results[x] = funcion(m, nint, fc)
            if x == 'ISO':
                funcion = self._llama_metodo(x)
                results[x] = funcion(self.l1, self.l2, fc, m, nint)
            if x == 'davy':
                funcion = self._llama_metodo(x)
                results[x] = funcion(fc, m, nint, rho, E, sigma)
        return results 

if __name__ == '__main__':
    a = Calculadora_TL(data_path='TABLA MATERIALES TP1.xlsx', t=0.52, l1=3, l2=5)
    resultados = a.calcular_r('Hormigón', ['ley1', 'sharp', 'davy', 'ISO'])
    print(resultados)