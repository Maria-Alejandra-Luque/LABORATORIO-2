Codigo :https://colab.research.google.com/drive/1DL1DXChiE5t4KtSDhACU42GRzzz_fLNJ?usp=sharing
# PROCESAMIENTO DIGITAL DE SEÑALES
# OBJETIVOS
- Comprender la convolución como una operación que permite obtener la respuesta de un sistema discreto ante una entrada determinada.
- Analizar la correlación como medida de similitud entre dos señales.
- Aplicar la Transformada de Fourier como herramienta de análisis en el dominio de la frecuencia. 
# DESCRIPCIÓN
En este repositorio encontraremos el analisis de la practica 2, donde se realizó la convolución, la correlación y la Transformada de Fourier para trabajar con diferentes señales. Inicialmente se desarrolló el ejercicio de manera manual y luego se trabajó en Python para entender cómo se comportan las señales cuando pasan por un sistema.  <br>

Además, con ayuda del generador se señales obtuvimos una señal biológica, la digitalizamos y analizamos sus comportamiento en el tiempo como en la frecuencia. Con el objetivo de observar de una manera más clara cómo se pueden comparar señales y también como cambia su información al analizarlas en otro dominio. Esta práctica permitió integrar la teoría y la programación y entender cómo se procesan y analizan señales tanto en el tiempo como en la frecuencia.
 
# PARTE A
## Convolución de Señales 
En esta sección realizamos la convolución entre una señal formada con los dígitos de la cédula de cada estudiante, y un sistema definido a partir de los dígitos de código universitarios.<br>
 <img width="380" height="287" alt="image" src="https://github.com/user-attachments/assets/b142d1b2-1d55-45f4-b9c7-4abe2252843c" /> <br>

1. Primero se realizo la convolución de manera manual, con los metodos aprendidos en la clase teórica.<br>
2.	Luego se representó gráficamente la señal obtenida de forma manual.<br>
3. Posteriormente, se programó la convolución en Python, verificando que el resultado coincidiera con el cálculo manual y se generaron las gráficas en Python para comparar y confirmar los resultados.<br>
### Manuela Mancera Herrera
Código= 5600874<br>
Cédula=1014187867 <br>

1.Convolución Manual <br>
  <img width="361" height="352" alt="image" src="https://github.com/user-attachments/assets/aeb78253-ce68-40e2-bd6e-6402411d470d" /><br>

2.Gráfica Manual<br>
    <img width="283" height="348" alt="image" src="https://github.com/user-attachments/assets/76bc5b58-999e-426f-adad-e50666822c93" /> <br>

3. Phyton<br>

```
import numpy as np
import matplotlib.pyplot as plt

 --- Datos Manu---
h = [5,6,0,0,8,7,4]                # sistema
x = [1,0,1,4,1,8,7,8,6,7]          # entrada

 --- Convolución ---
y = np.convolve(x, h)
print("y[n] =", y.tolist())

 --- Gráfica tipo stem ---
n = np.arange(len(y))
plt.figure(figsize=(10,4))

 Capturamos los objetos del stem
markerline, stemlines, baseline = plt.stem(n, y)
```
 <br><br>

 Este código calcula la convolución entre una señal de entrada (cedula) y un sistema (Codigo), imprime el resultado y lo grafica como señal discreta tipo stem. <br>
   
   ```
   Cambiamos los colores
plt.setp(markerline, color='rosybrown')  # puntos
plt.setp(stemlines, color='rosybrown')   # palitos
plt.setp(baseline, color='rosybrown')    # línea base

plt.title('Convolución y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()

 --- Tabla secuencial ---
for i, val in enumerate(y):
    print(f"n={i}, y[{i}]={val}")
  ```
   
   Este bloque permite personalizar la gráfica (colores, título, ejes, cuadrícula) y genera una tabla con los valores de la señal resultante.<br>
   
   <img width="523" height="239" alt="image" src="https://github.com/user-attachments/assets/16c5497f-962d-4e46-aeb7-fcd1f08d5e7b" /> <br>
   <img width="86" height="170" alt="image" src="https://github.com/user-attachments/assets/ec5c37d0-58c7-4516-abde-6345429311c9" /><br>
   Aca obtenemos nuestra señal convolucionada y la grafica, donde podemos confirmar que son identicas, es decir los calculos manuales son correctos. <br>

   ## Alejandra Martinez Luque
   Código=5600854 <br>
   Cédula= 1078367229<br>

   1.Convolución Manual <br>
  <img width="224" height="280" alt="image" src="https://github.com/user-attachments/assets/99085e2b-4711-4186-908f-85f379b58684" /> <br>

   2.Gráfica Manual<br>
   <img width="272" height="351" alt="image" src="https://github.com/user-attachments/assets/e0b83d80-4f06-4b7e-b13b-73cf8b1e720a" /><br>

   3. Phyton<br>
   ```
import numpy as np
import matplotlib.pyplot as plt

--- Datos Aleja ---
h = [5,6,0,0,8,5,4]                # sistema
x = [1,0,7,8,3,6,7,2,2,9]          # entrada

 --- Convolución ---
y = np.convolve(x, h)
print("y[n] =", y.tolist())

 --- Gráfica  ---
n = np.arange(len(y))
plt.figure(figsize=(10,4))

 Capturamos los objetos del stem
markerline, stemlines, baseline = plt.stem(n, y)

 Cambiamos los colores
plt.setp(markerline, color='aquamarine')  # puntos
plt.setp(stemlines, color='aquamarine')   # palitos
plt.setp(baseline, color='aquamarine')    # línea base

plt.title('Convolución y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()

--- Tabla secuencial ---
for i, val in enumerate(y):
    print(f"n={i}, y[{i}]={val}")
```
   <img width="551" height="291" alt="image" src="https://github.com/user-attachments/assets/6545fc3e-3ff1-4f90-b701-7915d707b23d" /> <br>
   <img width="94" height="173" alt="image" src="https://github.com/user-attachments/assets/9cc8a3f2-8b3a-4cd6-a21d-9c859db84cb4" /> <br>

## Valentina Gomez Fandiño
Código=5600867<br>
Cédula= 1007467467<br>
   1.Convolución Manual <br>
   
   <img width="215" height="269" alt="image" src="https://github.com/user-attachments/assets/bf628c23-2bd7-431c-bc21-0fa2cbbbcdbd" /> <br>
   2.Gráfica Manual<br>
   
   <img width="278" height="391" alt="image" src="https://github.com/user-attachments/assets/a494a268-42b8-4c5c-9a75-5d5d3296f3fe" /><br>
   
   3. Phyton<br>
```
import numpy as np
import matplotlib.pyplot as plt

--- Datos Vale ---
h = [5,6,0,0,8,6,7]                # sistema
x = [1,0,0,7,4,6,7,4,6,7]          # entrada

 --- Convolución ---
y = np.convolve(x, h)
print("y[n] =", y.tolist())

 --- Gráfica tipo stem ---
n = np.arange(len(y))
plt.figure(figsize=(10,4))

 Capturamos los objetos del stem
markerline, stemlines, baseline = plt.stem(n, y)

 Cambiamos los colores
plt.setp(markerline, color='PaleGreen')  # puntos
plt.setp(stemlines, color='PaleGreen')   # palitos
plt.setp(baseline, color='PaleGreen')    # línea base

plt.title('Convolución y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()

 --- Tabla secuencial ---
for i, val in enumerate(y):
    print(f"n={i}, y[{i}]={val}")
```
   <img width="523" height="241" alt="image" src="https://github.com/user-attachments/assets/c2bd51fc-8aa6-4959-8a98-8f10e394489b" /><br>
   <img width="89" height="170" alt="image" src="https://github.com/user-attachments/assets/04b8d004-d85c-485a-a906-0e6f8da2bb90" /> <br>

   # PARTE B
   ## Correlación de señales
   En la Parte B de la práctica utilizamos dos señales periódicas, un coseno y un seno, para comprender cómo funciona la correlación cruzada. Esta operación nos ayuda a comparar dos señales y ver qué tanta similitud tiene o cómo se relacionan entre sí.<br> 
   <img width="437" height="323" alt="image" src="https://github.com/user-attachments/assets/40ca7ee7-a36a-4d19-bc51-cb5f1d229fab" /> <br>

   ### Código
```
import numpy as np
import matplotlib.pyplot as plt

 Paso 1: Definir parámetros y n
Ts = 1.25e-3                # periodo de muestreo (s)
n = np.arange(0,9)          # n = 0..8

 Paso 2: Definir señales
x1 = np.cos(2*np.pi*100*n*Ts)   # cos(pi/4 * n)
x2 = np.sin(2*np.pi*100*n*Ts)   # sin(pi/4 * n)

 Mostrar valores
print("n =", n.tolist())
print("x1 =", np.round(x1,4).tolist())
print("x2 =", np.round(x2,4).tolist())
```
<br>
   Este bloque define las señales de entrada (coseno y seno), las evalúa en los primeros 9 puntos y muestra sus valores.<br>
 
   ```
    Paso 3: Calcular correlación cruzada
corr = np.correlate(x1, x2, mode='full')
lags = np.arange(-len(x1)+1, len(x1))  # lags de -(N-1) a +(N-1)

 Normalizar (opcional)
corr_norm = corr / np.max(np.abs(corr))

Mostrar correlación y lags
print("lags =", lags.tolist())
print("corr =", np.round(corr,4).tolist())
print("corr_norm =", np.round(corr_norm,4).tolist())

Paso 4: Encontrar el lag del pico
peak_idx = np.argmax(np.abs(corr))
peak_lag = lags[peak_idx]
peak_value = corr[peak_idx]
print(f"Pico absoluto en lag = {peak_lag}, valor = {peak_value:.4f}")
  ```
  <br>
   Este bloque calcula la correlación cruzada de las dos señales, obtiene todos los valores de desplazamiento posibles, normaliza los resultados, los muestra y además identifica el pico principal, que indica dónde hay mayor similitud entre las señales.<br>

  ```
 Paso 5: Graficar señales y correlación

 Gráfica x1
plt.figure(figsize=(8,3))
markerline, stemlines, baseline = plt.stem(n, x1)
plt.setp(markerline, color='plum')
plt.setp(stemlines, color='plum')
plt.setp(baseline, color='plum')
plt.title('x1[n] = cos(2π·100·n·Ts)')
plt.xlabel('n'); plt.ylabel('x1[n]')
plt.grid(True)
plt.show()

 Gráfica x2
plt.figure(figsize=(8,3))
markerline, stemlines, baseline = plt.stem(n, x2)
plt.setp(markerline, color='plum')
plt.setp(stemlines, color='plum')
plt.setp(baseline, color='plum')
plt.title('x2[n] = sin(2π·100·n·Ts)')
plt.xlabel('n'); plt.ylabel('x2[n]')
plt.grid(True)
plt.show()
  ```
  <br>
   Este bloque genera dos gráficas separadas, una para la señal coseno y otra para la señal seno, ambas en formato discreto (stem), con títulos y ejes identificados.<br>

```
Gráfica correlación (no normalizada)
plt.figure(figsize=(8,3))
markerline, stemlines, baseline = plt.stem(lags, corr)
plt.setp(markerline, color='plum')
plt.setp(stemlines, color='plum')
plt.setp(baseline, color='plum')
plt.title('Correlación cruzada r_{x1,x2}[lag]')
plt.xlabel('lag'); plt.ylabel('r[lag]')
plt.grid(True)
plt.show()

 Gráfica correlación normalizada
plt.figure(figsize=(8,3))
markerline, stemlines, baseline = plt.stem(lags, corr_norm)
plt.setp(markerline, color='plum')
plt.setp(stemlines, color='plum')
plt.setp(baseline, color='plum')
plt.title('Correlación cruzada normalizada')
plt.xlabel('lag'); plt.ylabel('r_norm[lag]')
plt.grid(True)
plt.show()
```
   <br>

   ## Resultado
   <img width="637" height="73" alt="image" src="https://github.com/user-attachments/assets/d4424f0d-2a70-4920-80a7-d0bbb19796a7" /><br>
   <img width="440" height="184" alt="image" src="https://github.com/user-attachments/assets/f30ec381-4711-4747-8705-1d324f5c55ab" /><br>
   <img width="454" height="179" alt="image" src="https://github.com/user-attachments/assets/912a0afd-a04c-48f1-b129-af969b52d644" /><br>
   <img width="429" height="185" alt="image" src="https://github.com/user-attachments/assets/099bc8b2-8944-4024-82ca-ca71b52a0b26" /><br>
   <img width="454" height="182" alt="image" src="https://github.com/user-attachments/assets/bac8e7af-fb9e-404d-9bba-49676791fc35" /><br>

   Al analizar las gráficas de la correlación se observa que el coseno y el seno no son iguales, pero sí tienen una relación clara cuando una señal se desplaza respecto a la otra. El pico más grande aparece en el lag 2 y con valor negativo, lo que confirma que están desfasadas. En la gráfica normalizada se ve mejor esta relación porque los valores están entre -1 y 1. En conclusión, la correlación nos permitió entender que aunque las señales no coinciden directamente, sí guardan una similitud marcada por el desfase que tienen. <br>
   # PARTE C
   Por ultimo el objetivo de la parte C es caracterizar una señal digitalizada tanto en el dominio del tiempo como en el de la frecuencia, aplicando técnicas de procesamiento digital de señales (DSP). El análisis temporal permite observar la forma de onda y su comportamiento dinámico, mientras que el análisis frecuencial permite identificar los componentes espectrales y la distribución de energía de la señal.
   <img width="1024" height="768" alt="Diagrama de Flujo Árbol de decisiones Sencillo Verde" src="https://github.com/user-attachments/assets/ff57c1fd-1149-46b9-a778-15a57f297c72" />

   ## ADQUISICIÓN DE LA SEÑAL
   La señal la adquirimos utilizando el sistema de adquisición de datos (DAQ), configurado con una frecuencia de muestreo de 400 Hz para asegurar una correcta digitalización. Para esto usamos el entorno de programación Spyder, desde donde establecimos la conexión con el dispositivo, al capturar los valores de voltaje en función del tiempo y se guaradaron en un archivo .csv. Este archivo contiene dos columnas: Tiempo (s) y Voltaje (V), que fueron la base para realizar el posterior procesamiento digital de la señal.
   ## CÓDIGO
   En la primera parte del código correspondió a la caracterización de la señal en el dominio del tiempo. Para esto se cargo directamente el archivo .csv obtenido con el DAQ y realizamos su representación gráfica, lo que nos permitió identificar de manera inicial las oscilaciones y el comportamiento dinámico del voltaje. 
 ```
     ### CARACTERIZACION DE LA SEÑAL

import pandas as pd
import numpy as np           # para usar np.mean, np.median...
import matplotlib.pyplot as plt

# si está separado por comas
df = pd.read_csv("/content/drive/MyDrive/senal_daq_400fm.csv", sep=",")

print(df.head())      # muestra las primeras filas
print(df.columns)     # muestra los nombres de columnas

# Graficar en azul pastel
plt.plot(df['Tiempo (s)'], df['Voltaje (V)'], color='#AEC6CF')  # 🔹 azul pastel
plt.title("Señal digitalizada ")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid()
plt.show()
```
   ## GRÁFICA 
Al obtener los datos se hizo la gráfica de la señal en el dominio del tiempo, representando el tiempo en el eje horizontal y el voltaje en el eje vertical. La curva fue visualizada, lo que permitió observar de manera clara las oscilaciones, amplitud y comportamiento general de la señal digitalizada.

<img width="749" height="731" alt="image" src="https://github.com/user-attachments/assets/32d9f134-f821-4744-8a7d-b9dcff4a95fe" /><br>
  # CARACTERIZACIÓN TEMPORAL 
En esta parte del código se cargó la señal desde el archivo .csv, se extrajeron las columnas de tiempo y voltaje y se verificó la correcta lectura de los datos con head() y columns. Posteriormente, la señal fue representada en el dominio temporal mediante una gráfica, lo que permitió observar su comportamiento inicial y confirmar que estaba lista para los análisis posteriores.
 ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la señal
df = pd.read_csv("/content/drive/MyDrive/senal_daq_400fm.csv")
# 2. Extraer datos
t = df["Tiempo (s)"].values
x = df["Voltaje (V)"].values   # Cambia a "Voltaje (V)" si tu archivo está en voltios

print(df.head())      # muestra las primeras filas
print(df.columns)     # muestra los nombres de columnas

# Graficar señal en fucsia
plt.plot(df['Tiempo (s)'], df['Voltaje (V)'], color='#FF00FF')
plt.title("Señal digitalizada ")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid()
plt.show()
```
## FRECUENCIA DE MUESTREO 
En esta sección también se calculó la frecuencia de muestreo a partir de la diferencia entre dos muestras consecutivas de tiempo. Esto es esencial ya que permitió conocer con precisión la tasa a la cual fue digitalizada la señal, necesaria para aplicar correctamente la Transformada de Fourier y realizar el análisis en el dominio de la frecuencia.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 ```
# 3. Calcular frecuencia de muestreo
dt = t[1] - t[0]   # diferencia entre muestras
fs = 1 / dt        # frecuencia de muestreo
N = len(x)         # número de muestras
```
 ## GRÁFICA 
 <img width="752" height="724" alt="image" src="https://github.com/user-attachments/assets/f344e521-6a8f-45c0-ba4b-a09506511935" />

 # TRANSFORMADA DE FOURIER
 Posteriormente, se aplicó la Transformada Rápida de Fourier (FFT) con el fin de pasar la señal del dominio del tiempo al dominio de la frecuencia. Este procedimiento permite identificar los componentes espectrales que conforman la señal y reconocer qué frecuencias predominan en ella. En este caso, el espectro de magnitud reveló los picos correspondientes a las frecuencias más significativas.
```
 # Transformada de Fourier
X = np.fft.fft(x) / N
freqs = np.fft.fftfreq(N, d=dt)
PSD = (1/(fs*N)) * (np.abs(X)**2)
PSD[1:-1] *= 2

# Nos quedamos solo con la mitad positiva
mask = freqs >= 0
freqs = freqs[mask]
X = X[mask]
X_mag = np.abs(X)
PSD = PSD[mask]   # ✅ corrección para evitar error de dimensiones

# 5. Estadísticos en el dominio de la frecuencia
PSD_norm = PSD / np.sum(PSD)  # normalizamos para usarlo como distribución
f_mean = np.sum(freqs * PSD_norm)
f_median = freqs[np.cumsum(PSD_norm) >= 0.5][0]
f_std = np.sqrt(np.sum(((freqs - f_mean)**2) * PSD_norm))

# 6. Graficar
## a) Transformada de la señal en fucsia
plt.figure(figsize=(10,4))
plt.plot(freqs, X_mag, color='#FF00FF')
plt.title("Transformada de Fourier (Magnitud)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()
```
 ## GRÁFICA 
<img width="747" height="341" alt="image" src="https://github.com/user-attachments/assets/9a4620c0-dfab-4b13-b407-bd253386a2be" />

# DENSIDAD ESPECTRAL DE POTENCIA
La densidad espectral de potencia (PSD) fue calculada a partir de la transformada de Fourier, con el fin de conocer cómo se distribuye la potencia de la señal entre las distintas frecuencias. Este análisis permite observar en qué rango se concentra la mayor energía de la señal, lo cual es esencial en aplicaciones de filtrado o caracterización espectral.
```
# Desidad espectral de potencia
PSD = (1/(fs*N)) * (np.abs(X)**2)
PSD[1:-1] *= 2

## b) Densidad espectral de potencia en fucsia
plt.figure(figsize=(10,4))
plt.semilogy(freqs, PSD, color='#FF00FF')
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V^2/Hz]")
plt.grid()
plt.show()
```
 ## GRÁFICA 
<img width="749" height="344" alt="image" src="https://github.com/user-attachments/assets/e310b848-7b4e-4301-9924-ee9923670335" />

# HISTOGRAMA DE FRECUENCIA
A partir de la normalización de la PSD se construyó un histograma de frecuencias, el cual muestra la probabilidad de aparición de los distintos componentes espectrales de la señal. Esto nos permite analizar de manera más intuitiva la distribución de la energía, resaltando rangos son más frecuentes en el espectro.
```
## c.iv) Histograma de frecuencias en fucsia
plt.figure(figsize=(8,4))
plt.hist(freqs, bins=50, weights=PSD_norm, color='#FF00FF')
plt.title("Histograma de Frecuencias (ponderado por PSD)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Probabilidad")
plt.grid()
plt.show()
```
 ## GRÁFICA 
<img width="738" height="422" alt="image" src="https://github.com/user-attachments/assets/c4e6a2dd-4976-4a09-98d9-982964ed2e3a" />

# ESTADÍSTICOS EN EL DOMINIO DE LA FRECUENCIA 
Finalmente, se calcularon algunos parámetros estadísticos que describen el comportamiento espectral de la señal.
La frecuencia media: Centro de gravedad de la distribución espectral
La frecuencia mediana: El punto donde el espectro se divide en dos partes iguales de energía
La desviación estándar: La dispersión de las frecuencias alrededor de la media
```
#Cálculos de estadísticos
f_mean = np.sum(freqs * PSD_norm)
f_median = freqs[np.cumsum(PSD_norm) >= 0.5][0]
f_std = np.sqrt(np.sum(((freqs - f_mean)**2) * PSD_norm))

# 7. Imprimir resultados estadísticos
print("📊 Estadísticos en el dominio de la frecuencia:")
print(f"Frecuencia media: {f_mean:.2f} Hz")
print(f"Frecuencia mediana: {f_median:.2f} Hz")
print(f"Desviación estándar: {f_std:.2f} Hz")
```
## Resultados obtenidos 
<img width="501" height="100" alt="image" src="https://github.com/user-attachments/assets/e7124aad-b896-4875-b917-874b85caa324" />












  



   


   




