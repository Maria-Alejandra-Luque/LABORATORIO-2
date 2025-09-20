# LABORATORIO-2

# DESCRIPCIÓN
En este repositorio encontraremos el analisis de la practica 2, donde se realizó la convolución, la correlación y la Transformada de Fourier para trabajar con diferentes señales. Inicialmente se desarrolló el ejercicio de manera manual y luego se trabajó en  Python para entender cómo se comportan las señales cuando pasan por un sistema.  <br>

Además, con ayuda del generador se señales obtuvimos una señal biológica, la digitalizamos y analizamos sus comportamiento en el tiempo como en la frecuencia. Con el objetivo de observar de una manera más clara cómo se pueden comparar señales y también como cambia su información al analizarlas en otro dominio. Esta práctica permitió integrar la teoría y la programación y entender cómo se procesan y analizan señales tanto en el tiempo como en la frecuencia.

# PARTE A
## Convolución de Señales 
En esta sección realizamos la convolución entre una señal formada con los dígitos de la cédula de cada estudiante, y un sistema definido a partir de los dígitos de código universitarios.<br>
 <img width="380" height="287" alt="image" src="https://github.com/user-attachments/assets/b142d1b2-1d55-45f4-b9c7-4abe2252843c" /> <br>

1. primero se realizo la convolución de manera manual, con los metodos aprendidos en la clase teorica.<br>
2.	Luego se representó gráficamente la señal obtenida de forma manual.<br>
3. Posteriormente, se programó la convolución en Python, verificando que el resultado coincidiera con el cálculo manual y se generaron las gráficas en Python para comparar y confirmar los resultados.<br>
### Manuela Mancera Herrera
Codigo= 5600874<br>
Cedula=1014187867 <br>

1.Convolución Manual <br>
  <img width="361" height="352" alt="image" src="https://github.com/user-attachments/assets/aeb78253-ce68-40e2-bd6e-6402411d470d" /><br>

2.Grafica Manual<br>
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

   este código calcula la convolución entre una señal de entrada (cedula) y un sistema (Codigo), imprime el resultado y lo grafica como señal discreta tipo stem. <br>
   
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
   
   este bloque permite personalizar la gráfica (colores, título, ejes, cuadrícula) y genera una tabla con los valores de la señal resultante.<br>
   
   <img width="523" height="239" alt="image" src="https://github.com/user-attachments/assets/16c5497f-962d-4e46-aeb7-fcd1f08d5e7b" /> <br>
   <img width="86" height="170" alt="image" src="https://github.com/user-attachments/assets/ec5c37d0-58c7-4516-abde-6345429311c9" /><br>
   Aca obtenemos nuestra señal convolucionada y la grafica, donde podemos confirmar que son identicas, es decir los calculos manuales son correctos. <br>

   ## Alejandra Martinez Luque
   Codigo=5600854 <br>
   Cedula= 1078367229<br>

   1.Convolución Manual <br>
  <img width="224" height="280" alt="image" src="https://github.com/user-attachments/assets/99085e2b-4711-4186-908f-85f379b58684" /> <br>

   2.Grafica Manual<br>
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
codigo=5600867<br>
Cedula= 1007467467<br>
   1.Convolución Manual <br>
   
   <img width="215" height="269" alt="image" src="https://github.com/user-attachments/assets/bf628c23-2bd7-431c-bc21-0fa2cbbbcdbd" /> <br>
   2.Grafica Manual<br>
   
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

   ### Codigo
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
   este bloque define las señales de entrada (coseno y seno), las evalúa en los primeros 9 puntos y muestra sus valores.<br>
 
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
   











  



   


   




