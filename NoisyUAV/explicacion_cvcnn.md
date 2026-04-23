# Explicación Detallada: `cvcnn.py` — Complex-Valued CNN

## 0. Por qué una red de valores complejos

Una señal de radio en banda base se representa como **I/Q**:

```
s(t) = I(t) + j·Q(t)
```

Donde `I` (In-phase) y `Q` (Quadrature) son dos canales reales desfasados 90°.
Juntos forman un número complejo que codifica tanto la **amplitud** como la
**fase** de la señal en cada instante.

Un protocolo FHSS hace algo específico: cada hop salta a una frecuencia nueva y
la señal llega con una **rotación de fase** distinta. Si tomáramos solo el
módulo `|IQ| = sqrt(I²+Q²)`, perderíamos esa información de fase — es como
escuchar música en mono: pierdes toda la espacialidad.

La CV-CNN resuelve esto operando **directamente en el dominio complejo**,
aprendiendo a reconocer patrones tanto en amplitud como en fase simultáneamente.

---

## 1. `ComplexConv1d` — El bloque más importante

```python
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ...):
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, ...)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, ...)
```

### Multiplicación Compleja vs Cálculo de Wirtinger

Lo que ocurre en la capa *Forward* (hacia delante) es una simple **multiplicación de números complejos** de forma cartesiana. Un filtro complejo `W = W_re + j·W_im` aplicado a `z = x_re + j·x_im` produce:

```
Re(W * z) = W_re(x_re) − W_im(x_im)
Im(W * z) = W_re(x_im) + W_im(x_re)
```

En el código (líneas 118-119 del archivo), esto se ve así:

```python
out_re = self.conv_re(x_re) - self.conv_im(x_im)   # parte real de la salida
out_im = self.conv_re(x_im) + self.conv_im(x_re)   # parte imaginaria de la salida
```

**¿Y qué es la regla (o Cálculo) de Wirtinger?**
El **Cálculo de Wirtinger** NO se usa en el paso *Forward*. Se usa en el paso *Backward* (Retro-propagación). Sirve para poder calcular el **gradiente de funciones complejas** (que estrictamente no son derivables bajo las ecuaciones de Cauchy-Riemann). Permite a PyTorch derivar tratando a la variable compleja `z` y a su conjugado `z*` como dos variables reales independientes, haciendo posible entrenar la red mediante Descenso de Gradiente (Gradient Descent).

**¿Por qué dos `Conv1d` reales?** PyTorch no tiene soporte nativo para CNN
complejas. Se simula con dos convoluciones reales y cableando el cruce I↔Q.
El cruzamiento (`-` y `+`) **no es opcional**: es la matemática de la
multiplicación compleja. Sin él, la red trataría I y Q como canales
independientes, perdiendo la coherencia de fase.

### Forma de los tensores

El tensor tiene forma `[B, 2·C_in, N]`:
- Primeros `C_in` canales → parte real (I)
- Últimos `C_in` canales → parte imaginaria (Q)

Con 1 canal complejo de entrada: `[B, 2, N]` = `[batch, I+Q, tiempo]`

---

## 2. `ComplexBatchNorm1d` — Normalización para señales complejas

```python
class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_complex_channels):
        self.bn_re = nn.BatchNorm1d(num_complex_channels)
        self.bn_im = nn.BatchNorm1d(num_complex_channels)
```

Aplica Batch Normalization por separado a parte real e imaginaria.
Sin BN, las magnitudes en capas profundas se disparan o se hacen cero y la
red no converge. Existe una versión covariante compleja (Trabelsi 2018) más
correcta matemáticamente pero mucho más costosa e inestable. Para señales RF
la versión simplificada da resultados prácticamente iguales.

---

## 3. `CReLU` — Activación no lineal compleja

```python
class CReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)   # ReLU element-wise sobre todo el tensor [Re|Im]
```

Aplica ReLU por separado a Re e Im:

```
CReLU(z) = ReLU(Re(z)) + j·ReLU(Im(z))
```

Es la activación más estable para CVNNs en aplicaciones de señal (Bassey et
al., 2021 §4.2). Las alternativas más sofisticadas (zReLU, modReLU) no
mejoran significativamente en práctica y son más difíciles de entrenar.

---

## 4. `modulus` — El puente del dominio complejo al real

```python
def modulus(x):
    c = x.shape[1] // 2
    re, im = x[:, :c, :], x[:, c:, :]
    return torch.sqrt(re**2 + im**2 + 1e-8)
```

Calcula `|z| = sqrt(Re² + Im²)`. Se aplica **solo al final** de la parte
convolucional para colapsar de 2×128 canales complejos → 128 canales reales,
necesarios para el clasificador lineal. El `+ 1e-8` previene NaN en gradientes
cuando el módulo es casi cero.

---

## 5. `ComplexConvBlock` — El bloque constructivo

```python
def forward(self, x):
    return self.act(self.bn(self.conv(x)))   # Conv → BN → CReLU
```

El patrón **Conv → BN → Activación** es estándar desde VGG (2014). El orden
importa: BN antes de la activación normaliza las pre-activaciones, haciendo
que ReLU corte en un punto más estable.

---

## 6. `ComplexConv1DNet` — Arquitectura completa

```
Input: [B, 2, N]              ← N muestras I/Q, longitud variable
│
├─ ComplexConvBlock( 1→ 32, k=31, stride=2) → [B,  64, N/2]
├─ ComplexConvBlock(32→ 64, k=15, stride=2) → [B, 128, N/4]
├─ ComplexConvBlock(64→128, k= 7, stride=2) → [B, 256, N/8]
├─ ComplexConvBlock(128→128,k= 3, stride=1) → [B, 256, N/8]  ← sin reducción
│
├─ modulus()                                → [B, 128, N/8]  ← dominio real
├─ AdaptiveAvgPool1d(64)                    → [B, 128,  64]  ← longitud fija
│
├─ Flatten                                  → [B, 8192]
├─ Linear(8192→512) + ReLU + Dropout(0.4)
├─ Linear( 512→ 64) + ReLU
└─ Linear(  64→  2)                         → [B, 2]  logits [Ruido, Dron]
```

### ¿Por qué kernels grandes al principio (k=31, k=15)?

A 14 MHz, un kernel de tamaño 31 cubre `31/14e6 = 2.2 µs`. Necesitas ver
varios ciclos de la portadora para detectar la fase coherentemente. Con k=3
al principio, el campo receptivo es demasiado pequeño para capturar la
estructura del hop FHSS. Los kernels se reducen (31→15→7→3) porque en capas
profundas las features ya son abstractas.

### ¿Por qué stride=2 en las 3 primeras capas?

Cada stride=2 divide la longitud temporal a la mitad (decimación):

```
N muestras → N/2 → N/4 → N/8
```

Comprime la secuencia temporal reduciendo coste computacional. Es el equivalente
digital de un filtro paso-baja + decimación, exactamente lo que hace el hardware
de radio al hacer downsampling.

### ¿Por qué `AdaptiveAvgPool1d(64)`?

Los bursts FHSS tienen longitudes variables. `AdaptiveAvgPool1d(64)` comprime
**cualquier** longitud temporal a exactamente 64 posiciones haciendo promedio
adaptativo. Permite que el clasificador lineal opere con dimensión fija
independientemente de la duración de cada burst.

### ¿Por qué Dropout(0.4)?

El 40% es agresivo frente al estándar (0.2-0.3). Justificación: el dataset
tiene **label noise** — archivos de ruido que contienen WiFi/BT parciales, y
archivos de drones con contaminación. El dropout fuerza representaciones
robustas que no dependan de neuronas individuales.

---

## 7. Inicialización de pesos

```python
nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # Conv1d
nn.init.xavier_normal_(m.weight)                          # Linear
```

**Kaiming (He)**: diseñado para capas con ReLU. Escala los pesos para que
la varianza de las activaciones sea constante a lo largo de las capas. Sin
esto, las capas profundas dan activaciones en cero o infinito y la red no
aprende.

**Xavier**: similar pero asume activaciones simétricas. Adecuado para las
capas lineales al final donde no hay CReLU.

---

## 8. Flujo completo de una inferencia — ejemplo con Turnigy a -2dB

```
pulso = iq_tensor[:, inicio:fin]    → [2, 19.600]   (1.4 ms × 14 MHz)

1. AGC: pulso / max(abs(pulso))     → [2, 19.600]   amplitud normalizada
2. .unsqueeze(0)                    → [1, 2, 19.600] batch de 1

3. ConvBlock(1→32,  k=31, s=2)     → [1,  64, 9.785]
4. ConvBlock(32→64, k=15, s=2)     → [1, 128, 4.886]
5. ConvBlock(64→128,k=7,  s=2)     → [1, 256, 2.440]
6. ConvBlock(128→128,k=3, s=1)     → [1, 256, 2.438]

7. modulus()                        → [1, 128, 2.438]
8. AdaptiveAvgPool1d(64)            → [1, 128,    64]
9. Flatten                          → [1, 8.192]
10. Linear 8192→512 + ReLU + Drop   → [1, 512]
11. Linear  512→ 64 + ReLU          → [1,  64]
12. Linear   64→  2                 → [1, 2]   logits
13. softmax(logits)                 → [0.42, 0.58] → DRON al 58%
```

---

## 9. Fortalezas y limitaciones honestas

| Fortaleza | Limitación |
|---|---|
| Preserva fase I/Q completa | No modela secuencia de hops (sin LSTM) |
| Maneja longitud variable nativamente | Límite físico a SNR ≤ -6dB |
| Arquitectura progresiva multiescala | Sesgo hacia noise por desequilibrio de clases |
| Justificación matemática sólida (Wirtinger) | Turnigy poco representado vs FutabaT14 |

Para una TFM, esta arquitectura es sólida, justificada con literatura académica,
y los resultados son comparables con el estado del arte en clasificación de
señales RF a baja SNR.
