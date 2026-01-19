# Guía de Configuración GPU para Entrenamiento

## ¿Por qué usar GPU?

El entrenamiento en GPU puede ser **10-100x más rápido** que en CPU para redes neuronales, especialmente con:
- Redes convolucionales grandes (MapCNN: 3.47M parámetros)
- Batch processing (Experience Replay con batch_size=32)
- Operaciones matriciales intensivas

## Estado Actual

El código **ya está preparado para GPU**. Los agentes automáticamente detectan y usan GPU si está disponible:

```python
# En explorer.py y tactician.py
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Cómo Activar GPU

### Opción 1: Instalar PyTorch con CUDA (Recomendado)

#### Windows con GPU NVIDIA:
```bash
# Desinstalar PyTorch CPU
pip uninstall torch

# Instalar PyTorch con CUDA 11.8 (compatible con la mayoría de GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O CUDA 12.1 para GPUs más nuevas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Linux con GPU NVIDIA:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Opción 2: Usar Google Colab (Gratis con GPU)

1. Abre [Google Colab](https://colab.research.google.com/)
2. Sube tu código o clona el repo:
   ```python
   !git clone https://github.com/Weryyy/Pokemoncito.git
   %cd Pokemoncito/PokemonRL
   ```
3. Activa GPU: Runtime → Change runtime type → GPU
4. Instala dependencias:
   ```python
   !pip install gymnasium pygame
   ```
5. Ejecuta entrenamiento:
   ```python
   !python train.py
   ```

### Opción 3: Kaggle Notebooks (Gratis con GPU)

Similar a Colab pero con 30h semanales de GPU T4.

## Verificar GPU

Ejecuta este código para verificar:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Usando CPU - Para GPU, reinstala PyTorch con CUDA")
```

## Rendimiento Esperado

### CPU (ejemplo: Intel i7):
- ~0.5-2 segundos por episodio
- **3000 episodios**: ~2-4 horas

### GPU (ejemplo: NVIDIA RTX 3060):
- ~0.05-0.2 segundos por episodio
- **3000 episodios**: ~15-30 minutos

### GPU en la nube (Tesla T4):
- ~0.1-0.3 segundos por episodio
- **3000 episodios**: ~20-40 minutos

## Nuevas Características de Timing

El script `train.py` ahora muestra:

### Durante el entrenamiento:
```
Ep 200/3000 | Mapa 1 | R: -25.3 | Avg100: -28.1 | Eps: 0.952
⏱️  Tiempo: 0:05:23 | ETA: 1:15:45 | Ep/s: 0.62
```

### En checkpoints:
```
💾 CHECKPOINT GUARDADO: Episodio 200
📊 Progreso: 6.7% | Tiempo transcurrido: 0:05:23 | Tiempo restante estimado: 1:15:45
```

### Al finalizar:
```
✅ ENTRENAMIENTO COMPLETADO!
   Tiempo total: 2:15:30
   Mejor recompensa promedio: -15.23
   Tiempo promedio por episodio: 2.71s
```

## Consejos de Optimización

1. **Usar GPU siempre que sea posible** - Es la mejora más significativa
2. **Batch size**: Ya optimizado en 32 (balance GPU memoria/velocidad)
3. **Paralelización**: PyTorch ya paraleliza operaciones en GPU automáticamente
4. **Múltiples GPUs**: No necesario para este proyecto (las redes son pequeñas)

## Requisitos de GPU

### Mínimo:
- NVIDIA GPU con CUDA Compute Capability 3.5+
- 2 GB VRAM
- Drivers NVIDIA actualizados

### Recomendado:
- NVIDIA GTX 1060 o superior
- 4+ GB VRAM
- CUDA 11.8 o superior

## Troubleshooting

### "CUDA out of memory"
Si aparece este error:
1. Reduce `batch_size` en los agentes (línea ~12 en explorer.py y tactician.py)
2. Cierra otros programas que usen GPU

### "CUDA not available" después de instalar
1. Verifica drivers NVIDIA: `nvidia-smi`
2. Reinstala PyTorch con la versión CUDA correcta
3. Reinicia el sistema

### GPU no detectada en Windows
1. Actualiza drivers desde [NVIDIA](https://www.nvidia.com/Download/index.aspx)
2. Verifica que tu GPU soporte CUDA
3. Reinstala PyTorch con la versión CUDA apropiada

## Comparación de Rendimiento

| Método | Velocidad | Costo | Mejor para |
|--------|-----------|-------|------------|
| CPU Local | 0.5-2 ep/s | Gratis | Testing rápido |
| GPU Local | 5-20 ep/s | Hardware | Entrenamiento frecuente |
| Google Colab | 3-10 ep/s | Gratis | Sin GPU local |
| GPU Cloud | 5-15 ep/s | Pago | Proyectos grandes |

## ¿Vale la pena?

**SÍ** si:
- Tienes GPU NVIDIA (gratis, solo instalar PyTorch-CUDA)
- Vas a entrenar múltiples veces o con más episodios
- Quieres experimentar rápidamente con hiperparámetros

**Usar Colab/Kaggle** si:
- No tienes GPU NVIDIA
- Solo necesitas entrenar ocasionalmente
- Quieres probar antes de comprar hardware

**CPU está bien** si:
- Solo entrenas una vez
- Tienes paciencia (2-4 horas)
- CPU moderno (8+ cores)
