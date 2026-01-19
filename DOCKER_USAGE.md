# 🐳 Docker Setup para Pokemoncito

Este documento explica cómo ejecutar Pokemoncito usando Docker para garantizar un entorno consistente en cualquier ordenador.

## Requisitos

- Docker instalado (versión 20.10 o superior)
- Docker Compose instalado (versión 1.29 o superior)

### Instalación de Docker

**Windows:**
- Descargar e instalar [Docker Desktop para Windows](https://docs.docker.com/desktop/install/windows-install/)

**macOS:**
- Descargar e instalar [Docker Desktop para Mac](https://docs.docker.com/desktop/install/mac-install/)

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Añadir tu usuario al grupo docker
sudo usermod -aG docker $USER
# Cerrar sesión y volver a iniciar para aplicar cambios
```

## Uso Rápido

### 1. Construir y Ejecutar con Docker Compose

Desde el directorio raíz del proyecto:

```bash
# Construir la imagen y ejecutar el contenedor
docker-compose up --build

# O en segundo plano (detached mode)
docker-compose up -d --build
```

### 2. Acceder a la Aplicación

Una vez que el contenedor esté ejecutándose, abre tu navegador y visita:

```
http://localhost:8501
```

### 3. Detener la Aplicación

```bash
# Si está en primer plano, presiona Ctrl+C

# Si está en segundo plano:
docker-compose down
```

## Comandos Útiles

### Ver logs de la aplicación
```bash
docker-compose logs -f
```

### Reconstruir la imagen (después de cambios en el código)
```bash
docker-compose up --build
```

### Eliminar todo (imagen, contenedor, volúmenes)
```bash
docker-compose down -v
docker rmi pokemoncito-pokemoncito
```

### Ejecutar comandos dentro del contenedor
```bash
# Abrir una shell interactiva
docker-compose exec pokemoncito bash

# Ejecutar un comando específico
docker-compose exec pokemoncito python PokemonRL/test_train.py
```

## Persistencia de Datos

Los checkpoints entrenados se guardan en volúmenes Docker que persisten entre reinicios:
- `./PokemonRL/checkpoints` - Modelos entrenados
- `./data` - Datos del juego
- `./PokemonRL/data` - Sprites y datos adicionales

Estos directorios se montan desde tu máquina host, por lo que los cambios se mantienen incluso si eliminas el contenedor.

## Solución de Problemas

### Puerto 8501 ya en uso
Si recibes un error de que el puerto 8501 ya está en uso:

```bash
# Cambiar el puerto en docker-compose.yml
ports:
  - "8502:8501"  # Usa el puerto 8502 en tu máquina
```

Luego accede en: `http://localhost:8502`

### Problemas de permisos en Linux
Si tienes problemas de permisos con los volúmenes:

```bash
# Asegúrate de que tu usuario tenga permisos en los directorios
sudo chown -R $USER:$USER PokemonRL/checkpoints data PokemonRL/data
```

### Contenedor se detiene inmediatamente
Ver los logs para diagnosticar:
```bash
docker-compose logs
```

### Reconstruir desde cero
```bash
# Eliminar todo y reconstruir
docker-compose down -v
docker system prune -a
docker-compose up --build
```

## Uso de GPU en Docker (Opcional)

Para usar GPU NVIDIA dentro de Docker, necesitas:

1. Instalar [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Modificar `docker-compose.yml`:
```yaml
services:
  pokemoncito:
    # ... configuración existente ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Reconstruir y ejecutar:
```bash
docker-compose up --build
```

## Desarrollo con Docker

Para desarrollo activo con recarga automática:

```bash
# Montar el código fuente como volumen (añadir en docker-compose.yml)
volumes:
  - ./streamlit_app.py:/app/streamlit_app.py
  - ./PokemonRL/src:/app/PokemonRL/src
```

Streamlit detectará los cambios y recargará automáticamente.

## Arquitectura del Contenedor

El contenedor Docker:
- **Base**: Python 3.10 slim
- **Puerto expuesto**: 8501
- **Directorio de trabajo**: `/app`
- **Comando de inicio**: `streamlit run streamlit_app.py`

## Variables de Entorno

Puedes personalizar el comportamiento con variables de entorno en `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Integración Continua

Ejemplo de construcción automatizada con GitHub Actions:

```yaml
name: Build Docker Image
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t pokemoncito:latest .
```

## Recursos Adicionales

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Nota**: El primer `docker-compose up --build` puede tardar varios minutos en descargar las imágenes base y las dependencias de Python.
