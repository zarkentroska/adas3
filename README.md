# Proyecto ADAS3

Sistema de detección de drones con análisis de audio y video.

## Configuración del Repositorio GitHub

### Paso 1: Crear el repositorio en GitHub

1. Ve a [GitHub](https://github.com) e inicia sesión
2. Haz clic en el botón "+" en la esquina superior derecha y selecciona "New repository"
3. Nombra el repositorio (por ejemplo: `adas3`)
4. **NO** inicialices con README, .gitignore o licencia (ya los tenemos localmente)
5. Haz clic en "Create repository"

### Paso 2: Conectar el repositorio local con GitHub

Después de crear el repositorio en GitHub, ejecuta estos comandos (reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub):

```bash
git remote add origin https://github.com/TU_USUARIO/adas3.git
git branch -M main
git push -u origin main
```

### Paso 3: Actualizar el repositorio cuando hagas cambios

Cada vez que modifiques archivos y quieras actualizar GitHub, ejecuta:

```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

O usa el script helper: `./actualizar_github.sh "Descripción de los cambios"`

## Estructura del Proyecto

- `testcam.py` - Script principal
- `pyinstaller.spec` - Configuración de empaquetado
- Modelos y archivos de configuración en la raíz del proyecto

## Notas

- La carpeta `.venv/` está excluida del repositorio (ver `.gitignore`)
- Los archivos compilados de PyInstaller también están excluidos

