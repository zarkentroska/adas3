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

El repositorio ya está configurado. Para hacer el push inicial, necesitas autenticarte:

**Opción A: Usar Token de Acceso Personal (Recomendado)**

1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Genera un nuevo token con permisos `repo`
3. Copia el token
4. Ejecuta:
```bash
git push -u origin main
```
Cuando te pida usuario, ingresa tu usuario de GitHub. Cuando te pida contraseña, pega el token (no tu contraseña de GitHub).

**Opción B: Configurar credenciales guardadas**

```bash
git config --global credential.helper store
git push -u origin main
```
(Te pedirá usuario y token una vez, luego se guardará)

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

