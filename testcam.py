import cv2
import time
import numpy as np
import requests
import pyaudio
import threading
import tkinter as tk
from tkinter import Tk, simpledialog, messagebox, filedialog
from tkinter import ttk
import json
import os
import subprocess
import shutil
import matplotlib
# Configurar backend no interactivo para hilos (CRÍTICO para evitar crasheos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ultralytics import YOLO
import librosa
import tensorflow as tf
import queue
import serial
import serial.tools.list_ports
import struct

# Obtener la ruta absoluta del directorio donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas a los recursos
CONFIG_FILE = os.path.join(BASE_DIR, "config_camara.json")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "drone_audio_model.h5")
YOLO_DEFAULT_MODEL = os.path.join(BASE_DIR, "best.pt")
YOLO_MODELS_CONFIG = os.path.join(BASE_DIR, "yolo_models_config.json")
AUDIO_MEAN_PATH = os.path.join(BASE_DIR, "audio_mean.npy")
AUDIO_STD_PATH = os.path.join(BASE_DIR, "audio_std.npy")
SETTINGS_ICON_PATH = os.path.join(BASE_DIR, "settings.png")

# Estado de modelos YOLO
yolo_model_path = YOLO_DEFAULT_MODEL
yolo_model_slots = []
yolo_default_slot = 0
yolo_options_thread = None

# --- VARIABLES GLOBALES UI ---
mouse_x, mouse_y = -1, -1
click_event_pos = None
mouse_is_down = False
pending_ip_change = None
ip_dialog_thread = None
adb_connected = False
adb_message_timer = 0
last_adb_check = 0
ADB_TARGET_IP = "127.0.0.1:8080"
ADB_CHECK_INTERVAL = 5.0
last_wifi_ip = None

def mouse_handler(event, x, y, flags, param):
    """Callback para manejar eventos del ratón"""
    global mouse_x, mouse_y, click_event_pos, mouse_is_down, yolo_slider_active
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_is_down = True
        click_event_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_down = False
        yolo_slider_active = None


def update_stream_endpoints(ip_with_port, record_wifi=True):
    global ip_y_puerto, base_url, video_url, audio_url, last_wifi_ip
    if record_wifi and ip_with_port != ADB_TARGET_IP:
        last_wifi_ip = ip_with_port
    ip_y_puerto = ip_with_port
    base_url = f"http://{ip_y_puerto}"
    video_url = base_url + "/video"
    audio_url = base_url + "/audio.wav"
    guardar_ip(ip_y_puerto)


def load_yolo_models_config():
    """Carga o inicializa la configuración de modelos YOLO."""
    global yolo_model_slots, yolo_default_slot, yolo_model_path

    default_slots = [
        {"path": YOLO_DEFAULT_MODEL, "description": "Modelo por defecto"},
    ] + [{"path": "", "description": ""} for _ in range(14)]

    if os.path.exists(YOLO_MODELS_CONFIG):
        try:
            with open(YOLO_MODELS_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
                slots = data.get("slots", [])
                while len(slots) < 15:
                    slots.append({"path": "", "description": ""})
                yolo_model_slots = slots[:15]
                yolo_default_slot = int(data.get("default_slot", 0))
        except Exception as e:
            print(f"[YOLO] No se pudo leer configuración de modelos: {e}")
            yolo_model_slots = default_slots
            yolo_default_slot = 0
    else:
        yolo_model_slots = default_slots
        yolo_default_slot = 0

    # Validar slot por defecto
    if not (0 <= yolo_default_slot < len(yolo_model_slots)):
        yolo_default_slot = 0

    default_path = yolo_model_slots[yolo_default_slot].get("path") or YOLO_DEFAULT_MODEL
    if not os.path.isabs(default_path):
        default_path = os.path.join(BASE_DIR, default_path)
    yolo_model_path = default_path


def save_yolo_models_config():
    """Guarda la configuración actual de modelos YOLO."""
    try:
        data = {
            "slots": yolo_model_slots,
            "default_slot": yolo_default_slot,
        }
        with open(YOLO_MODELS_CONFIG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[YOLO] No se pudo guardar configuración de modelos: {e}")


load_yolo_models_config()


def get_yolo_settings_icon():
    """Carga y retorna el icono de ajustes para YOLO."""
    global yolo_settings_icon
    if yolo_settings_icon is None:
        if os.path.exists(SETTINGS_ICON_PATH):
            icon = cv2.imread(SETTINGS_ICON_PATH, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                desired_size = 26
                icon = cv2.resize(icon, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
                yolo_settings_icon = icon
        if yolo_settings_icon is None:
            size = 26
            fallback = np.zeros((size, size, 4), dtype=np.uint8)
            cv2.circle(fallback, (size // 2, size // 2), size // 2 - 2, (90, 90, 90, 255), -1, cv2.LINE_AA)
            yolo_settings_icon = fallback
    return yolo_settings_icon

def cargar_ip():
    """Carga la última IP guardada o retorna la por defecto"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('ip', '192.168.1.129:8080')
        except:
            pass
    return '192.168.1.129:8080'

def guardar_ip(ip):
    """Guarda la IP en el archivo de configuración"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'ip': ip}, f)
    except Exception as e:
        print(f"Error al guardar IP: {e}")

def solicitar_nueva_ip(ip_actual):
    """Muestra diálogo para cambiar la IP"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    nueva_ip = simpledialog.askstring(
        "Cambiar Cámara IP",
        f"Introduce la nueva IP y puerto:\n(actual: {ip_actual})",
        initialvalue=ip_actual
    )
    
    root.destroy()
    
    if nueva_ip and nueva_ip.strip():
        return nueva_ip.strip()
    return None

# Cargar IP guardada al iniciar
ip_y_puerto = cargar_ip()
base_url = f"http://{ip_y_puerto}"
video_url = base_url + "/video"
audio_url = base_url + "/audio.wav"
window_name = 'Detector de Drones (Audio, Video y RF)'

print(f"Iniciando con IP guardada: {base_url}")

# Estados auxiliares Windows / conexión video
video_connection_attempts = []
windows_cursor_fixed = False
windows_cursor_warning = False

# --- CONFIGURACIÓN DE AUDIO ---
CHUNK = 1024
p = pyaudio.PyAudio()
audio_stream = None
stop_audio_thread = False
audio_enabled = False
audio_thread = None

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': '*/*',
    'Connection': 'keep-alive'
}

# --- CONFIGURACIÓN DETECCIÓN DE AUDIO ---
audio_model = None
audio_mean = None
audio_std = None
audio_detection_enabled = False
audio_detection_thread = None
audio_buffer = queue.Queue(maxsize=20)
audio_detection_result = {"is_drone": False, "confidence": 0.0}
audio_detection_lock = threading.Lock()

# Variables para espectrograma de audio (DATOS RAW)
audio_spectrogram_data = None
audio_spectrogram_freqs = None
audio_spectrogram_lock = threading.Lock()

# Variables para el RENDERIZADO del espectrograma
spectrogram_image_ready = None
spectrogram_render_thread = None
spectrogram_render_active = False
spectrogram_image_lock = threading.Lock()

AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 2
AUDIO_CONFIDENCE_THRESHOLD = 0.7
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# --- CONFIGURACIÓN TINYSA ULTRA+ ---
# Soporta dos modos: serial directo (PC) o HTTP (Android)
tinysa_serial = None
tinysa_running = False
tinysa_thread = None
tinysa_render_thread = None
tinysa_menu_thread = None
tinysa_http_response = None  # Stream HTTP para recibir datos
tinysa_use_http = False  # Indica si usar modo HTTP o serial

tinysa_data_lock = threading.Lock()
tinysa_render_lock = threading.Lock()

# Datos compartidos
tinysa_data_ready = None         # (freqs, levels) actual
tinysa_image_ready = None        # último frame RGBA renderizado

# Detección de drones por RF
rf_drone_detection_result = {"is_drone": False, "confidence": 0.0, "frequency": None}
rf_drone_detection_lock = threading.Lock()
rf_drone_detection_enabled = True
rf_drone_detection_history = []  # Historial de detecciones para persistencia

# Parámetros ajustables de detección RF (con sliders)
rf_peak_threshold = -80.0  # dBm - umbral mínimo para considerar un pico significativo
rf_min_peak_height_db = 15.0  # dB - altura mínima del pico sobre el ruido
rf_min_peak_width_mhz = 10.0  # MHz - ancho mínimo del pico
rf_max_peak_width_mhz = 50.0  # MHz - ancho máximo del pico
rf_sliders_visible = False  # Control de visibilidad de sliders RF
rf_detection_params_lock = threading.Lock()  # Lock para parámetros RF
tinysa_overlay_cache = None

# Configuración actual
current_tinysa_config = None
tinysa_sequence = []
tinysa_sequence_index = 0
TIN_YSA_SWEEPS_PER_RANGE = 5
tinysa_current_label = ""
ADVANCED_INTERVALS_FILE = os.path.join(BASE_DIR, "tinysa_advanced_intervals.json")
last_advanced_intervals = []
tinysa_detected = False
tinysa_last_check = 0.0
TIN_YSA_CHECK_INTERVAL = 5.0
tinysa_last_sequence_payload = None  # Copia del último payload enviado en modo HTTP
TINYSA_HTTP_CONNECT_TIMEOUT = 5.0
TINYSA_HTTP_READ_TIMEOUT = 120.0
TINYSA_STREAM_CHUNK_SIZE = 8192  # 8KB para JSON con 200 puntos (~5KB)
TINYSA_NO_DATA_TIMEOUT = 12.0
TINYSA_POINTS = 200  # Puntos por barrido

TINYSA_PRESETS = {
    "Normal": {"center": 2442000000, "span": 100000000, "points": TINYSA_POINTS},
    "Alt":    {"start": 5725000000, "stop": 5850000000, "points": TINYSA_POINTS}
}


def _preset_to_range(config, label):
    """Convierte un preset en un rango start/stop en Hz."""
    if "center" in config and "span" in config:
        start = int(config["center"] - config["span"] / 2)
        stop = int(config["center"] + config["span"] / 2)
    else:
        start = int(config["start"])
        stop = int(config["stop"])
    return {
        "start": start,
        "stop": stop,
        "points": int(config.get("points", TINYSA_POINTS)),
        "sweeps": TIN_YSA_SWEEPS_PER_RANGE,
        "label": label.replace("–", "-"),
    }


def build_tinysa_sequence(selection, custom_data=None, advanced_ranges=None):
    """Genera la secuencia de barridos según la selección del usuario."""
    sequence = []

    if selection == "preset1":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Normal"], "FPV-Normal 2.442 GHz"))
    elif selection == "preset2":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Alt"], "FPV-Alt 5.8 GHz"))
    elif selection == "mix":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Normal"], "FPV Mix - 2.442 GHz"))
        sequence.append(_preset_to_range(TINYSA_PRESETS["Alt"], "FPV Mix - 5.8 GHz"))
    elif selection == "custom" and custom_data:
        start_mhz, stop_mhz = custom_data
        start_hz = int(start_mhz * 1e6)
        stop_hz = int(stop_mhz * 1e6)
        sequence.append({
            "start": start_hz,
            "stop": stop_hz,
            "points": TINYSA_POINTS,
            "sweeps": TIN_YSA_SWEEPS_PER_RANGE,
            "label": f"Custom {start_mhz:.3f}-{stop_mhz:.3f} MHz",
        })
    elif selection == "advanced" and advanced_ranges:
        # Guardar la última configuración para reutilizarla
        last_advanced_intervals.clear()
        for idx, (start_mhz, stop_mhz, sweeps_val) in enumerate(advanced_ranges, start=1):
            last_advanced_intervals.append(
                {"start_mhz": start_mhz, "stop_mhz": stop_mhz, "sweeps": sweeps_val}
            )
            sequence.append({
                "start": int(start_mhz * 1e6),
                "stop": int(stop_mhz * 1e6),
                "points": TINYSA_POINTS,
                "sweeps": max(1, int(sweeps_val)),
                "label": f"Avanzado #{idx}: {start_mhz:.3f}-{stop_mhz:.3f} MHz",
            })

    return sequence


def show_advanced_interval_dialog(parent):
    """Diálogo para configurar hasta 5 intervalos personalizados."""
    dialog = tk.Toplevel(parent)
    dialog.title("Rango personalizado - Intervalo avanzado")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Hasta 5 intervalos (MHz)").grid(row=0, column=0, columnspan=5, pady=(0, 10))

    entries = []
    for i in range(5):
        start_var = tk.StringVar()
        stop_var = tk.StringVar()
        sweeps_var = tk.StringVar(value=str(TIN_YSA_SWEEPS_PER_RANGE))
        ttk.Label(frame, text=f"Intervalo {i + 1}:").grid(row=i + 1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, text="Inicio").grid(row=i + 1, column=1, sticky="e")
        start_entry = ttk.Entry(frame, textvariable=start_var, width=10)
        start_entry.grid(row=i + 1, column=2, padx=5, pady=2)
        ttk.Label(frame, text="Fin").grid(row=i + 1, column=3, sticky="e")
        stop_entry = ttk.Entry(frame, textvariable=stop_var, width=10)
        stop_entry.grid(row=i + 1, column=4, padx=5, pady=2)
        ttk.Label(frame, text="# barridos").grid(row=i + 1, column=5, sticky="e")
        sweeps_entry = ttk.Entry(frame, textvariable=sweeps_var, width=6)
        sweeps_entry.grid(row=i + 1, column=6, padx=5, pady=2)
        entries.append((start_var, stop_var, sweeps_var))

    # Prefill con la última configuración guardada
    for (start_var, stop_var, sweeps_var), saved in zip(entries, last_advanced_intervals):
        start_var.set(str(saved["start_mhz"]))
        stop_var.set(str(saved["stop_mhz"]))
        sweeps_var.set(str(saved["sweeps"]))

    result = {"ranges": None}

    def on_ok():
        ranges = []
        for idx, (start_var, stop_var, sweeps_var) in enumerate(entries, start=1):
            start_text = start_var.get().strip()
            stop_text = stop_var.get().strip()
            if not start_text and not stop_text:
                continue
            if not start_text or not stop_text:
                messagebox.showerror("Error", f"Completa inicio y fin para el intervalo {idx}.")
                return
            try:
                start_val = float(start_text)
                stop_val = float(stop_text)
                sweeps_val = int(sweeps_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", f"Valores inválidos en intervalo {idx}.")
                return
            if stop_val <= start_val:
                messagebox.showerror("Error", f"El fin debe ser mayor que el inicio en el intervalo {idx}.")
                return
            if sweeps_val <= 0:
                messagebox.showerror("Error", f"El número de barridos debe ser mayor a cero (intervalo {idx}).")
                return
            ranges.append((start_val, stop_val, sweeps_val))

        if not ranges:
            messagebox.showerror("Error", "Introduce al menos un intervalo válido.")
            return

        result["ranges"] = ranges
        try:
            with open(ADVANCED_INTERVALS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"start_mhz": r[0], "stop_mhz": r[1], "sweeps": r[2]}
                        for r in ranges
                    ],
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[TinySA] No se pudo guardar configuración avanzada: {e}")
        dialog.destroy()

    def on_cancel():
        result["ranges"] = None
        dialog.destroy()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=7, column=0, columnspan=5, pady=(15, 0))
    ttk.Button(btn_frame, text="OK", command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Cancelar", command=on_cancel, width=12).pack(side="left", padx=5)

    dialog.wait_window()
    return result["ranges"]


def show_tinysa_menu():
    """Muestra el selector gráfico para TinySA."""
    root = tk.Tk()
    root.title("TinySA - Selección de modo")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # Precargar intervalos avanzados guardados
    global last_advanced_intervals
    if os.path.exists(ADVANCED_INTERVALS_FILE):
        try:
            with open(ADVANCED_INTERVALS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    last_advanced_intervals = data
        except Exception as e:
            print(f"[TinySA] No se pudo leer configuración avanzada previa: {e}")

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    selection_var = tk.StringVar(value="preset1")
    custom_start = tk.StringVar()
    custom_stop = tk.StringVar()

    ttk.Label(main_frame, text="Selecciona un modo:", font=("Arial", 11, "bold")).pack(anchor="w")

    options_frame = ttk.Frame(main_frame)
    options_frame.pack(fill="x", pady=10)

    ttk.Radiobutton(
        options_frame, text="FPV-Normal (2.442 GHz)", variable=selection_var, value="preset1"
    ).pack(anchor="w", pady=2)
    ttk.Radiobutton(
        options_frame, text="FPV-Alt (5.8 GHz)", variable=selection_var, value="preset2"
    ).pack(anchor="w", pady=2)
    ttk.Radiobutton(
        options_frame,
        text="FPV Mix (2.4 y 5.8 GHz secuencial)",
        variable=selection_var,
        value="mix",
    ).pack(anchor="w", pady=2)

    custom_radio = ttk.Radiobutton(
        options_frame,
        text="Rango personalizado",
        variable=selection_var,
        value="custom",
    )
    custom_radio.pack(anchor="w", pady=2)

    custom_frame = ttk.Frame(options_frame)
    custom_frame.pack(anchor="w", padx=20, pady=(0, 5))
    ttk.Label(custom_frame, text="Inicio (MHz):").grid(row=0, column=0, sticky="w")
    custom_start_entry = ttk.Entry(custom_frame, textvariable=custom_start, width=10, state="disabled")
    custom_start_entry.grid(row=0, column=1, padx=5)
    ttk.Label(custom_frame, text="Fin (MHz):").grid(row=0, column=2, sticky="w")
    custom_stop_entry = ttk.Entry(custom_frame, textvariable=custom_stop, width=10, state="disabled")
    custom_stop_entry.grid(row=0, column=3, padx=5)

    ttk.Radiobutton(
        options_frame,
        text="Rango personalizado - Intervalo avanzado",
        variable=selection_var,
        value="advanced",
    ).pack(anchor="w", pady=2)

    result = {"selection": None, "custom": None, "advanced": None}

    def update_custom_state(*_):
        state = "normal" if selection_var.get() == "custom" else "disabled"
        custom_start_entry.configure(state=state)
        custom_stop_entry.configure(state=state)

    selection_var.trace_add("write", update_custom_state)

    def finish_and_close():
        root.quit()

    def on_ok():
        sel = selection_var.get()
        if sel == "custom":
            try:
                start_val = float(custom_start.get())
                stop_val = float(custom_stop.get())
            except ValueError:
                messagebox.showerror("Error", "Introduce valores numéricos válidos para inicio y fin.")
                return
            if stop_val <= start_val:
                messagebox.showerror("Error", "El fin debe ser mayor que el inicio.")
                return
            result["selection"] = sel
            result["custom"] = (start_val, stop_val)
            finish_and_close()
        elif sel == "advanced":
            try:
                root.attributes("-disabled", True)
            except Exception:
                pass
            ranges = show_advanced_interval_dialog(root)
            try:
                root.attributes("-disabled", False)
            except Exception:
                pass
            if ranges is None:
                root.focus_set()
                return
            result["selection"] = sel
            result["advanced"] = ranges
            finish_and_close()
        else:
            result["selection"] = sel
            finish_and_close()

    def on_cancel():
        result["selection"] = None
        finish_and_close()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(10, 0))
    ttk.Button(btn_frame, text="OK", command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Cancelar", command=on_cancel, width=12).pack(side="left", padx=5)

    def on_close():
        result["selection"] = None
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    if root.winfo_exists():
        root.destroy()
    return result

# --- FUNCIONES TINYSA HARDWARE ---

def find_tinysa_port():
    """Busca el puerto COM del TinySA Ultra"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == 0x0483 and port.pid == 0x5740:
            return port.device
    return None


def read_exactly(ser, n):
    """
    Lee exactamente n bytes del puerto serie, reintentando si es necesario.
    Devuelve menos bytes sólo si se agota el timeout.
    """
    data = b''
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            break  # Timeout
        data += chunk
    return data


def send_tinysa_command(command_json):
    """
    Envía un comando JSON al servidor Android para controlar TinySA.
    """
    try:
        command_url = base_url + "/tinysa/command"
        response = requests.post(
            command_url,
            json=command_json,
            headers={'Content-Type': 'application/json'},
            timeout=10  # Aumentado timeout
        )
        if response.status_code == 200:
            print(f"[TINYSA] Comando enviado: {command_json.get('action', 'unknown')}")
            return True
        else:
            print(f"[TINYSA] Error enviando comando: HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"[TINYSA] Timeout enviando comando (puede que el servidor no esté respondiendo)")
        return False
    except Exception as e:
        print(f"[TINYSA] Error enviando comando: {e}")
        return False

def tinysa_hardware_worker_serial():
    """
    Hilo de hardware para TinySA Ultra en modo scanraw con secuencias de barrido (serial directo).
    Recorre los rangos seleccionados ejecutando varios barridos en cada uno.
    """
    global tinysa_data_ready, tinysa_running, tinysa_serial
    global tinysa_sequence_index, tinysa_current_label

    print("[TINYSA] Hardware Worker Serial iniciado")

    if not tinysa_sequence:
        print("[TINYSA] Sin secuencia activa, saliendo worker.")
        return

    try:
        if tinysa_serial is None or not tinysa_serial.is_open:
            print("[TINYSA] Puerto serie no disponible en hardware worker.")
            return

        tinysa_serial.reset_input_buffer()
        tinysa_serial.write(b"abort\r")
        try:
            tinysa_serial.read_until(b"ch> ")
        except Exception:
            pass
        time.sleep(0.05)

        while tinysa_running and tinysa_sequence:
            config = tinysa_sequence[tinysa_sequence_index]
            start = int(config["start"])
            stop = int(config["stop"])
            points = int(config.get("points", TINYSA_POINTS))
            sweeps_target = max(1, int(config.get("sweeps", TIN_YSA_SWEEPS_PER_RANGE)))
            tinysa_current_label = config.get("label", "")

            cmd = f"scanraw {start} {stop} {points}\r".encode()
            sweeps_done = 0

            while tinysa_running and sweeps_done < sweeps_target:
                tinysa_serial.write(cmd)

                try:
                    raw_block = tinysa_serial.read_until(b"}")
                except Exception as e:
                    print(f"[TINYSA] Error leyendo bloque scanraw: {e}")
                    time.sleep(0.05)
                    continue

                if not raw_block:
                    time.sleep(0.02)
                    continue

                start_idx = raw_block.find(b"{")
                end_idx = raw_block.rfind(b"}")

                if start_idx == -1 or end_idx <= start_idx + 1:
                    time.sleep(0.02)
                    continue
                data_bytes = raw_block[start_idx + 1 : end_idx]

                if len(data_bytes) < 30:
                    time.sleep(0.02)
                    continue

                n_points = len(data_bytes) // 3
                if len(data_bytes) % 3 != 0:
                    data_bytes = data_bytes[: n_points * 3]

                if n_points != points:
                    print(
                        f"[TINYSA] Aviso: dispositivo devolvió {n_points} puntos "
                        f"en lugar de {points}."
                    )

                try:
                    values = [v[0] for v in struct.iter_unpack("<xH", data_bytes)]
                    if len(values) != n_points:
                        time.sleep(0.02)
                        continue

                    levels = (np.asarray(values, dtype=np.float32) / 32.0) - 174.0
                    freqs_dynamic = np.linspace(start, stop, n_points, dtype=np.float32)

                    with tinysa_data_lock:
                        tinysa_data_ready = (freqs_dynamic, levels)

                except Exception as e:
                    print(f"[TINYSA] Error parseando datos scanraw: {e}")
                    time.sleep(0.02)
                    continue

                try:
                    tinysa_serial.read_until(b"ch> ")
                except Exception:
                    pass

                sweeps_done += 1

            tinysa_sequence_index = (tinysa_sequence_index + 1) % len(tinysa_sequence)

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker: {e}")
    finally:
        tinysa_current_label = ""

    print("[TINYSA] Hardware Worker Serial finalizado")

def tinysa_hardware_worker():
    """
    Hilo de hardware para TinySA Ultra usando HTTP stream desde Android.
    Lee datos JSON del endpoint /tinysa/data en tiempo real.
    """
    global tinysa_data_ready, tinysa_running, tinysa_http_response
    global tinysa_current_label
    global tinysa_last_sequence_payload, tinysa_use_http

    print("[TINYSA] Hardware Worker HTTP iniciado")
    print(f"[TINYSA] HTTP timeouts -> connect: {TINYSA_HTTP_CONNECT_TIMEOUT}s, read: {TINYSA_HTTP_READ_TIMEOUT}s")

    def restart_remote_scanning(reason):
        """
        Reenvía la secuencia y el comando start al servidor Android si se corta el stream.
        """
        if not tinysa_running or not tinysa_use_http:
            return
        if not tinysa_last_sequence_payload:
            return
        print(f"[TINYSA] Reiniciando barrido remoto ({reason})")
        # No detener si ya está detenido; los comandos fallidos solo mostrarán el log
        send_tinysa_command({"action": "stop"})
        send_tinysa_command({"action": "set_sequence", "sequence": tinysa_last_sequence_payload})
        send_tinysa_command({"action": "start"})
        return time.time()

    buffers_since_data = 0
    BUFFER_LIMIT_BEFORE_TIMEOUT = 24  # 24 * 0.5s = 12s aprox
    
    try:
        data_url = base_url + "/tinysa/data"
        print(f"[TINYSA] Conectando a {data_url}...")
        
        # Conectar al stream de datos
        tinysa_http_response = requests.get(
            data_url,
            stream=True,
            headers=headers,
            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
        )
        
        if tinysa_http_response.status_code != 200:
            print(f"[TINYSA] Error conectando: HTTP {tinysa_http_response.status_code}")
            return
        
        print("[TINYSA] Conectado al stream de datos")
        
        tinysa_http_response.raw.decode_content = True
        
        # Buffer para acumular datos JSON (pueden llegar fragmentados)
        buffer = ""
        last_data_time = time.time()
        
        while tinysa_running:
            try:
                chunk = tinysa_http_response.raw.read(TINYSA_STREAM_CHUNK_SIZE)
                
                if not chunk:
                    if tinysa_running:
                        print("[TINYSA] Stream cerrado, reintentando...")
                        time.sleep(1)
                        # Reintentar conexión
                        try:
                            tinysa_http_response.close()
                        except:
                            pass
                        tinysa_http_response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
                        )
                        if tinysa_http_response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras stream cerrado")
                        if ts:
                            last_data_time = ts
                        else:
                            last_data_time = time.time()
                        continue  # Volver a intentar leer
                    else:
                        break
                
                # Decodificar y agregar al buffer
                buffer += chunk.decode('utf-8', errors='ignore')
                
                # Procesar líneas JSON completas
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Solo procesar si parece un JSON completo
                        if not line.startswith('{') or not line.endswith('}'):
                            # Línea incompleta, ignorar
                            continue
                        
                        # Parsear JSON
                        data = json.loads(line)
                        freqs_array = data.get('freqs', [])
                        levels_array = data.get('levels', [])
                        
                        if len(freqs_array) > 0 and len(levels_array) > 0:
                            freqs = np.array(freqs_array, dtype=np.float32)
                            levels = np.array(levels_array, dtype=np.float32)
                            
                            print(f"[HTTP {time.time():.2f}] Datos: {len(freqs)} pts")
                            
                            with tinysa_data_lock:
                                tinysa_data_ready = (freqs, levels)
                            last_data_time = time.time()
                    except json.JSONDecodeError as e:
                        # JSON incompleto o corrupto, ignorar silenciosamente
                        continue
                    except Exception as e:
                        print(f"[TINYSA] Error procesando datos: {e}")
                        continue
                        
            except requests.exceptions.RequestException as e:
                if tinysa_running:
                    print(f"[TINYSA] Error en stream: {e}, reintentando...")
                    time.sleep(1)
                    try:
                        tinysa_http_response.close()
                    except:
                        pass
                    try:
                        tinysa_http_response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
                        )
                        if tinysa_http_response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras error de red")
                        if ts:
                            last_data_time = ts
                        else:
                            last_data_time = time.time()
                    except:
                        break
                else:
                    break
            except Exception as e:
                print(f"[TINYSA] Error inesperado: {e}")
                time.sleep(0.1)

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker: {e}")
    finally:
        tinysa_current_label = ""
        try:
            if tinysa_http_response:
                tinysa_http_response.close()
        except:
            pass
        tinysa_http_response = None

    print("[TINYSA] Hardware Worker HTTP finalizado")

def detect_drone_rf(freqs, levels):
    """
    Detecta señales de drones en el espectro RF basándose en patrones característicos.
    
    Características de señales de drones FPV:
    - Picos significativos en bandas 2.4 GHz o 5.8 GHz
    - Ancho de banda típico: 20-40 MHz
    - Potencia por encima del ruido de fondo (> -80 dBm típicamente)
    - Forma de pico característica (montañitas)
    
    Args:
        freqs: Array de frecuencias en Hz
        levels: Array de niveles en dBm
        
    Returns:
        dict con is_drone (bool), confidence (float 0-1), frequency (Hz o None)
    """
    global rf_drone_detection_history
    
    if len(freqs) == 0 or len(levels) == 0 or len(freqs) != len(levels):
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    # Convertir a numpy arrays si no lo son
    freqs = np.array(freqs)
    levels = np.array(levels)
    
    # Bandas típicas de drones (en Hz)
    DRONE_BANDS = [
        (2400000000, 2500000000),  # 2.4 GHz (WiFi/FPV)
        (5725000000, 5875000000),  # 5.8 GHz (FPV)
    ]
    
    # Parámetros de detección (usar variables globales ajustables)
    global rf_peak_threshold, rf_min_peak_height_db, rf_min_peak_width_mhz, rf_max_peak_width_mhz
    
    with rf_detection_params_lock:
        PEAK_THRESHOLD = rf_peak_threshold
        MIN_PEAK_HEIGHT_DB = rf_min_peak_height_db
        MIN_PEAK_WIDTH_MHZ = rf_min_peak_width_mhz
        MAX_PEAK_WIDTH_MHZ = rf_max_peak_width_mhz
    
    NOISE_FLOOR = -100  # dBm - nivel de ruido de fondo típico
    
    # Filtrar datos válidos
    valid_mask = np.isfinite(levels) & (levels > -150) & (levels < 0)
    if np.sum(valid_mask) < 10:  # Necesitamos al menos 10 puntos válidos
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    freqs_valid = freqs[valid_mask]
    levels_valid = levels[valid_mask]
    
    # Calcular ruido de fondo (percentil 10 para evitar picos)
    noise_level = np.percentile(levels_valid, 10)
    
    # Buscar picos significativos usando detección de máximos locales (sin scipy)
    # Buscar picos que estén al menos MIN_PEAK_HEIGHT_DB por encima del ruido
    peak_threshold_relative = noise_level + MIN_PEAK_HEIGHT_DB
    
    # Detección simple de picos: un punto es un pico si es mayor que sus vecinos
    # y está por encima del umbral
    # Reducir min_distance para detectar picos más cercanos (mejor para 200 puntos)
    min_distance = max(1, len(levels_valid) // 100)  # Más permisivo: permite picos más cercanos
    peaks = []
    
    # Buscar picos con una ventana más pequeña para mejor detección
    for i in range(min_distance, len(levels_valid) - min_distance):
        if levels_valid[i] < peak_threshold_relative:
            continue
        
        # Verificar que sea un máximo local (comparar con vecinos más cercanos)
        is_peak = True
        window_size = max(2, min_distance // 2)  # Ventana más pequeña para mejor detección
        for j in range(max(0, i - window_size), min(len(levels_valid), i + window_size + 1)):
            if j != i and levels_valid[j] >= levels_valid[i]:
                is_peak = False
                break
        
        if is_peak:
            peaks.append(i)
    
    if len(peaks) == 0:
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    peaks = np.array(peaks)
    
    # Analizar cada pico
    best_peak = None
    best_confidence = 0.0
    best_frequency = None
    
    for peak_idx in peaks:
        peak_freq = freqs_valid[peak_idx]
        peak_level = levels_valid[peak_idx]
        
        # Verificar que esté en una banda de drones
        in_drone_band = False
        for band_start, band_stop in DRONE_BANDS:
            if band_start <= peak_freq <= band_stop:
                in_drone_band = True
                break
        
        if not in_drone_band:
            continue
        
        # Calcular ancho de banda del pico (FWHM - Full Width at Half Maximum)
        half_max = peak_level - (peak_level - noise_level) / 2
        
        # Encontrar puntos donde la señal cruza half_max
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and levels_valid[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(levels_valid) - 1 and levels_valid[right_idx] > half_max:
            right_idx += 1
        
        # Calcular ancho de banda en MHz
        if left_idx < right_idx:
            bandwidth_hz = freqs_valid[right_idx] - freqs_valid[left_idx]
            bandwidth_mhz = bandwidth_hz / 1e6
        else:
            bandwidth_mhz = 0
        
        # Verificar criterios de detección
        if (peak_level > PEAK_THRESHOLD and 
            MIN_PEAK_WIDTH_MHZ <= bandwidth_mhz <= MAX_PEAK_WIDTH_MHZ):
            
            # Calcular confianza basada en:
            # 1. Altura del pico sobre el ruido
            # 2. Ancho de banda (óptimo alrededor de 20-30 MHz)
            # 3. Potencia absoluta
            
            height_above_noise = peak_level - noise_level
            # Normalizar a 30 dB (más generoso que 40 dB)
            height_confidence = min(1.0, height_above_noise / 30.0)
            # Bonus si está por encima de 20 dB
            if height_above_noise > 20:
                height_confidence = min(1.0, height_confidence * 1.2)
            
            # Ancho de banda óptimo alrededor de 20-25 MHz, pero más permisivo
            optimal_bw = 22.5
            bw_diff = abs(bandwidth_mhz - optimal_bw)
            # Más permisivo: penalizar menos por diferencias de ancho de banda
            bw_confidence = max(0.0, 1.0 - (bw_diff / 30.0))  # 30 en lugar de 20
            
            # Potencia absoluta (picos más fuertes = más confianza)
            # Más generoso: normalizar a 20 dB en lugar de 30
            power_confidence = min(1.0, max(0.0, (peak_level - PEAK_THRESHOLD) / 20.0))
            # Bonus si está muy por encima del umbral
            if peak_level > PEAK_THRESHOLD + 10:
                power_confidence = min(1.0, power_confidence * 1.15)
            
            # Confianza combinada (dar más peso a la altura sobre el ruido)
            confidence = (height_confidence * 0.5 + bw_confidence * 0.25 + power_confidence * 0.25)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_peak = peak_idx
                best_frequency = peak_freq
    
    # Persistencia temporal: requerir detecciones consistentes
    current_time = time.time()
    rf_drone_detection_history = [
        (t, freq, conf) for t, freq, conf in rf_drone_detection_history
        if current_time - t < 3.0  # Mantener últimos 3 segundos (más permisivo)
    ]
    
    if best_confidence > 0.4:  # Umbral de confianza más bajo para capturar más detecciones
        rf_drone_detection_history.append((current_time, best_frequency, best_confidence))
        
        # Requerir al menos 1 detección en los últimos 3 segundos (más permisivo)
        if len(rf_drone_detection_history) >= 1:
            # Usar la confianza más alta del historial reciente
            max_confidence = max([conf for _, _, conf in rf_drone_detection_history])
            # Calcular frecuencia promedio ponderada por confianza
            total_weight = sum([conf for _, _, conf in rf_drone_detection_history])
            if total_weight > 0:
                avg_frequency = sum([freq * conf for _, freq, conf in rf_drone_detection_history]) / total_weight
            else:
                avg_frequency = best_frequency
            
            return {
                "is_drone": True,
                "confidence": min(1.0, max_confidence),  # Usar la confianza más alta
                "frequency": avg_frequency
            }
    
    return {"is_drone": False, "confidence": 0.0, "frequency": None}

def tinysa_render_worker():
    """
    Hilo que dibuja el gráfico TinySA con Matplotlib (Agg) y produce un frame RGBA
    listo para superponer en OpenCV.

    - Figura y ejes con fondo negro opaco.
    - Se crea UNA sola figura y UNA sola línea.
    - Sólo se actualiza la Y de la línea y se redibuja el canvas.
    - Se mantiene siempre el último frame válido.
    """
    global tinysa_image_ready
    print("[TINYSA] Render Worker iniciado")

    if current_tinysa_config is None:
        print("[TINYSA] Sin configuración activa, saliendo render worker.")
        return

    # --- Crear figura estática ---
    fig = Figure(figsize=(5, 2.5), facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Fondo y estética
    ax.set_facecolor("black")
    ax.grid(True, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("dBm", color="white", fontsize=8)
    ax.set_xlabel("MHz", color="white", fontsize=8)
    ax.tick_params(axis="x", colors="white", labelsize=7)
    ax.tick_params(axis="y", colors="white", labelsize=7)

    # --- Ejes iniciales a partir de la config ---
    if "center" in current_tinysa_config:
        start = int(current_tinysa_config["center"] - current_tinysa_config["span"] / 2)
        stop = int(current_tinysa_config["center"] + current_tinysa_config["span"] / 2)
    else:
        start = int(current_tinysa_config["start"])
        stop = int(current_tinysa_config["stop"])

    points = int(current_tinysa_config["points"])
    freqs_init = np.linspace(start, stop, points, dtype=np.float32)

    ax.set_xlim(freqs_init[0] / 1e6, freqs_init[-1] / 1e6)
    ax.set_ylim(-125, -10)

    modo = "2.4 GHz" if freqs_init[0] < 3e9 else "5.8 GHz"
    ax.set_title(
        f"TinySA Ultra - {modo}", color="#00FF00", fontsize=9, fontweight="bold"
    )

    # Línea inicial (todo a -110 dBm)
    line, = ax.plot(
        freqs_init / 1e6,
        np.full(points, -110.0, dtype=np.float32),
        color="#FFFF00",
        linewidth=1.5,
    )

    fig.tight_layout()

    last_levels_hash = None
    
    while tinysa_running:
        # Obtener datos
        with tinysa_data_lock:
            data = tinysa_data_ready

        if data is None:
            time.sleep(0.01)
            continue

        freqs, levels = data
        actual_points = len(freqs)

        if actual_points == 0 or len(levels) != actual_points:
            time.sleep(0.005)
            continue

        # Detectar cambios por contenido, no por referencia
        current_hash = hash(levels.tobytes())
        if current_hash == last_levels_hash:
            time.sleep(0.002)  # Polling más rápido
            continue
        last_levels_hash = current_hash
        
        # Detectar drones por RF si está habilitado
        if rf_drone_detection_enabled:
            try:
                detection = detect_drone_rf(freqs, levels)
                with rf_drone_detection_lock:
                    rf_drone_detection_result.update(detection)
                if detection["is_drone"]:
                    freq_mhz = detection["frequency"] / 1e6 if detection["frequency"] else 0
                    print(f"[RF DRONE] DETECTADO: {freq_mhz:.3f} MHz, confianza: {detection['confidence']:.2f}")
            except Exception as e:
                print(f"[RF DRONE] Error en detección: {e}")

        render_start = time.time()
        
        try:
            # Actualizar X e Y de la línea
            line.set_xdata(freqs / 1e9)
            line.set_ydata(levels)
            
            # Ajustar límites del eje X dinámicamente
            if len(freqs) > 0:
                ax.set_xlim(freqs[0] / 1e9, freqs[-1] / 1e9)

            # Renderizar a buffer RGBA
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(canvas.get_width_height()[::-1] + (4,))

            # Publicar imagen para overlay
            with tinysa_render_lock:
                tinysa_image_ready = img
            
            render_time = (time.time() - render_start) * 1000
            print(f"[RENDER {time.time():.2f}] {actual_points} pts en {render_time:.0f}ms")

        except Exception as e:
            print(f"[TINYSA] Error en render: {e}")
            time.sleep(0.05)

    print("[TINYSA] Render Worker finalizado")

def toggle_tinysa():
    """
    Activa/Desactiva y configura el TinySA.
    Soporta dos modos:
    - Serial directo: TinySA conectado al PC vía USB
    - HTTP: TinySA conectado al Android, usando servidor como pasarela
    """
    global tinysa_running, tinysa_serial, current_tinysa_config
    global tinysa_thread, tinysa_render_thread, tinysa_data_ready, tinysa_image_ready
    global tinysa_sequence, tinysa_sequence_index, tinysa_current_label
    global tinysa_menu_thread, tinysa_detected, tinysa_http_response
    global tinysa_use_http  # Nueva variable para indicar modo HTTP
    global tinysa_last_sequence_payload

    if tinysa_running:
        # Apagar
        tinysa_running = False
        
        if tinysa_use_http:
            # Enviar comando stop al servidor Android
            send_tinysa_command({"action": "stop"})
            # Cerrar conexión HTTP
            try:
                if tinysa_http_response:
                    tinysa_http_response.close()
            except:
                pass
            tinysa_http_response = None
        else:
            # Modo serial directo
            if tinysa_serial:
                try:
                    tinysa_serial.write(b"abort\r")
                    tinysa_serial.close()
                except Exception:
                    pass
                tinysa_serial = None

        # Limpiar buffers compartidos
        with tinysa_data_lock:
            tinysa_data_ready = None
        with tinysa_render_lock:
            tinysa_image_ready = None

        tinysa_sequence = []
        tinysa_sequence_index = 0
        tinysa_current_label = ""
        tinysa_use_http = False
        print("TinySA Desactivado")
        return

    if tinysa_menu_thread and tinysa_menu_thread.is_alive():
        print("El menú de TinySA ya está abierto.")
        return

    def enable_flow():
        global tinysa_menu_thread, tinysa_sequence, tinysa_sequence_index
        global current_tinysa_config, tinysa_current_label, tinysa_running
        global tinysa_thread, tinysa_render_thread, tinysa_detected
        global tinysa_serial, tinysa_use_http

        selection_data = show_tinysa_menu()
        selection = selection_data.get("selection")
        if not selection:
            tinysa_menu_thread = None
            return

        sequence = build_tinysa_sequence(
            selection,
            custom_data=selection_data.get("custom"),
            advanced_ranges=selection_data.get("advanced"),
        )

        if not sequence:
            print("Selección TinySA inválida.")
            tinysa_menu_thread = None
            return

        tinysa_sequence = sequence
        tinysa_sequence_index = 0
        current_tinysa_config = tinysa_sequence[0]
        tinysa_current_label = current_tinysa_config.get("label", "")

        # Decidir modo: primero intentar serial directo, luego HTTP
        port = find_tinysa_port()
        tinysa_detected = port is not None
        
        try:
            if port:
                # Modo serial directo (TinySA conectado al PC)
                try:
                    print(f"Conectando a TinySA en {port} (modo serial directo)...")
                    tinysa_serial = serial.Serial(port, 921600, timeout=8.0)

                    tinysa_serial.flushInput()
                    tinysa_serial.write(b"abort\r")
                    tinysa_serial.read_until(b"ch> ")

                    tinysa_running = True
                    tinysa_use_http = False

                    with tinysa_data_lock:
                        tinysa_data_ready = None

                    tinysa_thread = threading.Thread(
                        target=tinysa_hardware_worker_serial, daemon=True  # Usar worker serial
                    )
                    tinysa_thread.start()

                    tinysa_render_thread = threading.Thread(
                        target=tinysa_render_worker, daemon=True
                    )
                    tinysa_render_thread.start()

                    print("TinySA Activado (modo serial directo)")
                    tinysa_detected = True

                except Exception as e:
                    print(f"Error al conectar TinySA por serial: {e}")
                    if tinysa_serial:
                        try:
                            tinysa_serial.close()
                        except:
                            pass
                        tinysa_serial = None
                    tinysa_running = False
                    tinysa_menu_thread = None
                    return
            else:
                # Modo HTTP (TinySA conectado al Android)
                print(f"[TINYSA] TinySA no detectado localmente, intentando modo HTTP...")
                tinysa_use_http = True
                
                try:
                    # Convertir secuencia al formato JSON esperado por el servidor
                    sequence_json = []
                    for config in sequence:
                        sequence_json.append({
                            "start": int(config["start"]),
                            "stop": int(config["stop"]),
                            "points": int(config.get("points", TINYSA_POINTS)),
                            "sweeps": int(config.get("sweeps", TIN_YSA_SWEEPS_PER_RANGE)),
                            "label": config.get("label", "")
                        })
                    
                    # Guardar copia profunda para poder rearmar la secuencia si el stream se corta
                    try:
                        tinysa_last_sequence_payload = json.loads(json.dumps(sequence_json))
                    except Exception:
                        tinysa_last_sequence_payload = sequence_json[:]
                    
                    # Enviar comando set_sequence
                    command = {
                        "action": "set_sequence",
                        "sequence": sequence_json
                    }
                    
                    if not send_tinysa_command(command):
                        print("[TINYSA] Error configurando secuencia en servidor")
                        def show_warning():
                            root = Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            messagebox.showwarning(
                                "TinySA no disponible",
                                "TinySA no detectado localmente ni en el servidor Android.\n"
                                "Conéctalo vía USB al PC o al Android e intenta de nuevo."
                            )
                            root.destroy()
                        threading.Thread(target=show_warning, daemon=True).start()
                        tinysa_menu_thread = None
                        return
                    
                    # Iniciar scanning
                    if not send_tinysa_command({"action": "start"}):
                        print("[TINYSA] Error iniciando scanning en servidor")
                        tinysa_menu_thread = None
                        return

                    tinysa_running = True

                    with tinysa_data_lock:
                        tinysa_data_ready = None

                    # Iniciar thread para recibir datos HTTP
                    tinysa_thread = threading.Thread(
                        target=tinysa_hardware_worker, daemon=True  # Usar worker HTTP
                    )
                    tinysa_thread.start()

                    # Iniciar thread de renderizado
                    tinysa_render_thread = threading.Thread(
                        target=tinysa_render_worker, daemon=True
                    )
                    tinysa_render_thread.start()

                    print("TinySA Activado (modo HTTP)")
                    # El estado se actualizará mediante poll_tinysa_presence
                    tinysa_detected = True
                    tinysa_use_http = True

                except Exception as e:
                    print(f"Error al conectar TinySA por HTTP: {e}")
                    tinysa_running = False
                    tinysa_use_http = False
        finally:
            tinysa_menu_thread = None

    tinysa_menu_thread = threading.Thread(target=enable_flow, daemon=True)
    tinysa_menu_thread.start()


def overlay_tinysa_graph(frame):
    """
    Dibuja el gráfico del TinySA directamente con OpenCV, transparente sobre el vídeo,
    incluyendo cuadrícula y etiquetas de ejes.
    """

    if not tinysa_running:
        return frame

    # 1. Obtener datos actuales del TinySA
    with tinysa_data_lock:
        data = tinysa_data_ready

    if data is None:
        return frame

    freqs, levels = data
    if freqs is None or levels is None or len(freqs) == 0 or len(levels) == 0:
        return frame

    global tinysa_current_label

    try:
        h, w = frame.shape[:2]

        # Tamaño del panel RF reducido
        panel_w = int(w * 0.27)
        panel_h = int(h * 0.18)

        if panel_w <= 10 or panel_h <= 10:
            return frame

        # Esquina inferior derecha
        x0 = w - panel_w - 10
        y0 = h - panel_h - 10
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        if x0 < 0 or y0 < 0:
            return frame

        # ROI del vídeo donde se superpone el gráfico
        roi = frame[y0:y1, x0:x1]

        # Imagen negra donde dibujamos sólo el gráfico y la cuadrícula
        graph = np.zeros_like(roi)

        # 2. Parámetros de escala
        points = len(levels)

        # Rango de dBm del eje Y (ajústalo a tu gusto)
        db_min = -125.0
        db_max = -10.0

        # Clampear niveles
        lv = np.clip(levels, db_min, db_max)

        # Normalizar e invertir eje Y (dBm altos arriba)
        norm = (lv - db_min) / (db_max - db_min)  # 0..1
        ys = (1.0 - norm) * (panel_h - 1)

        xs = np.linspace(0, panel_w - 1, points)

        pts = np.vstack([xs, ys]).T.astype(np.int32)

        # 3. Dibujar fondo gris oscuro
        graph[:] = (40, 40, 40)

        # 4. Dibujar cuadrícula (ejemplo: 5 divisiones Y, 6 X)
        grid_color = (40, 40, 40)
        n_y = 5
        n_x = 6

        for i in range(1, n_y):
            gy = int(round(i * panel_h / n_y))
            cv2.line(graph, (0, gy), (panel_w - 1, gy), grid_color, 1)

        for i in range(1, n_x):
            gx = int(round(i * panel_w / n_x))
            cv2.line(graph, (gx, 0), (gx, panel_h - 1), grid_color, 1)

        # 5. Ejes y etiquetas de dBm (eje Y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_color = (200, 200, 200)
        thickness = 1

        for i in range(n_y + 1):
            frac = i / n_y
            gy = int(round(frac * (panel_h - 1)))
            db_val = db_max - frac * (db_max - db_min)
            text = f"{int(db_val)}"

            # Línea de referencia gruesa en el borde izquierdo
            cv2.line(graph, (0, gy), (5, gy), (80, 80, 80), 1)

            # Texto a la izquierda (dentro del panel)
            cv2.putText(
                graph,
                text,
                (8, max(10, gy - 2)),
                font,
                font_scale,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

        # 6. Etiquetas de frecuencia aproximadas (eje X)
        f_start_mhz = freqs[0] / 1e6
        f_stop_mhz = freqs[-1] / 1e6

        font_scale_x = 0.4
        for i in range(n_x + 1):
            frac = i / n_x
            gx = int(round(frac * (panel_w - 1)))
            f_val_mhz = f_start_mhz + frac * (f_stop_mhz - f_start_mhz)
            if f_val_mhz >= 1000.0:
                text = f"{f_val_mhz / 1000:.2f}"
                text_offset = 13
            else:
                text = f"{int(round(f_val_mhz))}"
                text_offset = 9

            cv2.line(graph, (gx, panel_h - 8), (gx, panel_h - 1), (80, 80, 80), 1)
            cv2.putText(
                graph,
                text,
                (gx - text_offset, panel_h - 4),
                font,
                font_scale_x,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

        # 7. Título
        ghz_start = freqs[0] / 1e9
        ghz_stop = freqs[-1] / 1e9
        title = f"TinySA Ultra - {ghz_start:.2f}-{ghz_stop:.2f} GHz"
        dynamic_label = tinysa_current_label or f"{freqs[0]/1e6:.2f}-{freqs[-1]/1e6:.2f} MHz"
        cv2.putText(
            graph,
            title,
            (8, 14),
            font,
            0.32,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            graph,
            f"Rango actual: {dynamic_label}",
            (8, 28),
            font,
            0.3,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # 8. Dibujar la traza en amarillo
        cv2.polylines(graph, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

        # 9. Mezclar ROI original con el gráfico usando alpha fijo (transparente)
        alpha = 0.45  # 75% gráfico, 25% vídeo
        cv2.addWeighted(graph, alpha, roi, 1.0 - alpha, 0.0, roi)

    except Exception:
        # Si algo falla, no tocamos el frame
        pass

    return frame

# --- FUNCIONES MODELO AUDIO (EXISTENTES) ---
def cargar_modelo_audio():
    """Carga el modelo de detección de audio y estadísticas de normalización"""
    global audio_model, audio_mean, audio_std
    try:
        if not os.path.exists(AUDIO_MODEL_PATH):
            print(f"Error: No se encuentra el modelo '{AUDIO_MODEL_PATH}'")
            return False
        
        if os.path.exists(AUDIO_MEAN_PATH) and os.path.exists(AUDIO_STD_PATH):
            audio_mean = np.load(AUDIO_MEAN_PATH)
            audio_std = np.load(AUDIO_STD_PATH)
            print(f"Estadísticas cargadas - Mean: {audio_mean:.4f}, Std: {audio_std:.4f}")
        else:
            print("ERROR: No se encontraron archivos de normalización")
            return False
        
        print("Cargando modelo de detección de audio...")
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print("Modelo de audio cargado correctamente")
        return True
    except Exception as e:
        print(f"Error al cargar modelo de audio: {e}")
        return False

def extract_features_realtime(audio_chunk):
    """Extrae features de un chunk de audio en tiempo real"""
    global audio_mean, audio_std, audio_spectrogram_data, audio_spectrogram_freqs
    
    try:
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        audio_data = audio_data / 32768.0
        
        if len(audio_data) > 0:
            audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=AUDIO_SAMPLE_RATE)
        
        required_length = AUDIO_SAMPLE_RATE * AUDIO_DURATION
        if len(audio_data) < required_length:
            audio_data = np.pad(audio_data, (0, required_length - len(audio_data)))
        else:
            audio_data = audio_data[:required_length]
        
        # Calcular espectrograma mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=AUDIO_SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Guardar datos RAW para que el otro thread los pinte
        with audio_spectrogram_lock:
            freqs_mel = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=AUDIO_SAMPLE_RATE/2)
            audio_spectrogram_freqs = freqs_mel
            audio_spectrogram_data = mel_spec_db
        
        # Normalizar para el modelo
        if audio_mean is not None and audio_std is not None:
            mel_spec_db_norm = (mel_spec_db - audio_mean) / (audio_std + 1e-8)
        else:
            return None
        
        if mel_spec_db_norm.shape[1] < 87:
            pad_width = 87 - mel_spec_db_norm.shape[1]
            mel_spec_db_norm = np.pad(mel_spec_db_norm, ((0, 0), (0, pad_width)))
        else:
            mel_spec_db_norm = mel_spec_db_norm[:, :87]
        
        return mel_spec_db_norm
        
    except Exception as e:
        print(f"[FEATURES] Error: {e}")
        return None

def audio_detection_worker():
    """Worker thread para detección de audio (Inferencia AI)"""
    global audio_detection_result
    
    accumulated_audio = b''
    required_bytes = int(44100 * AUDIO_DURATION * 2)
    
    print(f"[AUDIO] Detection Worker iniciado")
    
    while audio_detection_enabled:
        try:
            chunk = audio_buffer.get(timeout=1)
            accumulated_audio += chunk
            
            if len(accumulated_audio) >= required_bytes:
                try:
                    features = extract_features_realtime(accumulated_audio[:required_bytes])
                    
                    if features is not None and audio_model is not None:
                        features = features[..., np.newaxis]
                        features = np.expand_dims(features, axis=0)
                        
                        prediction = audio_model.predict(features, verbose=0)[0][0]
                        is_drone = prediction >= AUDIO_CONFIDENCE_THRESHOLD
                        
                        with audio_detection_lock:
                            audio_detection_result = {
                                "is_drone": is_drone,
                                "confidence": float(prediction)
                            }
                        
                        print(f"[AUDIO] Predicción: {prediction:.3f} | Drone: {is_drone}")
                        
                except Exception as e:
                    print(f"[AUDIO] Error: {e}")
                
                overlap_bytes = int(44100 * 0.5 * 2)
                accumulated_audio = accumulated_audio[-overlap_bytes:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[AUDIO] Error crítico: {e}")
    
    print("[AUDIO] Detection Worker finalizado")

# --- NUEVO: WORKER PARA RENDERIZAR ESPECTROGRAMA ---
def spectrogram_render_worker():
    """
    Thread dedicado a generar la imagen del espectrograma con Matplotlib.
    Esto evita que el bucle principal de video se congele.
    """
    global spectrogram_image_ready
    
    print("[RENDER] Worker iniciado")
    
    while spectrogram_render_active:
        try:
            # 1. Obtener datos RAW
            with audio_spectrogram_lock:
                if audio_spectrogram_data is None or audio_spectrogram_freqs is None:
                    time.sleep(0.1)
                    continue
                data = audio_spectrogram_data.copy()
                freqs = audio_spectrogram_freqs.copy()
            
            # 2. Obtener estado drone
            with audio_detection_lock:
                is_drone = audio_detection_result["is_drone"]
            
            # 3. Configurar colores
            if is_drone:
                blink = int(time.time() * 3) % 2 == 0
                line_color = '#FF0000' if blink else '#AA0000'
                cmap_name = 'hot'
            else:
                line_color = '#00FFFF'
                cmap_name = 'viridis'
            
            # 4. Crear figura Matplotlib (Heavy lifting)
            # Usamos Figure directamente para evitar problemas de thread-safety con plt.
            fig = Figure(figsize=(5, 1.8), facecolor='none')
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            ax.imshow(data, aspect='auto', origin='lower', 
                      cmap=cmap_name,
                      extent=[0, data.shape[1], freqs[0]/1000, freqs[-1]/1000],
                      alpha=0.8)
            
            ax.set_facecolor('none')
            ax.set_xlabel('Tiempo', color='white', fontsize=7)
            ax.set_ylabel('Freq (kHz)', color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            ax.set_title('Audio Spectrogram', color=line_color, fontsize=8, fontweight='bold')
            
            fig.tight_layout()
            
            # 5. Renderizar a buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(canvas.get_width_height()[::-1] + (4,))
            
            # 6. Guardar resultado para el main thread
            with spectrogram_image_lock:
                spectrogram_image_ready = img
            
            # Limitar FPS del renderizado (ahorra mucha CPU)
            # 15 FPS para el gráfico es más que suficiente para el ojo humano
            time.sleep(0.06) 
            
        except Exception as e:
            print(f"[RENDER] Error: {e}")
            time.sleep(0.5)
            
    print("[RENDER] Worker finalizado")

def toggle_audio_detection():
    """Activa/desactiva la detección de audio y el renderizado"""
    global audio_detection_enabled, audio_detection_thread
    global spectrogram_render_active, spectrogram_render_thread
    
    if not audio_detection_enabled:
        if audio_model is None:
            if not cargar_modelo_audio():
                return
        
        if not audio_enabled:
            print("Activa primero el audio (tecla 'M')")
            return
        
        # Iniciar thread de detección (IA)
        audio_detection_enabled = True
        audio_detection_thread = threading.Thread(target=audio_detection_worker, daemon=True)
        audio_detection_thread.start()
        
        # Iniciar thread de renderizado (Gráfico) - NUEVO
        spectrogram_render_active = True
        spectrogram_render_thread = threading.Thread(target=spectrogram_render_worker, daemon=True)
        spectrogram_render_thread.start()
        
        print("Detección y Renderizado activados")
    else:
        # Apagar detección
        audio_detection_enabled = False
        
        # Apagar renderizado
        spectrogram_render_active = False
        
        if audio_detection_thread:
            audio_detection_thread.join(timeout=2)
        
        if spectrogram_render_thread:
            spectrogram_render_thread.join(timeout=2)
        
        # Limpiar datos
        with audio_spectrogram_lock:
            global audio_spectrogram_data, audio_spectrogram_freqs
            audio_spectrogram_data = None
            audio_spectrogram_freqs = None
            
        with spectrogram_image_lock:
            global spectrogram_image_ready
            spectrogram_image_ready = None
        
        print("Detección desactivada")

# --- CONFIGURACIÓN YOLO CON THREADING ---
yolo_model = None
yolo_enabled = False
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
YOLO_SCALE = 0.5  # Procesar al 50% de resolución

# Threading YOLO
yolo_frame_queue = queue.Queue(maxsize=2)
yolo_result_queue = queue.Queue(maxsize=2)
yolo_worker_thread = None
yolo_worker_running = False
yolo_result_lock = threading.Lock()
ultimo_resultado_yolo = {"frame": None, "detecciones": 0, "boxes_data": []}
yolo_conf_threshold = CONFIDENCE_THRESHOLD
yolo_iou_threshold = IOU_THRESHOLD
yolo_threshold_lock = threading.Lock()
yolo_reload_requested = False
yolo_settings_icon = None
yolo_slider_active = None
rf_slider_active = None


def apply_yolo_model(new_path, save_default=False, selected_slot=None):
    """Configura el modelo YOLO a usar y marca recarga si estaba activo."""
    global yolo_model_path, yolo_default_slot, yolo_model, yolo_model_slots, yolo_reload_requested

    if not new_path or not os.path.exists(new_path):
        print(f"[YOLO] Ruta de modelo inválida: {new_path}")
        return False

    yolo_model_path = new_path

    if save_default and selected_slot is not None:
        yolo_default_slot = selected_slot

    save_yolo_models_config()

    if yolo_enabled:
        yolo_reload_requested = True
    else:
        yolo_model = None

    print(f"[YOLO] Modelo activo: {yolo_model_path}")
    return True
def cargar_modelo_yolo():
    """Carga el modelo YOLO"""
    global yolo_model
    try:
        if not os.path.exists(yolo_model_path):
            print(f"Error: No se encuentra el modelo '{yolo_model_path}'")
            return False
        
        print("Cargando modelo YOLO...")
        yolo_model = YOLO(yolo_model_path)
        print(f"Modelo YOLO cargado - Dispositivo: {yolo_model.device}")
        return True
    except Exception as e:
        print(f"Error al cargar modelo YOLO: {e}")
        return False

def yolo_inference_worker():
    """Thread worker dedicado para inferencia YOLO"""
    global ultimo_resultado_yolo
    
    print("[YOLO] Worker thread iniciado")
    
    while yolo_worker_running:
        try:
            # Obtener frame de la cola (con timeout para poder salir limpiamente)
            frame_original, original_shape = yolo_frame_queue.get(timeout=0.1)
            
            if yolo_model is None:
                continue
            
            # Redimensionar para procesamiento más rápido
            small_frame = cv2.resize(frame_original, 
                                    (int(original_shape[1] * YOLO_SCALE), 
                                     int(original_shape[0] * YOLO_SCALE)))
            
            with yolo_threshold_lock:
                conf_thr = yolo_conf_threshold
                iou_thr = yolo_iou_threshold

            # Inferencia YOLO
            results = yolo_model(
                small_frame,
                verbose=False,
                conf=conf_thr,
                iou=iou_thr
            )
            
            boxes_data = []
            detecciones = 0
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Coordenadas escaladas al tamaño original
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 / YOLO_SCALE)
                    y1 = int(y1 / YOLO_SCALE)
                    x2 = int(x2 / YOLO_SCALE)
                    y2 = int(y2 / YOLO_SCALE)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = yolo_model.names[cls] if cls < len(yolo_model.names) else f"Class {cls}"
                    boxes_data.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': conf, 'class_name': class_name
                    })
                    detecciones += 1
            
            # Guardar resultado
            with yolo_result_lock:
                ultimo_resultado_yolo = {
                    "frame": frame_original,
                    "detecciones": detecciones,
                    "boxes_data": boxes_data
                }
            
            # Marcar tarea completada
            yolo_frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[YOLO] Error en worker: {e}")
    
    print("[YOLO] Worker thread finalizado")

def start_yolo_worker():
    """Inicia el thread worker de YOLO"""
    global yolo_worker_thread, yolo_worker_running
    
    if yolo_worker_thread is not None and yolo_worker_thread.is_alive():
        return
    
    yolo_worker_running = True
    yolo_worker_thread = threading.Thread(target=yolo_inference_worker, daemon=True)
    yolo_worker_thread.start()
    print("[YOLO] Thread worker iniciado")

def stop_yolo_worker():
    """Detiene el thread worker de YOLO"""
    global yolo_worker_running, yolo_worker_thread
    
    if yolo_worker_thread is None:
        return
    
    yolo_worker_running = False
    
    if yolo_worker_thread.is_alive():
        yolo_worker_thread.join(timeout=2)
    
    # Limpiar colas
    while not yolo_frame_queue.empty():
        try:
            yolo_frame_queue.get_nowait()
        except queue.Empty:
            break
    
    print("[YOLO] Thread worker detenido")

def toggle_yolo():
    """Activa o desactiva YOLO"""
    global yolo_enabled, yolo_model
    
    print(f"[DEBUG] toggle_yolo llamado. Estado actual: {yolo_enabled}")

    if not yolo_enabled:
        if yolo_model is None:
            print("[DEBUG] Cargando modelo YOLO...")
            if not cargar_modelo_yolo():
                print("[ERROR] Fallo al cargar modelo YOLO")
                return
        
        print("[DEBUG] Iniciando worker YOLO...")
        start_yolo_worker()
        yolo_enabled = True
        print("YOLO activado")
    else:
        yolo_enabled = False
        stop_yolo_worker()
        
        # Limpiar último resultado
        with yolo_result_lock:
            global ultimo_resultado_yolo
            ultimo_resultado_yolo = {"frame": None, "detecciones": 0, "boxes_data": []}
        
        print("YOLO desactivado")

def enviar_frame_a_yolo(frame):
    """Envía frame a YOLO solo si no está ocupado"""
    if not yolo_enabled:
        return
    
    try:
        # Enviar sin bloquear - si la cola está llena, se salta el frame
        yolo_frame_queue.put_nowait((frame.copy(), frame.shape))
    except queue.Full:
        pass  # YOLO ocupado, saltar este frame

def obtener_resultado_yolo():
    """Obtiene el último resultado de YOLO disponible"""
    with yolo_result_lock:
        return ultimo_resultado_yolo.copy()

def dibujar_detecciones_yolo(frame, boxes_data):
    """Dibuja las detecciones YOLO en el frame"""
    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = box['conf']
        class_name = box['class_name']
        
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_name}: {conf:.2f}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    return frame

# --- CLASE DE CAPTURA DE VIDEO ROBUSTA ---
class ThreadedVideoCapture:
    """
    Captura de video optimizada con reconexión segura.
    """
    
    def __init__(self, src):
        self.src = src
        self.successful_init = False
        self.stopped = False
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.init_time = time.time() # Hora de inicio del objeto
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        
        # Inicialización del recurso de video
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)
            
        # Configuración básica
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass
            
        # VERIFICACIÓN CRÍTICA INICIAL
        if not self.cap.isOpened():
            print("[VIDEO] Error: No se pudo abrir el stream.")
            return
            
        self.successful_init = True
        
        # Iniciar thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        print("[VIDEO] Captura Low-Latency iniciada")
    
    def is_valid(self):
        """Retorna True si la captura se inicializó correctamente"""
        return self.successful_init
    
    def _update(self):
        """Loop de lectura"""
        while not self.stopped:
            if self.cap.isOpened():
                grabbed = self.cap.grab()
                if grabbed:
                    self.ret, frame = self.cap.retrieve()
                    if self.ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.frame_id += 1
                            self.last_frame_time = time.time()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.1)
    
    def read(self):
        """Devuelve frame con lógica inteligente de espera"""
        if not self.successful_init:
            return False, None, -1

        with self.lock:
            # SI TENEMOS FRAME (Caso normal)
            if self.frame is not None:
                # Watchdog estricto: Si hace más de 3s que no hay nada nuevo -> Muerte
                if time.time() - self.last_frame_time > 3.0:
                      return False, None, -1
                return True, self.frame.copy(), self.frame_id

            # SI NO TENEMOS FRAME (Caso inicial / Cargando)
            # Si estamos en los primeros 5 segundos de vida, decimos "Todo OK, espera"
            if time.time() - self.init_time < 5.0:
                return True, None, -1 # Retorna True (vivo) pero None (vacío)
            
            # Si pasaron 5 segundos y sigue sin haber frame -> Muerte
            return False, None, -1
    
    def release(self):
        self.stopped = True
        if self.successful_init and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap.isOpened():
            self.cap.release()
        print("[VIDEO] Captura liberada")


def schedule_video_connection(target_url, force=False):
    """
    Lanza un intento de conexión en background para no bloquear la UI.
    """
    global video_connection_attempts

    if not force:
        for attempt in video_connection_attempts:
            if attempt.get("url") == target_url and attempt.get("thread") and attempt["thread"].is_alive():
                return

    result_queue = queue.Queue(maxsize=1)

    def worker(url, result_queue):
        new_cap = None
        try:
            print(f"[VIDEO] Intentando conectar a {url} (async)...")
            new_cap = ThreadedVideoCapture(url)
            if not new_cap.is_valid():
                new_cap.release()
                new_cap = None
        except Exception as e:
            print(f"[VIDEO] Error al iniciar conexión: {e}")
            if new_cap:
                new_cap.release()
                new_cap = None
        finally:
            try:
                result_queue.put((url, new_cap), timeout=1)
            except queue.Full:
                if new_cap:
                    new_cap.release()

    thread = threading.Thread(target=worker, args=(target_url, result_queue), daemon=True)
    video_connection_attempts.append({"thread": thread, "queue": result_queue, "url": target_url})
    thread.start()


def process_pending_video_connections(current_cap, current_url):
    """
    Revisa conexiones finalizadas y retorna (cap, se_asignó_nuevo_cap).
    """
    global video_connection_attempts

    if not video_connection_attempts:
        return current_cap, False

    new_cap_assigned = False
    remaining_attempts = []

    for attempt in video_connection_attempts:
        q = attempt["queue"]
        try:
            result_url, candidate_cap = q.get_nowait()
        except queue.Empty:
            remaining_attempts.append(attempt)
            continue

        if candidate_cap and candidate_cap.is_valid() and current_cap is None and result_url == current_url and not new_cap_assigned:
            current_cap = candidate_cap
            new_cap_assigned = True
            print(f"[VIDEO] Conexión establecida con {result_url}")
        else:
            if candidate_cap:
                candidate_cap.release()

    video_connection_attempts = remaining_attempts
    return current_cap, new_cap_assigned


def ensure_windows_cursor(window_title):
    """
    Reemplaza el cursor en ventanas OpenCV por el puntero clásico.
    """
    global windows_cursor_fixed, windows_cursor_warning

    if windows_cursor_fixed or os.name != "nt":
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return

        IDC_ARROW = 32512
        cursor = user32.LoadCursorW(None, IDC_ARROW)
        GCL_HCURSOR = -12

        if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_long):
            user32.SetClassLongW(hwnd, GCL_HCURSOR, cursor)
        else:
            user32.SetClassLongPtrW(hwnd, GCL_HCURSOR, cursor)
        user32.SetCursor(cursor)
        windows_cursor_fixed = True
        print("[WINDOWS] Cursor estándar aplicado.")
    except Exception as e:
        if not windows_cursor_warning:
            print(f"[WINDOWS] No se pudo ajustar el cursor: {e}")
            windows_cursor_warning = True

# --- VISUALIZACIÓN OPTIMIZADA ---

def overlay_audio_spectrogram(frame):
    """
    Superpone el espectrograma.
    """
    if not spectrogram_render_active:
        return frame
    
    # Intentar obtener la imagen más reciente del worker
    spectrogram_img = None
    with spectrogram_image_lock:
        if spectrogram_image_ready is not None:
            spectrogram_img = spectrogram_image_ready.copy()
            
    if spectrogram_img is None:
        return frame
    
    try:
        h, w = frame.shape[:2]
        oh, ow = spectrogram_img.shape[:2]
        
        # Redimensionar si es necesario (el worker ya lo hace aprox, pero ajustamos al frame actual)
        target_h = int(h * 0.22)
        target_w = int(w * 0.35)
        
        if oh != target_h or ow != target_w:
             spectrogram_img = cv2.resize(spectrogram_img, (target_w, target_h))
             oh, ow = target_h, target_w

        y_offset = 110
        x_offset = 10
        
        # Alpha blending
        alpha = (spectrogram_img[:, :, 3] / 255.0) * 0.5
        
        for c in range(3):
            frame[y_offset:y_offset+oh, x_offset:x_offset+ow, c] = \
                frame[y_offset:y_offset+oh, x_offset:x_offset+ow, c] * (1 - alpha) + \
                spectrogram_img[:, :, c] * alpha
        
    except Exception as e:
        pass # Ignorar errores puntuales de renderizado para no congelar video
    
    return frame

# --- AUDIO STREAMING ---
def stream_audio():
    global audio_stream, stop_audio_thread
    
    try:
        with requests.get(audio_url, stream=True, timeout=10, headers=headers) as r:
            if r.status_code != 200:
                return
            
            header = r.raw.read(44)
            
            audio_stream = p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  output=True,
                                  frames_per_buffer=CHUNK)
            
            for chunk in r.iter_content(chunk_size=CHUNK):
                if stop_audio_thread:
                    break
                if chunk and audio_stream:
                    try:
                        audio_stream.write(chunk)
                        
                        if audio_detection_enabled:
                            try:
                                audio_buffer.put_nowait(chunk)
                            except queue.Full:
                                pass
                                
                    except Exception as e:
                        break
                        
    except Exception as e:
        print(f"Error audio: {e}")
    finally:
        if audio_stream:
            try:
                audio_stream.stop_stream()
                audio_stream.close()
                audio_stream = None
            except:
                pass

def start_audio():
    global audio_thread, stop_audio_thread, audio_enabled
    
    if audio_thread is not None and audio_thread.is_alive():
        return
    
    stop_audio_thread = False
    audio_enabled = True
    audio_thread = threading.Thread(target=stream_audio, daemon=True)
    audio_thread.start()
    print("Audio iniciado")

def stop_audio():
    global stop_audio_thread, audio_enabled, audio_thread, audio_detection_enabled
    
    if audio_thread is None or not audio_thread.is_alive():
        return
    
    if audio_detection_enabled:
        toggle_audio_detection()
    
    stop_audio_thread = True
    audio_enabled = False
    
    if audio_thread:
        audio_thread.join(timeout=2)
    
    print("Audio detenido")

def cambiar_ip_camara(cap_actual, nueva_ip=None):
    if audio_enabled:
        stop_audio()
    
    if cap_actual is not None:
        cap_actual.release()
    
    if nueva_ip is None:
        nueva_ip = solicitar_nueva_ip(ip_y_puerto)
    
    if nueva_ip:
        update_stream_endpoints(nueva_ip, record_wifi=True)
        schedule_video_connection(video_url, force=True)

    return None

# --- INDICADORES ---
def draw_interactive_button(frame, text, x_start, y_center, w, h, text_color, mouse_pos, click_pos, align_right=False):
    """
    Dibuja un botón redondeado transparente con efecto hover y detección de clic.
    Retorna: (frame_modificado, fue_cliqueado)
    """
    # Coordenadas de la caja
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x = 20
    padding_y = 12

    width = text_size[0] + padding_x
    height = text_size[1] + padding_y
    if w > 0:
        width = max(width, w)
    if h > 0:
        height = max(height, h)

    y1 = int(y_center - text_size[1] - padding_y / 2)
    y2 = y1 + int(height)

    if align_right:
        x2 = int(x_start)
        x1 = x2 - int(width)
        text_x = x1 + padding_x // 2
    else:
        x1 = int(x_start)
        x2 = x1 + int(width)
        text_x = x1 + padding_x // 2
    
    # Detectar Hover
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2) and (y1 <= my <= y2)
    
    # Detectar Clic
    is_clicked = False
    if click_pos:
        cx, cy = click_pos
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            is_clicked = True

    # Configuración visual
    overlay = frame.copy()
    bg_color = (0, 0, 0)
    alpha = 0.6 if is_hover else 0.4 # Más opaco si pasas el ratón
    radius = 10
    
    # Dibujar rectángulo redondeado
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, bg_color, -1)
    
    # Aplicar transparenciaopen_
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Texto
    text_y = y2 - padding_y // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    return frame, is_clicked

def draw_audio_indicator(frame, mouse_pos, click_pos):
    x = frame.shape[1] - 40
    y = 20
    
    if audio_enabled and audio_thread and audio_thread.is_alive():
        color = (0, 255, 0)
        text = "TRANSMISION DE AUDIO: ON"
    else:
        color = (0, 0, 255)
        text = "TRANSMISION DE AUDIO: OFF"
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_yolo_indicator(frame, mouse_pos, click_pos, detecciones=0):
    x = frame.shape[1] - 40
    y = 50
    
    if yolo_enabled:
        color = (0, 255, 0)
        text = f"YOLO: {detecciones} det."
    else:
        color = (0, 0, 255)
        text = "YOLO: OFF"
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_yolo_settings_icon(frame, mouse_pos, click_pos):
    """Dibuja el icono PNG de ajustes para YOLO."""
    icon = get_yolo_settings_icon()
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    padding = 10
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 45 - h // 2
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked

def draw_tinysa_indicator(frame, mouse_pos, click_pos):
    x = frame.shape[1] - 40
    y = 80
    
    if tinysa_running:
        color = (0, 255, 0)
        text = "TinySA: ON"
    else:
        color = (0, 0, 255)
        text = "TinySA: OFF"
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_audio_detection_toggle(frame, mouse_pos, click_pos):
    x = frame.shape[1] - 40
    y = 110

    if audio_detection_enabled:
        color = (0, 255, 0)
        text = "DET AUDIO: ON"
    else:
        color = (0, 0, 255)
        text = "DET AUDIO: OFF"

    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def open_yolo_options_dialog():
    """Abre la ventana de opciones YOLO en un hilo aparte."""
    global yolo_options_thread
    if yolo_options_thread and yolo_options_thread.is_alive():
        return

    def runner():
        try:
            show_yolo_options_window()
        finally:
            yolo_options_thread = None

    yolo_options_thread = threading.Thread(target=runner, daemon=True)
    yolo_options_thread.start()


def draw_ip_indicator(frame, mouse_pos, click_pos):
    x = 10
    y = 20
    text = f"IP: {ip_y_puerto}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    padding_x = 14
    padding_y = 10
    x1 = x - 6
    y1 = y - text_size[1] - padding_y // 2
    x2 = x1 + text_size[0] + padding_x
    y2 = y + padding_y // 2

    x1 = max(0, x1)
    y1 = max(0, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return frame, False


def draw_adb_message(frame):
    if not adb_connected:
        return frame
    text = "ADB conectado"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 85), font, 0.6, (0, 255, 255), 2)
    return frame


def draw_tinysa_message(frame):
    if not tinysa_detected:
        return frame
    # Mostrar mensaje diferente según el modo de conexión
    if tinysa_use_http:
        text = "TinySA conectado a Android"
    else:
        text = "TinySA conectado"
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Posición abajo a la izquierda
    x = 10
    y = frame.shape[0] - 15  # Abajo del frame
    cv2.putText(frame, text, (x, y), font, 0.55, (0, 255, 255), 2)
    
    # Mostrar alerta de dron detectado por RF
    with rf_drone_detection_lock:
        rf_result = rf_drone_detection_result.copy()
    
    if rf_result.get("is_drone", False) and rf_drone_detection_enabled:
        confidence = rf_result.get("confidence", 0.0)
        frequency = rf_result.get("frequency")
        
        # Solo mostrar aviso si la confianza es >= 60%
        if confidence < 0.6:
            return frame
        
        if frequency:
            freq_mhz = frequency / 1e6
            alert_text = f"DRON DETECTADO POR RF: {freq_mhz:.3f} GHz ({int(confidence * 100)}%)"
        else:
            alert_text = f"DRON DETECTADO POR RF ({int(confidence * 100)}%)"
        
        # Dibujar fondo semitransparente rojo para la alerta
        text_size, _ = cv2.getTextSize(alert_text, font, 0.7, 2)
        text_w, text_h = text_size
        alert_x = 10
        alert_y = y - text_h - 25
        
        # Rectángulo de fondo
        cv2.rectangle(
            frame,
            (alert_x - 5, alert_y - 5),
            (alert_x + text_w + 5, alert_y + text_h + 5),
            (0, 0, 255),  # Rojo
            -1
        )
        
        # Texto de alerta
        cv2.putText(
            frame,
            alert_text,
            (alert_x, alert_y + text_h),
            font,
            0.7,
            (255, 255, 255),  # Blanco
            2
        )
    
    return frame

def draw_ip_settings_icon(frame, mouse_pos, click_pos):
    icon = get_yolo_settings_icon()
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(f"IP: {ip_y_puerto}", font, 0.5, 2)
    x2 = 10 + text_size[0] + 20 + w
    x1 = x2 - w
    y_center = 15
    y1 = y_center - h // 2
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked


def open_ip_change_dialog():
    global ip_dialog_thread
    if ip_dialog_thread and ip_dialog_thread.is_alive():
        return

    def runner():
        global pending_ip_change, ip_dialog_thread
        nueva_ip = solicitar_nueva_ip(ip_y_puerto)
        if nueva_ip and nueva_ip.strip():
            pending_ip_change = nueva_ip.strip()
        ip_dialog_thread = None

    ip_dialog_thread = threading.Thread(target=runner, daemon=True)
    ip_dialog_thread.start()


def apply_pending_ip_change(cap_actual):
    global pending_ip_change
    if pending_ip_change:
        nueva_ip = pending_ip_change
        pending_ip_change = None
        cap_actual = cambiar_ip_camara(cap_actual, nueva_ip=nueva_ip)
    return cap_actual


def setup_adb_forward():
    try:
        subprocess.run(["adb", "forward", "--remove", "tcp:8080"], capture_output=True, text=True, timeout=3)
    except Exception:
        pass
    try:
        subprocess.run(["adb", "forward", "tcp:8080", "tcp:8080"], check=True, capture_output=True, text=True, timeout=3)
        return True
    except Exception as e:
        print(f"[ADB] Error configurando forward: {e}")
        return False


def teardown_adb_forward():
    try:
        subprocess.run(["adb", "forward", "--remove", "tcp:8080"], capture_output=True, text=True, timeout=3)
    except Exception:
        pass


def poll_adb_connection():
    global last_adb_check, adb_connected, pending_ip_change, adb_message_timer, last_wifi_ip

    now = time.time()
    if now - last_adb_check < ADB_CHECK_INTERVAL:
        return
    last_adb_check = now

    if shutil.which("adb") is None:
        if adb_connected:
            adb_connected = False
        return

    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=3
        )
        lines = result.stdout.strip().splitlines()
        connected = any("\tdevice" in line for line in lines[1:])
    except Exception:
        connected = False

    if connected and not adb_connected:
        if last_wifi_ip is None or (not last_wifi_ip or last_wifi_ip == ADB_TARGET_IP):
            last_wifi_ip = ip_y_puerto
        if setup_adb_forward():
            adb_connected = True
            pending_ip_change = ADB_TARGET_IP
            print("[ADB] Conectado. Cambiando IP a 127.0.0.1.")
        else:
            print("[ADB] Fallo configurando el túnel. Mantengo IP actual.")
    elif not connected and adb_connected:
        adb_connected = False
        teardown_adb_forward()
        if last_wifi_ip and last_wifi_ip != ADB_TARGET_IP:
            pending_ip_change = last_wifi_ip
            print("[ADB] Desconectado. Volviendo a IP anterior.")


def poll_tinysa_presence(force=False):
    """
    Verifica si TinySA está conectado (localmente vía USB o en el servidor Android).
    """
    global tinysa_last_check, tinysa_detected, tinysa_use_http
    now = time.time()
    if not force and now - tinysa_last_check < TIN_YSA_CHECK_INTERVAL:
        return
    tinysa_last_check = now
    
    # Verificar puerto local primero
    port = find_tinysa_port()
    if port is not None:
        tinysa_detected = True
        tinysa_use_http = False
    else:
        # Si no está localmente, verificar servidor Android
        try:
            status_url = base_url + "/tinysa/status"
            response = requests.get(status_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                is_connected = data.get("connected", False)
                tinysa_detected = is_connected
                if is_connected:
                    tinysa_use_http = True
                else:
                    tinysa_detected = False
            else:
                tinysa_detected = False
        except Exception:
            tinysa_detected = False

def show_yolo_options_window():
    """Ventana para gestionar modelos YOLO."""
    global yolo_model_slots, yolo_default_slot

    root = tk.Tk()
    root.title("Opciones de YOLO")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text="Modelos disponibles", font=("Arial", 11, "bold")).pack(anchor="w")

    slots_frame = ttk.Frame(main_frame)
    slots_frame.pack(fill="both", expand=True, pady=(10, 15))

    total_slots = len(yolo_model_slots)
    slots_per_page = 5
    total_pages = max(1, (total_slots + slots_per_page - 1) // slots_per_page)
    current_page = tk.IntVar(value=0)

    path_vars = [tk.StringVar(value=slot.get("path", "")) for slot in yolo_model_slots]
    desc_vars = [tk.StringVar(value=slot.get("description", "")) for slot in yolo_model_slots]
    selected_var = tk.IntVar(value=yolo_default_slot)

    def browse_file(idx):
        filepath = filedialog.askopenfilename(
            title="Seleccionar modelo YOLO",
            filetypes=[("Modelos YOLO (*.pt)", "*.pt"), ("Todos los archivos", "*.*")],
            parent=root
        )
        if filepath:
            path_vars[idx].set(filepath)

    total_slots = len(yolo_model_slots)
    slots_per_page = 5
    total_pages = max(1, (total_slots + slots_per_page - 1) // slots_per_page)
    current_page = tk.IntVar(value=0)

    rows_container = ttk.Frame(slots_frame)
    rows_container.pack(fill="both", expand=True)

    def build_page(page_idx):
        for child in rows_container.winfo_children():
            child.destroy()

        start_idx = page_idx * slots_per_page
        end_idx = min(start_idx + slots_per_page, total_slots)

        for idx in range(start_idx, end_idx):
            frame_slot = ttk.Frame(rows_container, padding=5)
            frame_slot.pack(fill="x", pady=3)

            ttk.Radiobutton(frame_slot, variable=selected_var, value=idx).grid(row=0, column=0, rowspan=2, padx=(0, 8))
            ttk.Label(frame_slot, text=f"Modelo {idx + 1}").grid(row=0, column=1, sticky="w")
            entry_path = ttk.Entry(frame_slot, textvariable=path_vars[idx], width=45)
            entry_path.grid(row=0, column=2, padx=5, sticky="we")
            ttk.Button(frame_slot, text="Examinar", command=lambda i=idx: browse_file(i)).grid(row=0, column=3, padx=5)
            ttk.Label(frame_slot, text="Descripción:").grid(row=1, column=1, sticky="e", pady=2)
            ttk.Entry(frame_slot, textvariable=desc_vars[idx], width=45).grid(row=1, column=2, padx=5, sticky="we")
            frame_slot.columnconfigure(2, weight=1)

    nav_frame = ttk.Frame(main_frame)
    nav_frame.pack(fill="x", pady=(5, 5))

    page_label_var = tk.StringVar()

    def update_page_label():
        page_label_var.set(f"Página {current_page.get() + 1}/{total_pages}")

    def go_prev():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            build_page(current_page.get())
            update_page_label()

    def go_next():
        if current_page.get() < total_pages - 1:
            current_page.set(current_page.get() + 1)
            build_page(current_page.get())
            update_page_label()

    ttk.Button(nav_frame, text="◀", width=3, command=go_prev).pack(side="left")
    ttk.Label(nav_frame, textvariable=page_label_var).pack(side="left", padx=10)
    ttk.Button(nav_frame, text="▶", width=3, command=go_next).pack(side="left")

    build_page(0)
    update_page_label()

    status_var = tk.StringVar(value="")

    def sync_slots():
        for idx in range(len(yolo_model_slots)):
            yolo_model_slots[idx]["path"] = path_vars[idx].get().strip()
            yolo_model_slots[idx]["description"] = desc_vars[idx].get().strip()

    def apply_action(save_default=False, reset_default=False):
        sync_slots()
        slot_idx = selected_var.get()

        if reset_default:
            yolo_model_slots[0]["path"] = YOLO_DEFAULT_MODEL
            yolo_model_slots[0]["description"] = "Modelo por defecto"
            path_vars[0].set(YOLO_DEFAULT_MODEL)
            desc_vars[0].set("Modelo por defecto")
            slot_idx = 0
            save_default = True

        path = yolo_model_slots[slot_idx]["path"]
        if not path:
            messagebox.showerror("Error", f"El modelo del slot {slot_idx + 1} está vacío.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Error", f"No se encontró el archivo:\n{path}")
            return

        if apply_yolo_model(path, save_default=save_default, selected_slot=slot_idx if save_default else None):
            status_var.set("Modelo actualizado correctamente.")
            root.destroy()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(5, 10))

    ttk.Button(btn_frame, text="Cargar modelo", command=lambda: apply_action(False)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Cargar y guardar por defecto", command=lambda: apply_action(True)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Cargar configuración por defecto", command=lambda: apply_action(True, True)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Cancelar", command=root.destroy).pack(side="right", padx=5)

    ttk.Label(main_frame, textvariable=status_var, foreground="#0077cc").pack(anchor="w")

    root.mainloop()


def draw_slider_control(frame, label, value, min_val, max_val, origin, size, mouse_pos, click_pos, slider_key):
    """Dibuja un slider semi-transparente y devuelve nuevo valor si se hizo click."""
    global yolo_slider_active, rf_slider_active
    x, y = origin
    width, height = size
    overlay = frame.copy()

    # Panel
    panel_y1 = y
    panel_y2 = y + height
    cv2.rectangle(overlay, (x, panel_y1), (x + width, panel_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Textos
    text_y = panel_y1 + 18
    cv2.putText(frame, label, (x + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
    cv2.putText(frame, f"{value:.2f}", (x + width - 55, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    slider_offset = 10
    panel_y1 += slider_offset
    panel_y2 += slider_offset

    # Slider track semitransparente
    track_x1 = x + 20
    track_x2 = x + width - 20
    track_y = panel_y1 + height - 25
    track_overlay = frame.copy()
    cv2.line(track_overlay, (track_x1, track_y), (track_x2, track_y), (210, 210, 210), 6, cv2.LINE_AA)
    cv2.addWeighted(track_overlay, 0.4, frame, 0.6, 0, frame)

    # Handle
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    handle_x = int(track_x1 + ratio * (track_x2 - track_x1))
    handle_overlay = frame.copy()
    cv2.circle(handle_overlay, (handle_x, track_y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(handle_overlay, (handle_x, track_y), 10, (0, 102, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(handle_overlay, 0.7, frame, 0.3, 0, frame)

    # Clic
    new_value = None
    active_slider = yolo_slider_active if slider_key.startswith("conf") or slider_key.startswith("iou") else rf_slider_active
    
    if click_pos:
        cx, cy = click_pos
        if track_y - 18 <= cy <= track_y + 18 and track_x1 <= cx <= track_x2:
            ratio = (cx - track_x1) / (track_x2 - track_x1)
            ratio = max(0.0, min(1.0, ratio))
            new_value = min_val + ratio * (max_val - min_val)
            if slider_key.startswith("rf_"):
                rf_slider_active = slider_key
            else:
                yolo_slider_active = slider_key
    elif active_slider == slider_key and mouse_is_down:
        mx, my = mouse_pos
        ratio = (mx - track_x1) / (track_x2 - track_x1)
        ratio = max(0.0, min(1.0, ratio))
        new_value = min_val + ratio * (max_val - min_val)
    elif active_slider == slider_key and not mouse_is_down:
        if slider_key.startswith("rf_"):
            rf_slider_active = None
        else:
            yolo_slider_active = None

    return frame, new_value


def draw_yolo_sliders(frame, mouse_pos, click_pos):
    """Muestra sliders de parámetros YOLO si está activado."""
    global yolo_conf_threshold, yolo_iou_threshold
    if not yolo_enabled:
        return frame, click_pos

    slider_width = int(frame.shape[1] * 0.16)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Confidence threshold", yolo_conf_threshold, 0.05, 0.99, "conf"),
        ("IoU threshold", yolo_iou_threshold, 0.05, 0.99, "iou"),
    ]

    # Working copy of click to avoid multiple updates con el mismo clic
    remaining_click = click_pos

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame,
            label,
            value,
            v_min,
            v_max,
            (x, y),
            (slider_width, slider_height),
            mouse_pos,
            remaining_click,
            key
        )

        if new_val is not None:
            with yolo_threshold_lock:
                if key == "conf":
                    yolo_conf_threshold = new_val
                else:
                    yolo_iou_threshold = new_val
            remaining_click = None

    return frame, remaining_click

def draw_rf_drone_sliders(frame, mouse_pos, click_pos):
    """Muestra sliders de parámetros de detección RF de drones si está activado."""
    global rf_peak_threshold, rf_min_peak_height_db, rf_min_peak_width_mhz, rf_max_peak_width_mhz
    global rf_sliders_visible
    
    if not rf_sliders_visible or not tinysa_running:
        return frame, click_pos

    slider_width = int(frame.shape[1] * 0.20)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Umbral Potencia (dBm)", rf_peak_threshold, -100.0, -50.0, "rf_peak_thresh"),
        ("Altura Min Ruido (dB)", rf_min_peak_height_db, 1.0, 40.0, "rf_min_height"),
        ("Ancho Min (MHz)", rf_min_peak_width_mhz, 1.0, 30.0, "rf_min_width"),
        ("Ancho Max (MHz)", rf_max_peak_width_mhz, 20.0, 80.0, "rf_max_width"),
    ]

    # Working copy of click to avoid multiple updates con el mismo clic
    remaining_click = click_pos

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame,
            label,
            value,
            v_min,
            v_max,
            (x, y),
            (slider_width, slider_height),
            mouse_pos,
            remaining_click,
            key
        )

        if new_val is not None:
            with rf_detection_params_lock:
                if key == "rf_peak_thresh":
                    rf_peak_threshold = new_val
                elif key == "rf_min_height":
                    rf_min_peak_height_db = new_val
                elif key == "rf_min_width":
                    rf_min_peak_width_mhz = new_val
                elif key == "rf_max_width":
                    rf_max_peak_width_mhz = new_val
            print(f"[RF SLIDER] {label}: {new_val:.2f}")
            remaining_click = None  # Consumir el click

    return frame, remaining_click

def draw_audio_detection_indicator(frame):
    if not audio_detection_enabled:
        return frame
    
    with audio_detection_lock:
        is_drone = audio_detection_result["is_drone"]
        confidence = audio_detection_result["confidence"]
    
    x = frame.shape[1] - 300
    y = frame.shape[0] - 30
    
    if is_drone:
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 255, 255) if blink else (0, 128, 128)
        text = f"AUDIO DRON DETECTED: {int(confidence * 100)}%"
    else:
        color = (0, 0, 255)
        text = f"NO AUDIO DRON: {int(confidence * 100)}%"
    
    overlay = frame.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(overlay, (x - 5, y - 23), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def process_pending_yolo_reload():
    """Reinicia YOLO en el hilo principal si hay cambios de modelo pendientes."""
    global yolo_reload_requested
    if yolo_reload_requested and yolo_enabled:
        yolo_reload_requested = False
        print("[YOLO] Recargando modelo seleccionado...")
        toggle_yolo()
        toggle_yolo()
    elif yolo_reload_requested:
        yolo_reload_requested = False
        # YOLO apagado: solo marcamos para cargar en siguiente activación
        print("[YOLO] Modelo actualizado para próximo inicio.")

def draw_fps_indicator(frame, fps):
    x = 10
    y = 50
    text = f"FPS: {fps:.1f}"
    
    overlay = frame.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(overlay, (x - 5, y - 18), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# --- MAIN ---
print("Iniciando programa FULL THREADED + TinySA Ultra (Modo Síncrono)...")
print("Controles:")
print("  Q - Salir")
print("  M - Audio ON/OFF")
print("  A - Detección audio ON/OFF")
print("  Y - YOLO ON/OFF")
print("  R - Sliders RF ON/OFF")
print("  T - TinySA (RF) ON/OFF")
print("  I - Cambiar IP")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
DEFAULT_WINDOW_SIZE = (1280, 720)
current_window_size = list(DEFAULT_WINDOW_SIZE)
cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
# IMPRESCINDIBLE: Activar el callback del ratón
cv2.setMouseCallback(window_name, mouse_handler)
ensure_windows_cursor(window_name)

cap = None
stop_program = False

schedule_video_connection(video_url, force=True)

detecciones_count = 0
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0.0
prev_frame_id = -1
last_reconnect_try = 0  # Timer para no saturar la reconexión

while not stop_program:
    # 1. Gestión del ratón al inicio del frame
    current_click = click_event_pos
    click_event_pos = None  # Resetear clic
    current_mouse = (mouse_x, mouse_y)
    poll_adb_connection()
    poll_tinysa_presence()
    ensure_windows_cursor(window_name)

    cap, new_cap_ready = process_pending_video_connections(cap, video_url)
    if new_cap_ready:
        fps_start_time = time.time()
        last_reconnect_try = time.time()

    if cap is None:
        # MOSTRAR PANTALLA DE ESPERA (NO SIGNAL)
        frame_negro = np.zeros((DEFAULT_WINDOW_SIZE[1], DEFAULT_WINDOW_SIZE[0], 3), dtype=np.uint8)
        texto = "NO SIGNAL"
        texto2 = "Intentando reconectar..."
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(texto, font, 1.5, 2)
        cv2.putText(frame_negro, texto, ((640-tw)//2, (480+th)//2 - 20), font, 1.5, (255, 255, 255), 2)
        
        (tw2, th2), _ = cv2.getTextSize(texto2, font, 0.7, 1)
        cv2.putText(frame_negro, texto2, ((640-tw2)//2, (480+th2)//2 + 30), font, 0.7, (200, 200, 200), 1)

        # Overlay TinySA incluso sin vídeo
        frame_negro = overlay_tinysa_graph(frame_negro)

        # Controles
        if yolo_enabled:
            frame_negro, current_click = draw_yolo_sliders(frame_negro, current_mouse, current_click)
        if tinysa_running:
            frame_negro, current_click = draw_rf_drone_sliders(frame_negro, current_mouse, current_click)

        frame_negro, _ = draw_ip_indicator(frame_negro, current_mouse, current_click)
        frame_negro = draw_adb_message(frame_negro)
        frame_negro, ip_settings_clicked = draw_ip_settings_icon(frame_negro, current_mouse, current_click)
        if ip_settings_clicked:
            open_ip_change_dialog()
            current_click = None
             
        frame_negro, tinysa_clicked = draw_tinysa_indicator(frame_negro, current_mouse, current_click)
        if tinysa_clicked:
             toggle_tinysa()
        
        frame_negro, _ = draw_audio_indicator(frame_negro, current_mouse, current_click)
        frame_negro, _ = draw_yolo_indicator(frame_negro, current_mouse, current_click)
        frame_negro, yolo_settings_clicked = draw_yolo_settings_icon(frame_negro, current_mouse, current_click)
        if yolo_settings_clicked:
            open_yolo_options_dialog()
            current_click = None
        frame_negro, _ = draw_audio_detection_toggle(frame_negro, current_mouse, current_click)

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame_negro = draw_adb_message(frame_negro)
        frame_negro = draw_tinysa_message(frame_negro)
        
        if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
            cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
            current_window_size[:] = DEFAULT_WINDOW_SIZE
        cv2.imshow(window_name, frame_negro)
        
        # GESTIÓN DE TECLAS EN MODO NO-SIGNAL
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            stop_program = True
            break
        elif key == ord('i') or key == ord('I'):
             cap = cambiar_ip_camara(None)
             continue
        elif key == ord('t') or key == ord('T'):
             toggle_tinysa()
             continue

        # INTENTO DE RECONEXIÓN CONTROLADO
        if time.time() - last_reconnect_try > 2.0:
            last_reconnect_try = time.time()
            schedule_video_connection(video_url)
        
    else:
        # MODO CONECTADO
        ret, frame, current_id = cap.read()
        
        if not ret:
            print("Señal perdida. Cerrando captura...")
            cap.release()
            cap = None
            last_reconnect_try = time.time()
            schedule_video_connection(video_url, force=True)
            continue
        
        if frame is None:
            if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
                cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
                current_window_size[:] = DEFAULT_WINDOW_SIZE
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            continue

        # ESCALAR EL FRAME A 1280x720 ANTES DE PROCESARLO
        # Esto asegura que todas las funciones de dibujo trabajen con el tamaño fijo
        if frame.shape[:2] != (DEFAULT_WINDOW_SIZE[1], DEFAULT_WINDOW_SIZE[0]):
            frame = cv2.resize(frame, DEFAULT_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
        
        if current_id != prev_frame_id:
            fps_frame_count += 1
            prev_frame_id = current_id
            
            if yolo_enabled:
                # Enviar el frame escalado a YOLO (YOLO redimensiona internamente para procesamiento)
                enviar_frame_a_yolo(frame)
        
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        resultado_yolo = obtener_resultado_yolo()
        if resultado_yolo["boxes_data"]:
            # Las detecciones ya están en el tamaño correcto (1280x720) porque YOLO procesó el frame escalado
            frame = dibujar_detecciones_yolo(frame, resultado_yolo["boxes_data"])
            detecciones_count = resultado_yolo["detecciones"]
        else:
            detecciones_count = 0
        
        # Renderizado capas
        frame = overlay_audio_spectrogram(frame)
        frame = overlay_tinysa_graph(frame) 
        
        # --- DIBUJAR INDICADORES INTERACTIVOS ---
        if yolo_enabled:
            frame, current_click = draw_yolo_sliders(frame, current_mouse, current_click)
        if tinysa_running:
            frame, current_click = draw_rf_drone_sliders(frame, current_mouse, current_click)

        # 1. Audio
        frame, audio_clicked = draw_audio_indicator(frame, current_mouse, current_click)
        if audio_clicked:
            if audio_enabled: stop_audio()
            else: start_audio()
            
        # 2. YOLO
        frame, yolo_clicked = draw_yolo_indicator(frame, current_mouse, current_click, detecciones_count)
        if yolo_clicked:
            toggle_yolo()
        frame, yolo_settings_clicked = draw_yolo_settings_icon(frame, current_mouse, current_click)
        if yolo_settings_clicked:
            open_yolo_options_dialog()
            current_click = None
            
        # 3. TinySA
        frame, tinysa_clicked = draw_tinysa_indicator(frame, current_mouse, current_click)
        if tinysa_clicked:
            toggle_tinysa()
            
        # 4. Detección audio
        frame, audio_det_clicked = draw_audio_detection_toggle(frame, current_mouse, current_click)
        if audio_det_clicked:
            if audio_enabled:
                toggle_audio_detection()
            else:
                print("Activa el audio (botón AUDIO) antes de habilitar la detección.")
                click_event_pos = None

        # 5. IP
        frame, _ = draw_ip_indicator(frame, current_mouse, current_click)
        frame = draw_adb_message(frame)
        frame, ip_settings_clicked = draw_ip_settings_icon(frame, current_mouse, current_click)
        if ip_settings_clicked:
            open_ip_change_dialog()
            current_click = None

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame = draw_adb_message(frame)
        frame = draw_tinysa_message(frame)

        frame = draw_audio_detection_indicator(frame)
        frame = draw_fps_indicator(frame, current_fps)
        
        # Asegurar que la ventana siempre tenga el tamaño correcto
        if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
            cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
            current_window_size[:] = DEFAULT_WINDOW_SIZE
        
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            stop_program = True
            break
        elif key == ord('m') or key == ord('M'):
            if audio_enabled: stop_audio()
            else: start_audio()
        elif key == ord('a') or key == ord('A'):
            toggle_audio_detection()
        elif key == ord('y') or key == ord('Y'):
            toggle_yolo()
        elif key == ord('t') or key == ord('T'): 
            toggle_tinysa()
        elif key == ord('r') or key == ord('R'):
            rf_sliders_visible = not rf_sliders_visible
            print(f"[RF] Sliders {'activados' if rf_sliders_visible else 'desactivados'}")
        elif key == ord('i') or key == ord('I'):
            cap = cambiar_ip_camara(cap)
        
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            stop_program = True
            break
    except cv2.error:
        stop_program = True
        break

print("Cerrando aplicación...")
if yolo_enabled:
    stop_yolo_worker()
if audio_detection_enabled:
    toggle_audio_detection()
if audio_enabled:
    stop_audio()
if tinysa_running:
    toggle_tinysa() 
if cap is not None:
    cap.release()

p.terminate()
cv2.destroyAllWindows()
print("Programa finalizado.")
