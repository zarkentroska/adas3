#!/bin/bash -i
# Este es tu script de shell
echo "🚀 Ejecutando el primer comando..."
conda init --all
conda activate drone_v2
ls -l
echo "⏳ Esperando 2 segundos..."
sleep 2
echo "🚀 Ejecutando el segundo comando..."
python3 /home/zarkentroska/Documentos/testcam.py
pwd
echo "✅ Proceso finalizado. La terminal se cerrará en 10 segundos..."
sleep 10
