#!/bin/bash

# Script para actualizar el repositorio GitHub fácilmente
# Uso: ./actualizar_github.sh "Descripción de los cambios"

if [ -z "$1" ]; then
    echo "Error: Debes proporcionar un mensaje de commit"
    echo "Uso: ./actualizar_github.sh \"Descripción de los cambios\""
    exit 1
fi

echo "Agregando cambios..."
git add .

echo "Creando commit con mensaje: $1"
git commit -m "$1"

echo "Subiendo cambios a GitHub..."
git push

echo "¡Actualización completada!"

