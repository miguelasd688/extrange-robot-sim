#!/bin/bash


NUM_ENVS=1000
MAX_ITER=1001

# Ruta base de logs
LOG_DIR="/workspace/extrange_robot/logs/rsl_rl/extrange_robot_ppo"

# Encuentra el último experimento
LATEST_RUN=$(ls -td ${LOG_DIR}/*/ | head -n 1 | xargs -n 1 basename)

LATEST_CHECKPOINT=$(find "${LOG_DIR}/${LATEST_RUN}" -maxdepth 1 -type f -name "model_*.pt" | sort -V | tail -n 1)
echo "$LATEST_CHECKPOINT"
# Comandos
case "$1" in
  train)
    if [ "$2" == "new" ]; then
      echo "▶️ Iniciando entrenamiento desde cero..."
      python scripts/rsl_rl/train.py \
        --task Template-Extrange-Robot-Direct-v0 \
        --num_envs $NUM_ENVS \
        --headless \
        --run_name extrange_run \
        --max_iterations $MAX_ITER

    elif [ "$2" == "resume" ]; then
      echo "▶️ Reanudando entrenamiento desde $LATEST_RUN"
      python scripts/rsl_rl/train.py \
        --task Template-Extrange-Robot-Direct-v0 \
        --num_envs $NUM_ENVS \
        --headless \
        --run_name extrange_run \
        --max_iterations $MAX_ITER \
        --load_run $LATEST_RUN \
        --resume
    else
      echo "❌ Uso inválido. Usa: ./launch.sh train new | resume"
    fi
    ;;

  play)
    if [ "$2" == "last" ]; then
      if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "❌ No se encontró checkpoint en directorio $LATEST_RUN"
        exit 1
      fi
      echo "▶️ Reproduciendo último checkpoint: $LATEST_CHECKPOINT"
      python scripts/rsl_rl/play.py \
        --task Template-Extrange-Robot-Direct-v0 \
        --rendering_mode=performance \
        --checkpoint $LATEST_CHECKPOINT \
        --num_envs 4
    else
      if [ -z "$2" ]; then
        echo "❌ No se encontró checkpoint en $2"
        exit 1
      fi
      echo "▶️ Reproduciendo último checkpoint: $2"
      python scripts/rsl_rl/play.py \
        --task Template-Extrange-Robot-Direct-v0 \
        --rendering_mode=performance \
        --checkpoint $2 \
        --num_envs 4
    fi
    ;;

  *)
    echo "❌ Comando no reconocido."
    echo "Opciones válidas:"
    echo "  ./launch.sh train new"
    echo "  ./launch.sh train resume"
    echo "  ./launch.sh play last"
    ;;
esac