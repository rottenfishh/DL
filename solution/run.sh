#!/usr/bin/env sh
# Запускать из любой директории — пути в скриптах абсолютные.
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/step1_select.py"
python "$DIR/step2_meta.py"
python "$DIR/step3_train.py"
