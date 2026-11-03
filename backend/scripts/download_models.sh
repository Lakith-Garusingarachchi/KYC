#!/usr/bin/env sh
# Downloads the OpenCV Zoo face models (YuNet detector + SFace recognizer) into backend/models.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
BASE="https://github.com/opencv/opencv_zoo/raw/main/models"
mkdir -p "$DIR"

fetch() {
  folder="$1"; file="$2"; sha="$3"
  if [ -s "$DIR/$file" ]; then
    echo "$file already present"
  else
    echo "downloading $file"
    curl -fsSL "$BASE/$folder/$file" -o "$DIR/$file"
  fi
  echo "$sha  $DIR/$file" | sha256sum -c -
}

fetch face_detection_yunet face_detection_yunet_2023mar.onnx 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
fetch face_recognition_sface face_recognition_sface_2021dec.onnx 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
