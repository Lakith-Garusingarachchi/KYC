#!/usr/bin/env sh
# Downloads the MediaPipe face landmark model into public/mediapipe/.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/public/mediapipe"
mkdir -p "$DIR"
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
if [ -s "$DIR/face_landmarker.task" ]; then
  echo "face_landmarker.task already present"
else
  curl -fsSL "$URL" -o "$DIR/face_landmarker.task" && echo "downloaded face_landmarker.task"
fi
