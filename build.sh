#!/bin/bash
# Build the C++ engine module on Jetson ORIN
# Usage: bash build.sh

set -e

# Install pybind11 if not present
pip install pybind11 2>/dev/null || true

cd "$(dirname "$0")/cpp"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Copy .so to project root for easy import
cp yolo_engine_cpp*.so ../../
echo ""
echo "Build done. You can now use: from yolo_engine_cpp import YoloEngine"
