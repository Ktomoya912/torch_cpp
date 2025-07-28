#!/bin/bash

# LibTorch C++でのビルドスクリプト
set -e

echo "=== Building Torch C++ 2048 Player ==="

# LibTorchのダウンロードとセットアップ
LIBTORCH_DIR="$(pwd)/libtorch"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip"
LIBTORCH_ZIP="libtorch-cu126.zip"

if [ -z "$LIBTORCH_PATH" ]; then
    # 既にLibTorchがダウンロードされているかチェック
    if [ -d "$LIBTORCH_DIR" ] && [ -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]; then
        export LIBTORCH_PATH="$LIBTORCH_DIR"
        echo "Using existing LibTorch: $LIBTORCH_PATH"
    else
        echo "Downloading LibTorch CUDA 12.6 version..."
        # 既存のLibTorchディレクトリを削除
        rm -rf "$LIBTORCH_DIR" "$LIBTORCH_ZIP"
        
        # LibTorchをダウンロード
        if command -v wget >/dev/null 2>&1; then
            wget "$LIBTORCH_URL" -O "$LIBTORCH_ZIP"
        elif command -v curl >/dev/null 2>&1; then
            curl -L "$LIBTORCH_URL" -o "$LIBTORCH_ZIP"
        else
            echo "Error: wget or curl is required to download LibTorch"
            exit 1
        fi
        
        # 解凍
        echo "Extracting LibTorch..."
        if command -v unzip >/dev/null 2>&1; then
            unzip -q "$LIBTORCH_ZIP"
        else
            echo "Using Python to extract archive..."
            python -c "
import zipfile
import sys
with zipfile.ZipFile('$LIBTORCH_ZIP', 'r') as zip_ref:
    zip_ref.extractall('.')
print('Extraction completed')
"
        fi
        rm "$LIBTORCH_ZIP"
        
        export LIBTORCH_PATH="$LIBTORCH_DIR"
        echo "Downloaded and extracted LibTorch to: $LIBTORCH_PATH"
    fi
fi

# buildディレクトリを作成
mkdir -p build
cd build

# CMakeを実行
echo "Running CMake..."
echo "Using LibTorch with CUDA 12.6 support"

# CUDA環境の確認
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found: $(nvcc --version | grep -o 'release [0-9.]*')"
    CUDA_AVAILABLE=true
else
    echo "Warning: nvcc not found in PATH"
    CUDA_AVAILABLE=false
fi

# nvidia-smiでGPUの確認
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
else
    echo "No NVIDIA GPU detected"
    CUDA_AVAILABLE=false
fi

cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" ..

# ビルド
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "=== Build completed! ==="
echo "Executables built:"
echo "  - play_greedy_test (in bin/)"
echo "  - play_greedy_nn (in bin/)"  
echo "  - mcts_nn (in bin/)"
echo ""
echo "To run:"
echo "  cd bin && ./play_greedy_nn"

