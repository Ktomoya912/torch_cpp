#!/bin/bash

# LibTorch C++でのビルドスクリプト
set -e

echo "=== Building Torch C++ 2048 Player ==="

# libtorchのパスを設定（環境に応じて調整してください）
# Condaでインストールした場合の例
if [ -z "$LIBTORCH_PATH" ]; then
    if [ -d "$CONDA_PREFIX/lib/python*/site-packages/torch" ]; then
        export LIBTORCH_PATH="$CONDA_PREFIX/lib/python*/site-packages/torch"
        echo "Using conda libtorch: $LIBTORCH_PATH"
    elif [ -d "/usr/local/lib/python*/site-packages/torch" ]; then
        export LIBTORCH_PATH="/usr/local/lib/python*/site-packages/torch"
        echo "Using system libtorch: $LIBTORCH_PATH"
    else
        echo "Error: LibTorch not found. Please set LIBTORCH_PATH environment variable."
        echo "Example: export LIBTORCH_PATH=/path/to/libtorch"
        exit 1
    fi
fi

# buildディレクトリを作成
mkdir -p build
cd build

# CMakeを実行
echo "Running CMake..."
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" ..

# ビルド
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "=== Build completed! ==="
echo "Executable: ./play_torchscript"
echo ""
echo "Before running, make sure to:"
echo "1. Convert your model to TorchScript: python convert_to_torchscript.py"
echo "2. Run the executable: ./play_torchscript"
