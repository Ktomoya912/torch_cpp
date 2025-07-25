# PyTorch → TorchScript → C++ 変換ガイド

このプロジェクトでは、Pythonで学習したPyTorchモデルをC++で実行する方法を示します。

## 必要な環境

### Python環境
- PyTorch
- NumPy

### C++環境
- CMake (3.18以上)
- LibTorch (PyTorchのC++ライブラリ)
- C++20対応コンパイラ

## 手順

### 1. モデル変換の準備

まず、実際のモデル構造に合わせて `convert_to_torchscript.py` を修正してください：

```python
# あなたの実際のモデル定義をここに配置
class YourActualModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 実際のレイヤー構成
    
    def forward(self, x):
        # 実際のforward処理
        return x
```

### 2. TorchScript変換の実行

```bash
python convert_to_torchscript.py
```

これにより `model_torchscript.pt` ファイルが生成されます。

### 3. C++ビルド（CMakeの使い方）

本リポジトリはCMakeでビルドします。最適化フラグ（-O3）が有効になっています。

#### 推奨手順

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

#### 個別ターゲットのビルド

`play_greedy_nn`や`mcts_nn`のみをビルドしたい場合は、以下のように個別にmakeできます：

```bash
make play_greedy_nn
make mcts_nn
```

それぞれ `bin/play_greedy_nn` および `bin/mcts_nn` が生成されます。

#### 実行例

```bash
./bin/play_greedy_nn
./bin/mcts_nn
```

`/path/to/libtorch` はダウンロードしたLibTorchのパスに置き換えてください。

#### 既存のbuild.shを使う場合

```bash
./build.sh
```

#### 最適化フラグについて

`CMakeLists.txt` では `-O3` フラグが全ターゲットに設定されています。より高速な実行を目指す場合に有効です。

### 3. C++側の調整

`play_torchscript.cpp` の以下の部分を実際の入力形式に合わせて修正：

- `boardToTensor()` 関数の入力エンコーディング
- ボードサイズ（現在は3x3用）
- チャンネル数（現在は16チャンネル）

### 4. LibTorchのインストール

#### 方法1: Condaを使用
```bash
conda install pytorch cpuonly -c pytorch
```

#### 方法2: pip経由でのインストール
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 方法3: 公式サイトからダウンロード
https://pytorch.org/cppdocs/installing.html

### 5. ビルド

```bash
./build.sh
```

または手動で：

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

### 6. 実行

```bash
cd build
./play_torchscript
```

## トラブルシューティング

### モデル読み込みエラー
- モデル構造が一致しているか確認
- 入力テンソルの形状が正しいか確認
- TorchScript変換時のエラーメッセージを確認

### ビルドエラー
- LibTorchのパスが正しく設定されているか確認
- CMakeバージョンが3.18以上か確認
- C++コンパイラがC++20に対応しているか確認

### 実行時エラー
- model_torchscript.ptファイルが存在するか確認
- CUDAを使用する場合、適切なCUDAバージョンがインストールされているか確認

## カスタマイズのポイント

1. **モデル構造の調整**: 実際の学習済みモデルの構造に合わせる
2. **入力エンコーディング**: ボードの表現方法を統一する
3. **出力の解釈**: 行動価値の意味を正しく解釈する
4. **パフォーマンス最適化**: バッチ処理や量子化の検討

## 注意事項

- TorchScript変換時は、動的な制御フロー（if文、ループ）に制限があります
- C++とPythonで浮動小数点演算の精度に微小な差が出る場合があります
- メモリ使用量とパフォーマンスを定期的に監視してください
