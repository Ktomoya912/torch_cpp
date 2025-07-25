#!/usr/bin/env python3
"""
PyTorchモデルをTorchScript形式に変換するスクリプト
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# CNN_DEEP.pyからモデルをインポート
from CNN_DEEP import Model


def convert_model_to_torchscript(model_path: str, output_path: str):
    """
    PyTorchモデルをTorchScript形式に変換
    
    Args:
        model_path: .pthファイルのパス
        output_path: 出力する.ptファイルのパス
    """
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルの作成と重みの読み込み
    model = Model()
    
    try:
        # 状態辞書を読み込み
        state_dict = torch.load(model_path, map_location=device)
        
        # 'model' キーがある場合（optimizer等も含まれている場合）
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # ダミー入力を作成（3x3ボード、11チャンネル）
    # 実際の入力形状に合わせて調整してください
    dummy_input = torch.randn(1, 11, 3, 3).to(device)
    
    try:
        # トレースモードでTorchScriptに変換
        traced_model = torch.jit.trace(model, dummy_input)
        
        # 保存
        traced_model.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        
        # 変換後のモデルをテスト
        with torch.no_grad():
            original_output = model(dummy_input)
            traced_output = traced_model(dummy_input)
            
            # 出力の差を確認
            diff = torch.abs(original_output - traced_output).max().item()
            print(f"Max difference between original and traced model: {diff}")
            
            if diff < 1e-5:
                print("✓ Conversion successful!")
                return True
            else:
                print("⚠ Warning: Outputs differ significantly")
                return False
                
    except Exception as e:
        print(f"Error during TorchScript conversion: {e}")
        return False


def main():
    # 入力と出力のパス
    model_path = "1_[learning-nn.py]_20250715T062509_[model-CNN_DEEP][seed-1][symmetry-True].pth"
    output_path = "model_torchscript.pt"
    
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found")
        return
    
    print("Converting PyTorch model to TorchScript...")
    success = convert_model_to_torchscript(model_path, output_path)
    
    if success:
        print("✓ Conversion completed successfully!")
        print(f"You can now use {output_path} in C++ with libtorch")
    else:
        print("✗ Conversion failed")


if __name__ == "__main__":
    main()