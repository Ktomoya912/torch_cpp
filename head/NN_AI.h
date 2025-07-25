#ifndef GAME2048AI_H
#define GAME2048AI_H

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_map>

// デバッグ用：テンソルの内容を表示する関数
inline void printTensor(const torch::Tensor& tensor) {
    std::cout << "Tensor shape: [";
    for (int i = 0; i < tensor.dim(); i++) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // CPUに移動してから表示
    auto cpu_tensor = tensor.to(torch::kCPU);
    std::cout << "Tensor data:" << std::endl;
    std::cout << cpu_tensor << std::endl;
}

class Game2048AI {
private:
    torch::jit::script::Module model;
    torch::Device device;
    
public:
    // コンストラクタ：TorchScriptモデルを読み込み
    Game2048AI(const std::string& model_path) 
        : device(torch::kCPU) {
        
        // CUDA が利用可能かチェック
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }
        
        try {
            // TorchScriptモデルを読み込み
            model = torch::jit::load(model_path);
            model.to(device);
            model.eval();
            std::cout << "Model loaded successfully from " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.msg() << std::endl;
            throw;
        }
    }
    
    // int配列（3x3=9要素）を3x3ボードに変換するヘルパー関数
    std::array<std::array<int, 3>, 3> intArrayToBoard(const int board[9]) {
        std::array<std::array<int, 3>, 3> result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i][j] = board[i * 3 + j];
            }
        }
        return result;
    }
    
    // int配列から直接テンソルに変換
    torch::Tensor intArrayToTensor(const int board[9]) {
        // auto board_array = intArrayToBoard(board);
        auto tensor = torch::zeros({1,99}, torch::kFloat32);
        int board_value = 0;
        for (int i = 0; i < 9; i++) {
            board_value = board[i];
            tensor[0][board_value * 9 + i] = 1;
        }
        return tensor.to(device);
    }
    
    // calcEv互換の関数：int配列から価値を計算
    double calcEv(const int board[9]) {
        try {
            auto input_tensor = intArrayToTensor(board);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            torch::NoGradGuard no_grad;
            at::Tensor output = model.forward(inputs).toTensor();
            
            output = output.to(torch::kCPU);
            return static_cast<double>(output[0].item<float>());
            
        } catch (const c10::Error& e) {
            std::cerr << "Error getting value: " << e.msg() << std::endl;
            return 0.0;
        }
    }
    
    // 最適な行動を予測（価値関数の場合は4方向試して最高値を選択）
    int predictBestAction(const int board[9]) {
        try {
            std::vector<float> action_values(4, -std::numeric_limits<float>::infinity());
            
            // 4方向それぞれの行動を試して価値を評価
            for (int action = 0; action < 4; action++) {
                // 実際にはここで行動後の状態を生成する必要があります
                // 簡単な例として現在のボードの価値を使用
                auto input_tensor = intArrayToTensor(board);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                torch::NoGradGuard no_grad;
                at::Tensor output = model.forward(inputs).toTensor();
                
                output = output.to(torch::kCPU);
                action_values[action] = output[0].item<float>();
            }
            
            // 最大値のインデックスを返す
            return std::max_element(action_values.begin(), action_values.end()) - action_values.begin();
            
        } catch (const c10::Error& e) {
            std::cerr << "Error during prediction: " << e.msg() << std::endl;
            return -1;
        }
    }
    
    // 単一の価値を取得
    float getValue(const int board[9]) {
        try {
            auto input_tensor = intArrayToTensor(board);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            torch::NoGradGuard no_grad;
            at::Tensor output = model.forward(inputs).toTensor();
            
            output = output.to(torch::kCPU);
            return output[0].item<float>();
            
        } catch (const c10::Error& e) {
            std::cerr << "Error getting value: " << e.msg() << std::endl;
            return 0.0f;
        }
    }
    
    // 全ての行動の価値を取得（価値関数の場合は同じ値が返される）
    std::vector<float> getActionValues(const int board[9]) {
        try {
            auto input_tensor = intArrayToTensor(board);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            torch::NoGradGuard no_grad;
            at::Tensor output = model.forward(inputs).toTensor();
            
            // CPUに移動
            output = output.to(torch::kCPU);
            float value = output[0].item<float>();
            
            // 価値関数なので同じ値を4つ返す
            return std::vector<float>(4, value);
            
        } catch (const c10::Error& e) {
            std::cerr << "Error getting action values: " << e.msg() << std::endl;
            return std::vector<float>(4, 0.0f);
        }
    }
    
    // ユーティリティ関数：ボードの表示
    void printBoard(const int board[9]) {
        std::cout << "Board:" << std::endl;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << board[i * 3 + j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 行動名を取得
    std::string getActionName(int action) {
        switch (action) {
            case 0: return "UP";
            case 1: return "DOWN";
            case 2: return "LEFT";
            case 3: return "RIGHT";
            default: return "UNKNOWN";
        }
    }
    
    // デバッグ情報付きで最適行動を予測
    int predictBestActionWithDebug(const int board[9], bool verbose = true) {
        if (verbose) {
            printBoard(board);
        }
        
        int best_action = predictBestAction(board);
        float board_value = getValue(board);
        
        if (verbose) {
            std::cout << "Best action: " << best_action << " (" << getActionName(best_action) << ")" << std::endl;
            std::cout << "Board value: " << board_value << std::endl;
        }
        
        return best_action;
    }
};

// グローバル変数として使用するためのAIインスタンス
extern Game2048AI* g_ai;

// calcEv関数の代替となるグローバル関数（メモ化付き）
inline double calcEvAI(const int board[9]) {
    static std::unordered_map<size_t, double> cache;
    // 盤面ハッシュ計算（9要素int配列→size_t）
    size_t hashValue = 100;
    const int base = 12;
    for (int i = 0; i < 9; i++) {
        hashValue = hashValue * base + board[i];
    }
    auto it = cache.find(hashValue);
    if (it != cache.end()) {
        return it->second;
    }
    double value = 0.0;
    if (g_ai) {
        value = g_ai->calcEv(board);
    }
    cache[hashValue] = value;
    return value;
}

// AIを初期化する関数
inline void initAI(const std::string& model_path) {
    g_ai = new Game2048AI(model_path);
}

// AIを解放する関数
inline void cleanupAI() {
    if (g_ai) {
        delete g_ai;
        g_ai = nullptr;
    }
}

#endif // GAME2048AI_H
