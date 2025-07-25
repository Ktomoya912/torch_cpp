#!/bin/bash

# ベンチマーク実行スクリプト
# 使用方法: ./benchmark.sh [seed] [game_count] [model_file] [output_file]

SEED=${1:-42}
GAME_COUNT=${2:-1000}
MODEL_FILE=${3:-data/model_torchscript.pt}
OUTPUT_FILE=${4:-benchmark_results.txt}

echo "========================================="
echo "2048 AI Benchmark"
echo "========================================="
echo "Seed: $SEED"
echo "Game Count: $GAME_COUNT"
echo "Model File: $MODEL_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# 実行時間計測開始
START_TIME=$(date +%s.%N)

# プログラム実行（binディレクトリから）
./bin/play_greedy_ai $SEED $GAME_COUNT $MODEL_FILE > $OUTPUT_FILE

# 実行時間計測終了
END_TIME=$(date +%s.%N)

# 実行時間計算
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

echo "========================================="
echo "Benchmark Completed!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Execution Time: ${EXECUTION_TIME} seconds"
echo "========================================="

# 結果の統計情報を表示
if [ -f "$OUTPUT_FILE" ]; then
    echo "Results Summary:"
    echo "----------------"
    
    # 平均スコアを計算（最後の行から抽出）
    AVERAGE_SCORE=$(tail -1 "$OUTPUT_FILE" | grep "Average score:" | cut -d' ' -f3)
    if [ ! -z "$AVERAGE_SCORE" ]; then
        echo "Average Score: $AVERAGE_SCORE"
    fi
    
    # ゲーム数をカウント
    COMPLETED_GAMES=$(grep "Game .* finished:" "$OUTPUT_FILE" | wc -l)
    echo "Completed Games: $COMPLETED_GAMES"
    
    # 1ゲームあたりの平均時間
    if [ $COMPLETED_GAMES -gt 0 ]; then
        AVERAGE_TIME_PER_GAME=$(echo "scale=4; $EXECUTION_TIME / $COMPLETED_GAMES" | bc -l)
        echo "Average Time per Game: ${AVERAGE_TIME_PER_GAME} seconds"
    fi
    
    # 最高スコア
    MAX_SCORE=$(grep "Game .* finished:" "$OUTPUT_FILE" | sed 's/.*Score \([0-9]*\).*/\1/' | sort -n | tail -1)
    echo "Max Score: $MAX_SCORE"
    
    # 最低スコア
    MIN_SCORE=$(grep "Game .* finished:" "$OUTPUT_FILE" | sed 's/.*Score \([0-9]*\).*/\1/' | sort -n | head -1)
    echo "Min Score: $MIN_SCORE"
    
    echo "----------------"
    echo "Results saved to: $OUTPUT_FILE"
fi

echo "========================================="
