#!/bin/bash

# MCTS NNベンチマーク実行スクリプト
# 使用方法: ./benchmark_mcts_nn.sh [seed] [game_count] [model_file] [output_file]

SEED=${1:-0}
GAME_COUNT=${2:-1000}
MODEL_FILE=${3:-./data/model_torchscript.pt}
OUTPUT_FILE=${4:-benchmark_mcts_nn_results.txt}
SIMULATIONS=400
RANDOM_TURN=0
EXPAND_COUNT=1
C=1000
BOLTZMANN=0
EXPECTIMAX=0
DEBUG=1

echo "========================================="
echo "MCTS NN Benchmark"
echo "========================================="
echo "Seed: $SEED"
echo "Game Count: $GAME_COUNT"
echo "Model File: $MODEL_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

START_TIME=$(date +%s.%N)

CMD="./bin/mcts_nn $SEED $GAME_COUNT $MODEL_FILE $SIMULATIONS $RANDOM_TURN $EXPAND_COUNT $C $BOLTZMANN $EXPECTIMAX $DEBUG"
echo "Running: $CMD"
$CMD > "$OUTPUT_FILE"

END_TIME=$(date +%s.%N)
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

echo "========================================="
echo "Benchmark Completed!"
echo "End Time: $(date)"
echo "Total Execution Time: ${EXECUTION_TIME} seconds"
echo "========================================="

if [ -f "$OUTPUT_FILE" ]; then
    echo "Results Summary:"
    echo "----------------"
    # 平均スコア
    AVERAGE_SCORE=$(tail -1 "$OUTPUT_FILE" | grep "Average score:" | cut -d' ' -f3)
    if [ ! -z "$AVERAGE_SCORE" ]; then
        echo "Average Score: $AVERAGE_SCORE"
    fi
    # ゲーム数
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
