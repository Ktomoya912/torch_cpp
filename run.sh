

#!/bin/bash
# 使用方法: ./run.sh [モデル名プレフィックス]
# 例: ./run.sh TDA
# 例: ./run.sh D-TDA-c

# デフォルト値
MODEL_PREFIX=${1:-"TDA"}
MODEL_DIR="$(pwd)/models"

# 評価するモデル番号の配列（ここを編集して使用したい番号を指定）
# 例: MODEL_NUMBERS=(1 2 3)        # 1, 2, 3番を実行
# 例: MODEL_NUMBERS=(2 3)          # 2, 3番のみ実行  
# 例: MODEL_NUMBERS=(1)            # 1番のみ実行
# 例: MODEL_NUMBERS=(1 3 5)        # 1, 3, 5番を実行
MODEL_NUMBERS=(1 3 2) # ここを編集して使用したい番号を指定

echo "=== Starting model evaluation ==="
echo "Model prefix: $MODEL_PREFIX"
echo "Model numbers: ${MODEL_NUMBERS[*]}"
echo "Model directory: $MODEL_DIR"
cd ./build/bin
# プロセスIDを記録するための配列
declare -a PIDS

# 配列の各要素に対してループ
for i in "${MODEL_NUMBERS[@]}"; do
    MODEL_FILE="$MODEL_DIR/$MODEL_PREFIX-$i.pt"
    
    # ファイルの存在確認
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Warning: Model file not found: $MODEL_FILE"
        echo "Skipping $MODEL_PREFIX-$i.pt"
        continue
    fi
    
    echo "Starting evaluation for $MODEL_PREFIX-$i.pt"
    
    # MCTS NN実行
    echo "  - Starting MCTS NN for $MODEL_PREFIX-$i.pt"
    nohup ./mcts_nn 1 1000 "$MODEL_FILE" 400 0 1 1000 0 1 0 > "mcts_nn_${MODEL_PREFIX}_$i.log" 2>&1 &
    MCTS_PID=$!
    PIDS+=($MCTS_PID)
    echo "    MCTS NN PID: $MCTS_PID"
    
    # EXP NN実行
    echo "  - Starting EXP NN for $MODEL_PREFIX-$i.pt"
    nohup ./exp_nn 1 1000 3 "$MODEL_FILE" > "exp_nn_${MODEL_PREFIX}_$i.log" 2>&1 &
    EXP_PID=$!
    PIDS+=($EXP_PID)
    echo "    EXP NN PID: $EXP_PID"
    
    echo "Waiting for MCTS and EXP processes to finish..."
    # MCTS_PIDとEXP_PIDが終了するまで待つ
    wait $MCTS_PID
    wait $EXP_PID
    echo "Finished evaluation for $MODEL_PREFIX-$i.pt"
done

echo ""
echo "=== All processes started ==="
echo "Running processes:"
for pid in "${PIDS[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        echo "  PID $pid: Running"
    else
        echo "  PID $pid: Not running"
    fi
done

echo ""
echo "To monitor progress:"
echo "  tail -f mcts_nn_${MODEL_PREFIX}_*.log"
echo "  tail -f exp_nn_${MODEL_PREFIX}_*.log"
echo ""
echo "To stop all processes:"
echo "  kill ${PIDS[*]}"