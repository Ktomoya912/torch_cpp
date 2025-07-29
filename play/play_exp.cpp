// AI版のplay_greedy_test: PyTorchモデルを使用してゲームプレイ
#include <array>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <list>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using namespace std;

// #include "../head/Game2048_3_3.h"
// #include "../head/NN_AI.h"
// #include "../head/expmax.h"
#include "Game2048_3_3.h"
#include "NN_AI.h"
#include "expmax.h"

class GameOver {
 public:
  int gameover_turn;
  int game;
  int progress;
  int score;
  GameOver(int gameover_turn_init, int game_init, int progress_init, int score_init)
      : gameover_turn(gameover_turn_init),
        game(game_init),
        progress(progress_init),
        score(score_init) {}
};

int progress_calculation(int board[9]) {
  int sum = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] != 0) {
      sum += 1 << board[i];
    }
  }
  return sum / 2;
}

string getCurrentTimeString() {
  auto now = chrono::system_clock::now();
  auto time_t = chrono::system_clock::to_time_t(now);
  auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
  
  stringstream ss;
  ss << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  ss << "." << setfill('0') << setw(3) << ms.count();
  return ss.str();
}

string parseFileName(const char *filename)
{

  char basename[256];
  strcpy(basename, filename);

  // ファイルパスから最後の'/'以降を取得
  char *last_slash = strrchr(basename, '/');
  char *actual_name = last_slash ? last_slash + 1 : basename;

  // .zip 拡張子を削除
  char *ext = strstr(actual_name, ".pt");
  if (ext)
  {
    *ext = '\0'; // 拡張子を削除
  }
  return actual_name;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: play_greedy_test <seed> <game_counts> <number_of_depth> <model_file>\n");
    exit(1);
  }
  
  int seed = atoi(argv[1]);
  int game_count = atoi(argv[2]);
  int number_of_depth = atoi(argv[3]);
  char* model_file = argv[4];
  
  // AIを初期化
  try {
    initAI(model_file);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error initializing AI: %s\n", e.what());
    exit(1);
  }
  
  string parsed = parseFileName(model_file);
  // 出力ディレクトリの作成
  fs::create_directory("../board_data");
  string dir = "../board_data/exp-" + parsed + "_" + to_string(seed) + "/";
  fs::create_directory(dir);
  
  srand(seed);
  
  list<array<int, 9>> state_list;
  list<array<int, 9>> after_state_list;
  const int eval_length = 5;
  list<array<double, eval_length>> eval_list;
  list<GameOver> GameOver_list;
  double score_sum = 0;
  
  for (int gid = 1; gid <= game_count; gid++) {
    state_t state = initGame();
    int turn = 0;
    
    while (true) {
      turn++;
      state_t copy;
      double max_evr = -DBL_MAX;
      // int selected = -1;
      const int n = 5;
      double evals[4];
      int selected = expectimax(state, number_of_depth,evals);
      
      // 評価値を初期化
    //   for (int i = 0; i < n; i++) {
    //     evals[i] = -1.0e10;
    //   }
      
    //   // 4方向の行動を評価
    //   for (int d = 0; d < 4; d++) {
    //     if (play(d, state, &copy)) {
    //       evals[d] = calcEvAI(copy.board);  // AIの評価関数を使用
          
    //       if (max_evr < evals[d]) {
    //         max_evr = evals[d];
    //         selected = d;
    //       }
    //     }
    //   }
      
      // 状態を記録
      state_list.push_back(
          array<int, 9>{state.board[0], state.board[1], state.board[2],
                        state.board[3], state.board[4], state.board[5],
                        state.board[6], state.board[7], state.board[8]});
      if (gid == 1) {
        printf("[%s] Game %d, Turn %d, Move: %d, Eval: %.2f, %.2f, %.2f, %.2f\n",
          getCurrentTimeString().c_str(), gid, turn, selected, evals[0], evals[1], evals[2], evals[3]);
      }
      
      // 選択した行動を実行
      play(selected, state, &state);
      
      // 行動後の状態を記録
      after_state_list.push_back(
          array<int, 9>{state.board[0], state.board[1], state.board[2],
                        state.board[3], state.board[4], state.board[5],
                        state.board[6], state.board[7], state.board[8]});
      
      // 評価値を記録
      eval_list.push_back(array<double, eval_length>{
          evals[0], evals[1], evals[2], evals[3],
          (double)progress_calculation(state.board)});
      
      // 新しいタイルを配置
      putNewTile(&state);

      // ゲーム終了判定
      if (gameOver(state)) {
        GameOver_list.push_back(GameOver(
            turn, gid, progress_calculation(state.board), state.score));
        score_sum += state.score;
        printf("Game %d finished: Score %d, Turns %d\n", gid, state.score, turn);
        break;
      }
    }
  }
  
  printf("Average score: %.2f\n", score_sum / game_count);
  
  // ファイル出力
  string file;
  string fullPath;
  const char* filename;
  FILE* fp;
  int i;
  auto trun_itr = GameOver_list.begin();
  
  // state.txt
  file = "state.txt";
  fullPath = dir + file;
  filename = fullPath.c_str();
  fp = fopen(filename, "w+");
  i = 0;
  trun_itr = GameOver_list.begin();
  for (auto itr = state_list.begin(); itr != state_list.end(); itr++) {
    i++;
    if ((trun_itr)->gameover_turn == i) {
      i = 0;
      fprintf(fp, "gameover_turn: %d; game: %d; progress: %d; score: %d\n",
              (trun_itr)->gameover_turn, (trun_itr)->game, (trun_itr)->progress,
              (trun_itr)->score);
      trun_itr++;
    } else {
      for (int j = 0; j < 9; j++) {
        fprintf(fp, "%d ", (*itr)[j]);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
  
  // after-state.txt
  file = "after-state.txt";
  fullPath = dir + file;
  filename = fullPath.c_str();
  fp = fopen(filename, "w+");
  i = 0;
  trun_itr = GameOver_list.begin();
  for (auto itr = after_state_list.begin(); itr != after_state_list.end(); itr++) {
    i++;
    if ((trun_itr)->gameover_turn == i) {
      i = 0;
      fprintf(fp, "gameover_turn: %d; game: %d; progress: %d; score: %d\n",
              (trun_itr)->gameover_turn, (trun_itr)->game, (trun_itr)->progress,
              (trun_itr)->score);
      trun_itr++;
    } else {
      for (int j = 0; j < 9; j++) {
        fprintf(fp, "%d ", (*itr)[j]);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
  
  // eval.txt
  file = "eval.txt";
  fullPath = dir + file;
  filename = fullPath.c_str();
  fp = fopen(filename, "w+");
  i = 0;
  trun_itr = GameOver_list.begin();
  for (auto itr = eval_list.begin(); itr != eval_list.end(); itr++) {
    i++;
    if ((trun_itr)->gameover_turn == i) {
      i = 0;
      fprintf(fp, "gameover_turn: %d; game: %d; progress: %d; score: %d\n",
              (trun_itr)->gameover_turn, (trun_itr)->game, (trun_itr)->progress,
              (trun_itr)->score);
      trun_itr++;
    } else {
      for (int j = 0; j < eval_length; j++) {
        if (j + 1 >= eval_length) {
          fprintf(fp, "%d ", (int)(*itr)[j]);
        } else {
          fprintf(fp, "%f ", (*itr)[j]);
        }
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
  
  // AIのクリーンアップ
  cleanupAI();
  
  return 0;
}
