#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <thread>

#include "Game2048_3_3.h"
using namespace std;
namespace fs = std::filesystem;
#include "NN_AI.h"
// string NT = "NT6";
#include "mcts_NN.hpp"

class GameOver
{
public:
  int gameover_turn;
  int game;
  int progress;
  int score;
  GameOver(int gameover_turn_init, int game_init, int progress_init,
           int score_init)
      : gameover_turn(gameover_turn_init),
        game(game_init),
        progress(progress_init),
        score(score_init) {}
};

int progress_calculation(int board[9])
{
  int sum = 0;
  for (int i = 0; i < 9; i++)
  {
    if (board[i] != 0)
    {
      sum += 1 << board[i];
    }
  }
  return sum / 2;
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

int main(int argc, char **argv)
{
  if (argc < 10 + 1)
  {
    fprintf(stderr,
            "Usage: mcts_nn <seed> <game_counts> <model_file> <simulations> <randomTurn> <expand_count> <c> <Boltzmann> <expectimax> <debug>\n");
    exit(1);
  }

  int seed = atoi(argv[1]);
  int game_count = atoi(argv[2]);
  char *model_file = argv[3];
  int simulations = atoi(argv[4]);
  int randomTurn = atoi(argv[5]);
  int expand_count = atoi(argv[6]);
  int c = atoi(argv[7]);
  bool Boltzmann = atoi(argv[8]);
  bool expectimax = atoi(argv[9]);
  bool debug = atoi(argv[10]);

  // NN_AIモデル初期化
  try
  {
    initAI(model_file);
  }
  catch (const std::exception &e)
  {
    fprintf(stderr, "Error initializing AI: %s\n", e.what());
    exit(1);
  }
  string parsed = parseFileName(model_file);
  fs::create_directory("../board_data");

  string dir = "../board_data/mcts-" + parsed + "_" + to_string(seed) + "/";
  fs::create_directory(dir);

  srand(seed);

  mcts_searcher_t mcts_searcher(simulations, randomTurn, expand_count, c, Boltzmann, expectimax, debug);

  list<array<int, 9>> state_list;
  list<array<int, 9>> after_state_list;
  list<GameOver> GameOver_list;
  list<array<double, 5>> eval_list;

  for (int gid = 1; gid <= game_count; gid++)
  {
    state_t state = initGame();
    mcts_searcher.clearCache();

    for (int turn = 1;; turn++)
    {
      // printf("Game %d, Turn %d\n", gid, turn);
      array<double, 5> evals = {-1e10, -1e10, -1e10, -1e10, 0.0};
      int move = mcts_searcher.search(state, evals); // search内でcalcEvAIを使う

      state_t nextstate;
      bool result = play(move, state, &nextstate);
      assert(result);

      // 状態記録
      state_list.push_back(
          array<int, 9>{state.board[0], state.board[1], state.board[2],
                        state.board[3], state.board[4], state.board[5],
                        state.board[6], state.board[7], state.board[8]});

      after_state_list.push_back(array<int, 9>{
          nextstate.board[0], nextstate.board[1], nextstate.board[2],
          nextstate.board[3], nextstate.board[4], nextstate.board[5],
          nextstate.board[6], nextstate.board[7], nextstate.board[8]});

      // 評価値記録
      evals[4] = (double)progress_calculation(nextstate.board);
      eval_list.push_back(evals);

      putNewTile(&nextstate);

      if (gameOver(nextstate))
      {
        GameOver_list.push_back(GameOver(turn, gid,
                                         progress_calculation(nextstate.board),
                                         nextstate.score));
        // 終了情報を標準出力
        printf("Game %d finished: Score %d, Turns %d\n", gid, nextstate.score, turn);
        break;
      }
      state = nextstate;
    }
  }

  // 出力処理
  ofstream stateFile(dir + "state.txt");
  ofstream afterStateFile(dir + "after-state.txt");
  ofstream evalFile(dir + "eval.txt");

  auto trun_itr = GameOver_list.begin();
  int i = 0;
  for (auto itr = state_list.begin(); itr != state_list.end(); itr++)
  {
    i++;
    if (trun_itr != GameOver_list.end() && trun_itr->gameover_turn == i)
    {
      stateFile << "gameover_turn: " << trun_itr->gameover_turn
                << "; game: " << trun_itr->game
                << "; progress: " << trun_itr->progress
                << "; score: " << trun_itr->score << "\n";
      afterStateFile << "gameover_turn: " << trun_itr->gameover_turn
                     << "; game: " << trun_itr->game
                     << "; progress: " << trun_itr->progress
                     << "; score: " << trun_itr->score << "\n";
      evalFile << "gameover_turn: " << trun_itr->gameover_turn
               << "; game: " << trun_itr->game
               << "; progress: " << trun_itr->progress
               << "; score: " << trun_itr->score << "\n";
      trun_itr++;
      i = 0;
    }
    else
    {
      for (int j = 0; j < 9; j++)
      {
        stateFile << (*itr)[j] << " ";
      }
      stateFile << "\n";

      auto afterItr = after_state_list.begin();
      advance(afterItr, distance(state_list.begin(), itr));
      for (int j = 0; j < 9; j++)
      {
        afterStateFile << (*afterItr)[j] << " ";
      }
      afterStateFile << "\n";

      auto evalItr = eval_list.begin();
      advance(evalItr, distance(state_list.begin(), itr));
      for (int j = 0; j < 5; j++)
      {
        evalFile << std::fixed << (*evalItr)[j] << (j == 4 ? "" : " ");
      }
      evalFile << "\n";
    }
  }

  stateFile.close();
  afterStateFile.close();
  evalFile.close();

  cleanupAI();
  return 0;
}
