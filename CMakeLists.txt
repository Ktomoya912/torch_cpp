cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_cpp_2048)

# C++20以上を要求（PyTorch 2.7.1では必須、play_greedy_test.cppでも必要）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PyTorchのC++ライブラリ（libtorch）を見つける
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# zlibライブラリのリンク（macOSの場合）
find_package(ZLIB REQUIRED)

# ヘッダファイルのディレクトリを追加
include_directories(head)

# Greedy player with AI support (zlib使用)
add_executable(play_greedy_test play/play_greedy_test.cpp head/NN_AI.cpp)
target_link_libraries(play_greedy_test ${TORCH_LIBRARIES} ${ZLIB_LIBRARIES})
target_compile_options(play_greedy_test PRIVATE -mcmodel=large -O3)
set_property(TARGET play_greedy_test PROPERTY CXX_STANDARD 20)

# NN AI版 Greedy
add_executable(play_greedy_nn play/play_greedy_nn.cpp head/NN_AI.cpp)
target_link_libraries(play_greedy_nn ${TORCH_LIBRARIES})
target_compile_options(play_greedy_nn PRIVATE -mcmodel=large -O3)
set_property(TARGET play_greedy_nn PROPERTY CXX_STANDARD 20)
set_target_properties(play_greedy_nn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# NN AI版 MCTS
add_executable(mcts_nn play/mcts_NN.cpp head/NN_AI.cpp)
target_link_libraries(mcts_nn ${TORCH_LIBRARIES} ${ZLIB_LIBRARIES})
target_compile_options(mcts_nn PRIVATE -mcmodel=large -O3)
set_property(TARGET mcts_nn PROPERTY CXX_STANDARD 20)
set_target_properties(mcts_nn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 実行ファイルの出力先を設定
## (不要なターゲット設定を削除)
set_target_properties(play_greedy_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(play_greedy_nn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(mcts_nn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# コンパイラとプラットフォーム固有の設定
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET play_greedy_ai
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:play_greedy_ai>)
endif (MSVC)

# libtorchのバージョン情報を表示
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch library: ${TORCH_LIBRARIES}")
message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Headers directory: head/")
message(STATUS "Source directory: play/")
message(STATUS "Data directory: data/")
message(STATUS "Python scripts: py/")
