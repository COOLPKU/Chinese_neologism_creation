@echo off
REM 运行语料过滤器的批处理脚本

REM 设置参数
set WORD_LIST=word_list.json
set CORPUS=path\to\your\corpus.jsonl
set OUTPUT_DIR=output
set MAX_CONTEXT=2000
set NUM_THREADS=8
set CHUNK_SIZE=1000

REM 运行脚本
python corpus_filter.py ^
    --word_list %WORD_LIST% ^
    --corpus %CORPUS% ^
    --output_dir %OUTPUT_DIR% ^
    --max_context_length %MAX_CONTEXT% ^
    --num_threads %NUM_THREADS% ^
    --chunk_size %CHUNK_SIZE%

pause
