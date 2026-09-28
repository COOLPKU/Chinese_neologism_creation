# 语料库过滤器 (Corpus Filter)

从大规模JSONL语料库中提取包含目标词的上下文信息。

## 功能特点

- ✅ 支持处理超大文件(60G+)
- ✅ 多线程并行处理,提高效率
- ✅ 自动提取目标词上下文
- ✅ 标注目标词在上下文中的位置
- ✅ 智能截取过长上下文
- ✅ 分类保存三种类型的词(existing/positive/negative)

## 安装依赖

```bash
# 无需额外依赖,使用Python标准库即可
python --version  # 建议Python 3.7+
```

## 使用方法

### 基本用法

```bash
python corpus_filter.py \
    --word_list word_list.json \
    --corpus /path/to/corpus.jsonl \
    --output_dir output
```

### 完整参数

```bash
python corpus_filter.py \
    --word_list word_list.json \          # 目标词列表JSON文件
    --corpus /path/to/corpus.jsonl \      # 输入语料库文件
    --output_dir output \                 # 输出目录
    --max_context_length 500 \            # 上下文最大长度(字符)
    --num_threads 8 \                     # 线程数
    --chunk_size 1000                     # 每批处理行数
```

### Windows 批处理

修改 `run_filter.bat` 中的路径参数后,直接双击运行:

```bat
run_filter.bat
```

## 输入格式

### word_list.json

```json
{
    "existing_words": ["词1", "词2", ...],
    "positive_words": ["词3", "词4", ...],
    "negative_words": ["词5", "词6", ...]
}
```

### corpus.jsonl

每行一个JSON对象:

```json
{"text": "这是一段包含目标词的文本内容..."}
{"text": "另一段文本内容..."}
```

## 输出格式

生成三个JSONL文件,每个文件对应一类词:

- `existing_words_contexts.jsonl`
- `positive_words_contexts.jsonl`
- `negative_words_contexts.jsonl`

每行格式:

```json
{
    "word": "目标词",
    "context": "...目标词的上下文...",
    "word_start": 10,              // 目标词在context中的起始位置
    "word_end": 13                 // 目标词在context中的结束位置
}
```

## 性能优化建议

### 线程数设置

- **CPU密集型**: 设置为CPU核心数
- **IO密集型**: 可设置为CPU核心数的2-4倍
- 建议值: 8-16 (根据机器配置调整)

```bash
--num_threads 16
```

### 批处理大小

- 较小值(100-500): 内存占用少,但线程切换频繁
- 较大值(1000-5000): 减少线程切换,但内存占用增加
- 建议值: 1000

```bash
--chunk_size 1000
```

### 上下文长度

- 根据实际需求调整
- 过长会增加存储空间和处理时间
- 建议值: 1000-2000字符(用于保留完整语义)

```bash
--max_context_length 2000
```

## 处理速度参考

在标准配置下(8线程,1000行/批):

- 处理速度: 约 1,000-5,000 行/秒
- 60GB文件: 预计需要 2-10 小时(取决于硬件和匹配率)

## 常见问题

### Q: 内存占用过高?

A: 减小 `--chunk_size` 参数值

### Q: 处理速度太慢?

A: 
1. 增加 `--num_threads` 参数
2. 使用SSD存储
3. 增大 `--chunk_size` 参数

### Q: 如何中断和恢复?

A: 当前版本不支持断点续传。建议:
1. 先对文件进行分片处理
2. 分别处理后合并结果

## 示例

```bash
# 示例1: 快速测试(小文件)
python corpus_filter.py \
    --word_list word_list.json \
    --corpus test_corpus.jsonl \
    --output_dir test_output \
    --num_threads 4

# 示例2: 生产环境(大文件)
python corpus_filter.py \
    --word_list word_list.json \
    --corpus large_corpus.jsonl \
    --output_dir production_output \
    --max_context_length 2000 \
    --num_threads 16 \
    --chunk_size 2000
```

## 注意事项

1. 确保磁盘空间充足(输出文件可能较大)
2. 对于超大文件,建议在服务器上运行
3. 定期查看输出目录,确认结果正确生成
4. 处理过程中会定期输出进度信息

## 许可证

内部使用
