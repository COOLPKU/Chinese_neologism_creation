import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import time
import os
import multiprocessing
from functools import partial
import pkuseg
import random
from tqdm import tqdm

class TrieNode:
    """Trie树节点"""
    __slots__ = ('children', 'is_end', 'word', 'word_type')
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.word_type = None

class Trie:
    """简单的Trie树实现，用于多模式匹配"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, word_type: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word
        node.word_type = word_type

    def search_in_text(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        在文本中搜索所有词
        Returns: List of (word, word_type, start_pos, end_pos)
        """
        results = []
        text_len = len(text)
        
        # 简单的遍历匹配，虽然不如AC自动机快，但比对每个词find要快得多
        # 尤其是当词表很大时
        for i in range(text_len):
            # 快速跳过不在Trie树第一层的字符
            if text[i] not in self.root.children:
                continue
                
            node = self.root
            j = i
            while j < text_len and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    results.append((node.word, node.word_type, i, i + len(node.word)))
                j += 1
        return results

# 全局变量，用于在worker进程中存储模型
global_seg = None

def init_worker(pkuseg_model_path):
    """Worker进程初始化函数"""
    global global_seg
    try:
        if pkuseg_model_path and os.path.exists(pkuseg_model_path):
            global_seg = pkuseg.pkuseg(model_name=pkuseg_model_path)
        else:
            global_seg = pkuseg.pkuseg()
    except Exception as e:
        print(f"Failed to load pkuseg model in worker: {e}")
        # 再次尝试
        import time
        time.sleep(random.uniform(0.1, 1.0))
        try:
            if pkuseg_model_path and os.path.exists(pkuseg_model_path):
                global_seg = pkuseg.pkuseg(model_name=pkuseg_model_path)
            else:
                global_seg = pkuseg.pkuseg()
        except Exception as e2:
            print(f"Retry failed: {e2}")
            global_seg = None

def process_chunk_worker_wrapper(args):
    """辅助函数，用于解包参数以适配imap"""
    return process_chunk_worker(*args)

def process_chunk_worker(
    chunk_lines: List[str], 
    word_list_path: str, 
    max_context_length: int,
    output_dir: str,
    worker_id: int,
    pkuseg_model: str = None
) -> Dict[str, int]:
    """
    工作进程函数
    """
    global global_seg
    
    # 如果初始化失败，尝试在函数内加载（虽然不推荐，但作为fallback）
    if global_seg is None:
        try:
            if pkuseg_model and os.path.exists(pkuseg_model):
                seg = pkuseg.pkuseg(model_name=pkuseg_model)
            else:
                seg = pkuseg.pkuseg()
        except Exception as e:
            print(f"Worker {worker_id}: Failed to load pkuseg model locally: {e}")
            return {'existing_words': 0, 'positive_words': 0, 'negative_words': 0}
    else:
        seg = global_seg
    
    with open(word_list_path, 'r', encoding='utf-8') as f:
        word_lists_data = json.load(f)
    
    trie = Trie()
    word_senses = {}
    
    for word_type in ['existing_words', 'positive_words', 'negative_words']:
        items = word_lists_data.get(word_type, [])
        senses = {}
        for item in items:
            if isinstance(item, list) and len(item) >= 2:
                word, sense = item[0], item[1]
                trie.insert(word, word_type)
                senses[word] = sense
            elif isinstance(item, str):
                word = item
                trie.insert(word, word_type)
                senses[word] = ""
        word_senses[word_type] = senses

    # 准备输出文件
    output_files = {}
    for wt in ['existing_words', 'positive_words', 'negative_words']:
        # 使用进程ID作为文件名后缀，避免锁竞争
        p = Path(output_dir) / f"{wt}_temp_{worker_id}.jsonl"
        # 使用 'a' 模式追加，防止覆盖（虽然每个worker_id应该是唯一的，但为了安全）
        # 并且设置 buffering=1 (行缓冲) 或者更大的buffer，避免频繁IO
        output_files[wt] = open(p, 'a', encoding='utf-8', buffering=8192)

    stats = {'existing_words': 0, 'positive_words': 0, 'negative_words': 0}

    try:
        # 设置超时机制，防止某个chunk处理时间过长
        # 但在多进程worker中设置signal.alarm比较复杂，这里采用简单的计数器检查
        
        for line_idx, line in enumerate(chunk_lines):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if not text:
                    continue
                
                # 长度检查，防止pkuseg卡死
                # 进一步降低长度限制，50000可能还是太长
                if len(text) > 10000:
                    # print(f"Worker {worker_id}: Skipping line {line_idx} due to length {len(text)}")
                    continue

                # 使用Trie树一次性搜索所有类型的词
                matches = trie.search_in_text(text)
                
                if not matches:
                    continue

                # 使用pkuseg分词并获取边界
                try:
                    # 增加异常捕获，防止seg.cut内部错误导致进程崩溃
                    words = seg.cut(text) # 返回 [word, ...]
                    boundaries = {} 
                    curr = 0
                    boundaries[0] = None # Start
                    
                    # 构建位置索引：pos -> (word_segment, end_pos)
                    # 用于后续检查匹配词是否由完整的切分词组成
                    pos_map = {} 
                    
                    for w in words:
                        start = curr
                        end = curr + len(w)
                        pos_map[start] = (w, end)
                        curr = end
                        boundaries[curr] = None
                        
                except Exception as e:
                    print(f"分词错误: {e}")
                    continue
                
                # 按位置去重（可选，这里暂不去重，允许重叠）
                
                for word, word_type, start_pos, end_pos in matches:
                    # 1. 边界检查：确保匹配的词与分词边界对齐
                    if start_pos not in boundaries or end_pos not in boundaries:
                        continue
                        
                    # 2. 结构检查
                    # 检查构成该词的所有切分词
                    
                    curr_chk = start_pos
                    valid_structure = True
                    
                    while curr_chk < end_pos:
                        if curr_chk not in pos_map:
                            # 理论上不会发生，因为start_pos和end_pos都在boundaries中
                            valid_structure = False
                            break
                        w_seg, w_end = pos_map[curr_chk]
                        curr_chk = w_end
                    
                    if not valid_structure:
                        continue

                    # 提取上下文
                    word_length = end_pos - start_pos
                    if len(text) <= max_context_length:
                        context = text
                        word_start = start_pos
                    else:
                        context_budget = max_context_length - word_length
                        left_budget = context_budget // 2
                        right_budget = context_budget - left_budget
                        
                        context_start = max(0, start_pos - left_budget)
                        context_end = min(len(text), end_pos + right_budget)
                        
                        if start_pos - context_start < left_budget:
                            context_end = min(len(text), context_end + (left_budget - (start_pos - context_start)))
                        if context_end - end_pos < right_budget:
                            context_start = max(0, context_start - (right_budget - (context_end - end_pos)))
                            
                        context = text[context_start:context_end]
                        word_start = start_pos - context_start

                    result = {
                        'word': word,
                        'sense': word_senses[word_type].get(word, ""),
                        'context': context,
                        'word_start': word_start,
                        'word_end': word_start + len(word)
                    }
                    
                    output_files[word_type].write(json.dumps(result, ensure_ascii=False) + '\n')
                    stats[word_type] += 1
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
                
    finally:
        for f in output_files.values():
            f.close()
            
    return stats

class CorpusFilter:
    """从大规模JSONL语料库中提取包含目标词的上下文 (多进程优化版)"""
    
    def __init__(
        self,
        word_list_path: str,
        corpus_path: str,
        output_dir: str,
        max_context_length: int = 2000,
        num_threads: int = 8, # 这里实际代表进程数
        chunk_size: int = 5000, # 增大chunk size以减少进程开销
        pkuseg_model: str = None
    ):
        self.word_list_path = word_list_path
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.max_context_length = max_context_length
        self.num_processes = num_threads # 重命名为num_processes
        self.chunk_size = chunk_size
        self.pkuseg_model = pkuseg_model
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_results(self, num_chunks: int):
        """合并所有临时文件"""
        print("正在合并结果文件...")
        file_types = ['existing_words', 'positive_words', 'negative_words']
        
        for ft in file_types:
            final_path = self.output_dir / f"{ft}_contexts.jsonl"
            with open(final_path, 'w', encoding='utf-8') as outfile:
                for i in range(num_chunks):
                    temp_path = self.output_dir / f"{ft}_temp_{i}.jsonl"
                    if temp_path.exists():
                        with open(temp_path, 'r', encoding='utf-8') as infile:
                            # 使用shutil.copyfileobj可能更快，但这里为了简单直接读写
                            for line in infile:
                                outfile.write(line)
                        # 删除临时文件
                        os.remove(temp_path)
            print(f"已生成: {final_path}")

    def read_chunks(self):
        """生成器:按块读取文件"""
        chunk = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk.append(line)
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    
    def process(self):
        """主处理函数:使用多进程处理整个语料库"""
        print(f"\n开始处理语料库: {self.corpus_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"进程数: {self.num_processes}")
        print(f"每批处理行数: {self.chunk_size}")
        print("-" * 80)
        
        start_time = time.time()
        
        # 收集所有chunks
        chunks = list(self.read_chunks())
        total_chunks = len(chunks)
        print(f"共 {total_chunks} 个数据块，准备开始多进程处理...")
        
        total_stats = {'existing_words': 0, 'positive_words': 0, 'negative_words': 0}
        
        # 使用ProcessPoolExecutor
        # 注意：当数据量非常大时，将所有数据加载到内存(chunks列表)并传递给子进程可能会导致内存溢出或IPC通信卡死
        # 尤其是当chunk_size很大且进程数很多时。
        # 更好的做法是只传递文件偏移量，让子进程自己去读文件，或者减小chunk_size。
        # 但为了最小化改动，我们先尝试减小传递给starmap的数据量，或者使用imap。
        
        # 这里的问题可能是starmap一次性提交所有任务，导致管道塞满。
        # 我们可以改用imap，并且不一次性生成所有tasks列表，而是用生成器。
        # 但由于read_chunks已经把文件读进内存了，内存占用已经发生了。
        
        # 既然已经读进内存了，我们尝试使用imap_unordered来处理，它比starmap更流式。
        # 为了适配imap，我们需要把参数打包。
        
        # 使用ProcessPoolExecutor
        # 注意：当数据量非常大时，将所有数据加载到内存(chunks列表)并传递给子进程可能会导致内存溢出或IPC通信卡死
        # 尤其是当chunk_size很大且进程数很多时。
        # 更好的做法是只传递文件偏移量，让子进程自己去读文件，或者减小chunk_size。
        # 但为了最小化改动，我们先尝试减小传递给starmap的数据量，或者使用imap。
        
        # 这里的问题可能是starmap一次性提交所有任务，导致管道塞满。
        # 我们可以改用imap，并且不一次性生成所有tasks列表，而是用生成器。
        # 但由于read_chunks已经把文件读进内存了，内存占用已经发生了。
        
        # 既然已经读进内存了，我们尝试使用imap_unordered来处理，它比starmap更流式。
        # 为了适配imap，我们需要把参数打包。
        
        # 优化：不一次性生成所有task_args，而是使用生成器，减少内存占用
        # 但chunks列表本身已经占用了内存。如果chunks很大，应该修改read_chunks为生成器，
        # 并且在这里也不要list(self.read_chunks())。
        # 但为了不大幅重构，我们先假设内存足够，只是IPC卡死。
        
        task_args = ((chunk, self.word_list_path, self.max_context_length, str(self.output_dir), i, self.pkuseg_model) 
                     for i, chunk in enumerate(chunks))
        
        # 使用initializer来初始化worker进程中的pkuseg模型
        # 减少maxtasksperchild，强制定期重启子进程，释放内存和潜在的死锁
        with multiprocessing.Pool(processes=self.num_processes, initializer=init_worker, initargs=(self.pkuseg_model,), maxtasksperchild=10) as pool:
            # 使用imap_unordered，这样可以尽快释放处理完的chunk的内存（虽然在主进程里chunks列表还在引用它）
            # 关键是它允许我们在tqdm中看到进度，并且不会像starmap那样一次性等待所有结果
            
            # 定义一个辅助函数来解包参数，因为imap只接受一个参数
            # 但由于我们不能在类方法里定义可被pickle的函数，我们需要把worker改成接受元组，或者用starmap
            # 实际上 pool.starmap 是不支持 tqdm 迭代的，它是一次性返回列表。
            # 上面的代码 `for stat in tqdm(pool.starmap(...))` 其实是等 starmap 全部跑完返回一个大列表后，tqdm 瞬间跑完。
            # 这就是为什么你看不到进度条更新，直到最后突然结束（或者卡死在starmap内部）。
            
            # 正确的做法是使用 imap_unordered + 包装函数
            
            results = []
            # 增加chunksize，减少IPC交互频率
            for stat in tqdm(pool.imap_unordered(process_chunk_worker_wrapper, task_args, chunksize=1), total=total_chunks, desc="Processing chunks"):
                for k, v in stat.items():
                    total_stats[k] += v
                    
        # 主动释放内存
        del chunks
        
        # 合并文件
        self.merge_results(total_chunks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 80)
        print(f"处理完成! 耗时: {duration:.2f} 秒")
        print(f"统计结果:")
        print(f"  - existing_words: {total_stats['existing_words']}")
        print(f"  - positive_words: {total_stats['positive_words']}")
        print(f"  - negative_words: {total_stats['negative_words']}")

def main():
    parser = argparse.ArgumentParser(
        description='从大规模JSONL语料库中提取包含目标词的上下文',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--word_list',
        type=str,
        required=True,
        help='目标词列表JSON文件路径'
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        required=True,
        help='输入的JSONL语料库文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出文件夹路径'
    )
    
    parser.add_argument(
        '--max_context_length',
        type=int,
        default=200,
        help='上下文最大长度(字符数)'
    )
    
    parser.add_argument(
        '--num_threads',
        type=int,
        default=8,
        help='工作线程数'
    )
    
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=10000,
        help='每次处理的行数'
    )

    parser.add_argument(
        '--pkuseg_model',
        type=str,
        default=None,
        help='pkuseg模型路径 (如果已下载到本地)'
    )
    
    args = parser.parse_args()
    
    # 创建过滤器并处理
    filter_obj = CorpusFilter(
        word_list_path=args.word_list,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        max_context_length=args.max_context_length,
        num_threads=args.num_threads,
        chunk_size=args.chunk_size,
        pkuseg_model=args.pkuseg_model
    )
    
    filter_obj.process()


if __name__ == '__main__':
    main()
