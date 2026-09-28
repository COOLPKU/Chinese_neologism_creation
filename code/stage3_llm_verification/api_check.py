"""批量调用大模型验证新词候选的合法性。

使用方式示例：

	python -m discriminate.api_based_process.api_check \
		--input discriminate/api_based_process/filtered_prediction_pairs_step4.json \
		--output discriminate/api_based_process/api_checked_pairs.json \
		--models gpt-5,gemini-3-pro-preview,claude-3-7-sonnet@20250219 \
		--max-count 100

默认会对每个候选调用 3 次（可指定具体模型或重复使用同一个模型）并进行少数服从多数（≥2 票即通过）。
支持 --workers 参数并行处理多条记录，并提供 tqdm 进度条显示处理进度，可通过 --dry-run 查看提示词和输出路径而不真正调 API。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional visual progress dependency
	from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
	tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from Util.utils import (  # noqa: E402
	extract_json_from_text,
	get_response,
	load_client,
)


DEFAULT_MODELS = [
	"gpt-5-mini",
	"gemini-2.5-flash",
	"gpt-4o",
]

SYSTEM_PROMPT = (
	"你是一位严谨的汉语词汇学家，需要依据给定的构词信息判断一个候选二字词" \
	"是否能在汉语中成立。请严格结合语法、语义与语感三方面，确保词性搭配、" \
	"语义逻辑与常用语感一致，并注意读音不要与常见词产生歧义。"
)

CRITERIA_TEXT = """
判断标准：
1. 语法上合理 —— 满足给定语法关系下的构词规则，词性搭配恰当；
2. 语义上合理 —— 能表达自洽、易懂且与两字释义相关的意义，避免牵强附会；
3. 语感可接受 —— 常用语感/读音自然，不与常见常用词读音或意义形成混淆。
"""

PROMPT_TEMPLATE = """
请审查下述候选新词并仅返回 JSON。判断要求：
1. 候选词由【首字】和【尾字】按给定语法关系组合，需符合汉语构词习惯；
2. 若组合能表达自洽、易懂且可在真实语境中使用的概念，则判定为 YES；
3. 若语义牵强、搭配不当或违背语法语义常识，则判定为 NO；
4. 严格依据以下三条判断标准：
{criteria}
5. 请在 reason 中用不超过 40 个汉字解释理由。

候选信息：
- 首字：{head_char}（{head_pos}，释义：{head_sense}）
- 尾字：{tail_char}（{tail_pos}，释义：{tail_sense}）
- 语法关系：{relation}
- 几个可能的语义组合示例：
  * 将“首字”视为主要语义成分，尾字作搭配补充；
  * 若语法关系为定中/主谓/状中，请按常见构词规则分析其自然度。

输出格式（必须是单个 JSON 对象，不得包含额外文字）：
{{
  "word": "{candidate_word}",
  "judgement": "YES 或 NO",
	"reason": "简短理由"
}}
"""


@dataclass
class VoteResult:
	model: str
	judgement: str
	reason: str
	raw_response: str

	@property
	def is_positive(self) -> bool:
		return self.judgement.upper() in {"YES", "Y", "TRUE", "VALID", "成立"}

	def to_dict(self) -> Dict[str, str]:
		payload = asdict(self)
		payload["is_positive"] = self.is_positive
		return payload


def parse_vote_response(raw_text: str, fallback_word: str) -> Tuple[str, str]:
	"""将模型输出解析为 (judgement, reason)。"""

	if not raw_text:
		return "ERROR", "空响应"

	candidate_json = None
	try:
		candidate_json = json.loads(raw_text)
	except json.JSONDecodeError:
		candidate_json = extract_json_from_text(raw_text)
		if candidate_json == "None":
			candidate_json = None

	if isinstance(candidate_json, dict):
		judgement = str(
			candidate_json.get("judgement")
			or candidate_json.get("decision")
			or candidate_json.get("verdict")
			or candidate_json.get("is_valid")
			or ""
		).strip()
		reason = str(
			candidate_json.get("reason")
			or candidate_json.get("analysis")
			or candidate_json.get("comment")
			or ""
		).strip()
	else:
		judgement = "ERROR"
		reason = raw_text[:200]

	if not judgement:
		judgement = "ERROR"
	if not reason:
		reason = "无法解析模型输出"
	return judgement, reason


def build_prompt(entry: Dict[str, object]) -> Tuple[str, str]:
	head_char = str(entry.get("head_id", "?"))[:1]
	tail_char = str(entry.get("tail_id", "?"))[:1]
	candidate_word = f"{head_char}{tail_char}"
	prompt = PROMPT_TEMPLATE.format(
		head_char=head_char,
		head_pos=entry.get("head_pos", "未知词性"),
		head_sense=entry.get("head_sense", "未提供释义"),
		tail_char=tail_char,
		tail_pos=entry.get("tail_pos", "未知词性"),
		tail_sense=entry.get("tail_sense", "未提供释义"),
		relation=entry.get("relation", "未知关系"),
		confidence=entry.get("confidence", "未知"),
		criteria=CRITERIA_TEXT,
		candidate_word=candidate_word,
	)
	return candidate_word, prompt


def build_call_plan(model_names: Sequence[str], votes_per_entry: int) -> List[str]:
	if votes_per_entry <= 0:
		raise ValueError("votes_per_entry 必须为正整数")
	if not model_names:
		raise ValueError("至少需要一个模型名称用于调用")
	plan = []
	for idx in range(votes_per_entry):
		plan.append(model_names[idx % len(model_names)])
	return plan


def call_model(
	client,
	model_name: str,
	prompt: str,
	*,
	system_prompt: str,
	retry: int = 2,
	backoff: float = 2.5,
) -> str:
	"""调用单个模型并在失败时指数回退重试。"""

	attempt = 0
	last_err: Optional[str] = None
	while attempt <= retry:
		try:
			message = [{"role": "user", "content": prompt}]
			response = get_response(
				client=client,
				model=model_name,
				stream=False,
				think=False,
				system_prompt=system_prompt,
				cur_message=message,
			)
			if isinstance(response, list):
				# 某些 SDK 可能返回列表
				response = "\n".join(str(chunk) for chunk in response)
			return str(response)
		except Exception as err:  # noqa: BLE001
			last_err = f"{type(err).__name__}: {err}"
			if attempt == retry:
				break
			time.sleep(backoff * (attempt + 1))
			attempt += 1
	raise RuntimeError(f"调用模型 {model_name} 失败: {last_err}")


def evaluate_entry(
	entry: Dict[str, object],
	clients: Dict[str, object],
	call_plan: Sequence[str],
	*,
	system_prompt: str,
	inter_call_sleep: float,
	dry_run: bool = False,
) -> Tuple[bool, List[VoteResult], str]:
	candidate_word, prompt = build_prompt(entry)
	if dry_run:
		votes: List[VoteResult] = [
			VoteResult(
				model=model,
				judgement="SKIPPED",
				reason="dry-run 模式未调用 API",
				raw_response="",
			)
			for model in call_plan
		]
		return False, votes, candidate_word

	votes = []
	positive_count = 0
	for idx, model_name in enumerate(call_plan):
		client = clients[model_name]
		raw = call_model(
			client,
			model_name,
			prompt,
			system_prompt=system_prompt,
		)
		judgement, reason = parse_vote_response(raw, candidate_word)
		vote = VoteResult(
			model=model_name,
			judgement=judgement,
			reason=reason,
			raw_response=raw,
		)
		if vote.is_positive:
			positive_count += 1
		votes.append(vote)
		if idx < len(call_plan) - 1 and inter_call_sleep > 0:
			time.sleep(inter_call_sleep)

	required_votes = len(call_plan)
	is_valid = positive_count >= ((required_votes // 2) + 1)
	return is_valid, votes, candidate_word


def iter_entries(
	path: Path,
	start: int = 0,
	max_count: Optional[int] = None,
) -> Iterable[Tuple[int, Dict[str, object]]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	end_index = len(data) if max_count is None else min(len(data), start + max_count)
	for idx in range(start, end_index):
		yield idx, data[idx]


def load_clients(model_names: Sequence[str], gemini_config: Optional[str]) -> Dict[str, object]:
	clients = {}
	for model_name in model_names:
		config_file = gemini_config if "gemini" in model_name else None
		clients[model_name] = load_client(model_name, config_file=config_file)
	return clients


def save_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)



def run_pipeline(args: argparse.Namespace) -> None:
	model_names = [name.strip() for name in args.models.split(",") if name.strip()]
	if not model_names:
		raise ValueError("必须至少指定一个模型名称")
	if args.workers <= 0:
		raise ValueError("--workers 必须为正整数")
	call_plan = build_call_plan(model_names, args.votes_per_entry)

	clients = {name: None for name in model_names}
	if not args.dry_run:
		clients = load_clients(model_names, args.gemini_config)

	input_path = Path(args.input).resolve()
	output_path = Path(args.output).resolve()
	rejected_detail: List[Dict[str, object]] = []
	accepted_entries: List[Dict[str, object]] = []
	results_lock = Lock()
	total_counter = 0

	entries = list(
		iter_entries(
			input_path,
			start=args.start_index,
			max_count=args.max_count,
		)
	)
	progress_bar = None
	if not args.no_progress:
		if tqdm is None:
			print("[提示] 未安装 tqdm，无法显示进度条。可运行 `pip install tqdm` 或添加 --no-progress 关闭提示。")
		elif entries:
			progress_bar = tqdm(total=len(entries), desc="验证进度", unit="词")

	def worker(args_tuple: Tuple[int, Dict[str, object]]):
		idx, entry = args_tuple
		is_valid, votes, candidate_word = evaluate_entry(
			entry,
			clients,
			call_plan,
			system_prompt=SYSTEM_PROMPT,
			inter_call_sleep=args.sleep,
			dry_run=args.dry_run,
		)
		enriched_entry = dict(entry)
		enriched_entry["word"] = candidate_word
		enriched_entry["votes"] = [vote.to_dict() for vote in votes]
		enriched_entry["accepted"] = is_valid
		return idx, enriched_entry

	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = [executor.submit(worker, item) for item in entries]
		for future in as_completed(futures):
			idx, enriched_entry = future.result()
			votes = enriched_entry["votes"]
			is_valid = enriched_entry["accepted"]
			with results_lock:
				target_list = accepted_entries if is_valid else rejected_detail
				target_list.append(enriched_entry)
				total_counter += 1
				if args.flush_every and total_counter % args.flush_every == 0:
					save_json(output_path, accepted_entries)
					if args.rejected_output:
						save_json(Path(args.rejected_output).resolve(), rejected_detail)
			if progress_bar:
				progress_bar.update(1)
			if args.verbose:
				status = "✅" if is_valid else "❌"
				positive_votes = sum(v["is_positive"] for v in votes)
				print(f"[{idx}] {enriched_entry['word']} {status} ({positive_votes}/{len(votes)})")

	if progress_bar:
		progress_bar.close()

	save_json(output_path, accepted_entries)
	if args.rejected_output:
		save_json(Path(args.rejected_output).resolve(), rejected_detail)

	print(
		f"处理完成：保留 {len(accepted_entries)} 条，拒绝 {len(rejected_detail)} 条。\n"
		f"输出文件：{output_path}"
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="使用 LLM 复核新词候选")
	parser.add_argument(
		"--input",
		default=str(
			PROJECT_ROOT
			/ "discriminate"
			/ "api_based_process"
			/ "filtered_prediction_pairs_step4.json"
		),
		help="输入候选 JSON 路径",
	)
	parser.add_argument(
		"--output",
		default=str(
			PROJECT_ROOT
			/ "discriminate"
			/ "api_based_process"
			/ "api_checked_pairs.json"
		),
		help="保留（通过）结果输出路径",
	)
	parser.add_argument(
		"--rejected-output",
		default="",
		help="可选：保存被拒绝样本的路径",
	)
	parser.add_argument(
		"--models",
		default=",".join(DEFAULT_MODELS),
		help="逗号分隔的模型列表，可重复同一模型以实现多次判定",
	)
	parser.add_argument(
		"--votes-per-entry",
		type=int,
		default=3,
		help="每个候选需要的判定次数，默认 3 次",
	)
	parser.add_argument(
		"--gemini-config",
		default="",
		help="可选：Gemini 服务账号配置文件路径",
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=0,
		help="从第几条开始处理（用于断点续跑）",
	)
	parser.add_argument(
		"--max-count",
		type=int,
		default=None,
		help="限制处理的样本数量",
	)
	parser.add_argument(
		"--sleep",
		type=float,
		default=1.0,
		help="每个模型调用之间的休眠秒数，避免限流",
	)
	parser.add_argument(
		"--flush-every",
		type=int,
		default=50,
		help="每处理多少条强制落盘一次",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=40,
		help="并行处理的线程数，默认 15",
	)
	parser.add_argument(
		"--no-progress",
		action="store_true",
		help="禁用进度条显示",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="只打印提示信息，不真正调用模型",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="打印逐条判定信息",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	run_pipeline(args)


if __name__ == "__main__":
	main()
