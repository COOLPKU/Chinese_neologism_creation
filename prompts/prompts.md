# Prompt templates

Prompts used by the three LLM-dependent stages, transcribed from the released code and aligned with the paper's appendix. Placeholders are shown as `{...}`.

---

## 1. Definition generation (Stage 1 enrichment)

Model: Claude-3.7-Sonnet. Source: `code/enrichment/definition_generate.py`.

### System prompt (Chinese)

```
你是一位专业的词典编撰专家。现在你需要基于汉语二字词的一些信息，为汉语二字新词编撰准确、简洁的释义。
你获得的信息包括：前字的字形、前字的语素类（词性）、前字的释义、后字的字形、后字的语素类（词性）、后字的释义，以及它们之间的构词结构关系（如：定中、主谓等）。

任务要求：
1. 理解两个字通过给定的构词结构组成一个潜在的新词
2. 基于字的语素类（词性）、语义信息和它们之间的关系类型生成合成词的释义
3. 释义应该准确反映合成词的核心含义
4. 释义应该简洁明了，通常 1-2 句话，适用于词典
5. 直接输出释义内容，不需要额外说明

请根据以下信息生成合成词的释义：
```

### User prompt template

```
字符组合信息：
- 前字: {head_char}
- 前字字符词性: {head_pos}
- 前字释义: {head_sense}
- 后字: {tail_char}
- 后字词性: {tail_pos}
- 后字释义: {tail_sense}
- 构词结构: {relation}

该词({head_char}{tail_char})的释义为：
```

---

## 2. Context sentence generation (Stage 1 enrichment)

Models: five LLMs, one sentence each (Gemini-2.0-flash, Gemini-2.5-flash, GPT-4o-mini, GPT-4.1-mini, GPT-5-mini).
Source: `code/enrichment/sample_sentence_generate.py`.
The target word form is masked in the prompt; the generated sentence carries the placeholder `～` at the target position.

### System prompt (Chinese)

```
你是一位专业的语言学家和例句编写专家。现在你需要为给定的汉语合成词创作自然、地道的例句。

任务要求：
1. 根据提供的释义，创作一个自然流畅的例句
2. 例句应该准确体现词语的含义和用法
3. 例句应该符合现代汉语的表达习惯
4. 例句长度适中，通常10-30个字
5. 直接输出例句，不需要额外说明或编号
6. 在例句中用～代替目标词。例如，词语是“电脑”，例句可以是“我用～工作和娱乐。”

请为以下词语创作一个例句：
```

### User prompt template

```
词语：～
释义：{definition}

例句：
```

---

## 3. Semantic verification of candidates (Stage 3)

Models: GPT-4o, GPT-5-mini, Gemini-2.5-Flash; a candidate is accepted with at least two positive votes (majority vote $\geq 2$).
Source: `code/stage3_llm_verification/api_check.py` (`PROMPT_TEMPLATE` + `CRITERIA_TEXT`).

### User prompt template (Chinese)

```
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
{
  "word": "{candidate_word}",
  "judgement": "YES 或 NO",
  "reason": "简短理由"
}
```

### Judging criteria (`CRITERIA_TEXT`)

```
(1) 语法上合理 —— 满足给定语法关系下的构词规则，词性搭配恰当；
(2) 语义上合理 —— 能表达自洽、易懂且与两字释义相关的意义，避免牵强附会；
(3) 语感可接受 —— 常用语感/读音自然，不与常见常用词读音或意义形成混淆。
```

---

## 4. Discriminative filter training templates (Stage 3)

Four templates (full word input, first morpheme masked, and variants) are defined in
`code/stage3_filtering/corpus_filter.py` and reproduced in the paper's appendix
(section "Prompt Templates"). The second template masks one morpheme so that the model
cannot solve the task through surface-form memorization.

---

## 5. Human annotation instructions

The annotation guidelines given to the three expert annotators (layered morphology →
semantics → pragmatics criteria, including the "bad example sentence but plausible word"
rule) are reproduced in the paper's appendix (section "Annotation Guidelines for Human
Judgment"); the raw annotations are in `data/evaluation/annotation/`.
