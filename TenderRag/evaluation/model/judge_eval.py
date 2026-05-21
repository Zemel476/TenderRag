# -*- coding: utf-8 -*-
"""
judge_eval.py - LLM Judge 评估方案

替代 RAGAS，直接用 LLM 打分，速度快、节省 token。

Author: shui-
Date: 2026/4/23
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


EVAL_MODEL = "qwen3.5-plus-2026-04-20"
EVAL_API_KEY = ""
EVAL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

INSTRUCTION = "回答必须基于上下文的法律条文，不能编造，且需注明依据的条款号。"

TARGET_MODELS = [
    {
        "model_name": "future12/qwen2.5_7b_law",
        "display_name": "future12/qwen2.5_7b_law", # 通义千问2.5-7B-法律领域微调模型
        "base_url": "http://1390997827248152.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/qwen_2_7_legal/v1",
        "api_key": ""
    },
]

QA_INPUT = Path(__file__).parent / "output.json"
OUTPUT_CSV = Path(__file__).parent / f"ragas_evaluation_results_{time.time()}.csv"

JUDGE_PROMPT = """你是 RAG 系统的评估专家，请对模型回答进行以下三个维度的打分：

【评估要求】
1. **忠实度 (0-1)**：答案是否忠实于参考资料，没有编造信息。
   - 1.0：完全忠实，所有信息都能在参考资料中找到依据
   - 0.5：部分编造，有一些资料中没有的内容
   - 0.0：严重幻觉，大量信息是编造的

2. **答案相关性 (1-5)**：答案是否正面回应了问题。
   - 5分：完全紧扣问题，精准回答
   - 4分：基本相关，有少量偏离
   - 3分：部分相关，答非所问的内容较多
   - 2分：仅少量内容与问题相关
   - 1分：完全不相关

3. **指令遵循度 (1-5)**：答案是否遵循以下指令要求。
   指令：{instruction}
   - 5分：完全遵循
   - 4分：基本遵循
   - 3分：部分遵循
   - 2分：仅有少量遵循
   - 1分：完全不遵循

【输出格式】
请严格按 JSON 格式输出，不要任何额外内容：
{{"忠实度": 0.8, "答案相关性": 4, "指令遵循度": 3}}

【待评估内容】
问题：{question}
参考资料：{context}
模型回答：{answer}
"""


def generate_answer(
    client: OpenAI,
    model_name: str,
    question: str,
    contexts: list[str],
    extra_body: dict | None = None,
) -> tuple[str, float]:
    """调用目标模型生成答案，返回 (答案, 耗时秒)"""
    context_text = "\n\n".join(contexts)
    prompt = f"请根据以下参考资料回答问题。\n\n参考资料：\n{context_text}\n\n问题：{question}\n\n回答要求：{INSTRUCTION}\n"
    start = time.time()
    for attempt in range(3):
        try:
            prompt = prompt[:500]
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body=extra_body
            )
            return resp.choices[0].message.content or "", time.time() - start
        except Exception as e:
            if attempt < 2:
                logger.warning("API 调用失败 (第%d次): %s, 重试中...", attempt + 1, e)
                time.sleep(2)
            else:
                logger.error("API 调用最终失败: %s", e)
                return f"[ERROR] {e}", time.time() - start


def judge_single(
    eval_client: OpenAI,
    question: str,
    context: str,
    answer: str,
) -> dict:
    """单条 LLM Judge 评估"""
    prompt = JUDGE_PROMPT.format(
        instruction=INSTRUCTION,
        question=question,
        context=context[:3000],  # 截断过长的 context
        answer=answer,
    )
    for attempt in range(3):
        try:
            resp = eval_client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256, # 模型输出限制
            )
            text = resp.choices[0].message.content.strip()
            # 清理可能的 markdown 包裹
            text = text.strip("`").lstrip("json").strip()
            result = json.loads(text)
            return {
                "忠实度": float(result.get("忠实度", 0.5)),
                "答案相关性": float(result.get("答案相关性", 3)),
                "指令遵循度": float(result.get("指令遵循度", 3)),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error("Judge 失败: %s", e)
                return {"忠实度": 0.5, "答案相关性": 3.0, "指令遵循度": 3.0}


def evaluate_single_model(
    eval_client: OpenAI,
    qa_data: list[dict],
    model_config: dict,
) -> pd.DataFrame:
    """评估单个模型"""
    display = model_config["display_name"]
    model_name = model_config["model_name"]
    logger.info("=== 开始评估模型: %s (%s) ===", display, model_name)

    client = OpenAI(base_url=model_config.get("base_url", EVAL_BASE_URL),
                    api_key=model_config.get("api_key", EVAL_API_KEY),
                    timeout=30)

    # 批量生成答案
    questions, answers, contexts_list, times = [], [], [], []
    for i, item in enumerate(qa_data, 1):
        q = item["question"]
        ctx = item["context"]
        ans, t = generate_answer(client, model_name, q, ctx, extra_body={"enable_thinking": False})
        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctx)
        times.append(round(t, 2))
        logger.info("  [%d/%d] 耗时 %.2fs | %s...", i, len(qa_data), t, q[:30])

    # 批量 Judge 评估
    logger.info("  正在评估忠实度/相关性/指令遵循度...")
    scores = []
    for i, (q, ctx, ans) in enumerate(zip(questions, contexts_list, answers), 1):
        if ans.startswith("[ERROR]"):
            logger.warning("  [%d/%d] 答案生成失败，跳过 Judge 评估: %s", i, len(questions), ans[:80])
            s = {"忠实度": 0.0, "答案相关性": 1.0, "指令遵循度": 1.0}
        else:
            s = judge_single(eval_client, q, ctx, ans)
        scores.append(s)
        logger.info("  [%d/%d] 忠实度=%s, 相关性=%s, 指令=%s",
                     i, len(questions), s["忠实度"], s["答案相关性"], s["指令遵循度"])

    df = pd.DataFrame({
        "模型名称": display,
        "模型请求时间(s)": times,
        "问题": questions,
        "生成答案": answers,
        "忠实度": [s["忠实度"] for s in scores],
        "答案相关性": [s["答案相关性"] for s in scores],
        "指令遵循度": [s["指令遵循度"] for s in scores],
    })

    # 各指标已经是 0-1 范围，忠实度 0-1、相关性/指令遵循度归一化到 0-1
    df["忠实度_norm"] = df["忠实度"]
    df["答案相关性_norm"] = df["答案相关性"] / 5
    df["指令遵循度_norm"] = df["指令遵循度"] / 5

    return df


def main():
    with open(QA_INPUT, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    logger.info("加载 %d 条问答对", len(qa_data))

    eval_client = OpenAI(base_url=EVAL_BASE_URL, api_key=EVAL_API_KEY)

    all_results = []
    for model_cfg in TARGET_MODELS:
        model_df = evaluate_single_model(eval_client, qa_data, model_cfg)
        all_results.append(model_df)

    combined = pd.concat(all_results, ignore_index=True)

    # 时间做 min-max 归一化（越慢效率越低）
    t = combined["模型请求时间(s)"]
    t_min, t_max = t.min(), t.max()
    t_range = t_max - t_min if t_max != t_min else 1.0
    combined["效率"] = 1 - (t - t_min) / t_range

    # 综合得分 = 0.4×忠实度 + 0.3×答案相关性 + 0.2×指令遵循度 + 0.1×效率
    combined["综合得分"] = (
        0.4 * combined["忠实度_norm"]
        + 0.3 * combined["答案相关性_norm"]
        + 0.2 * combined["指令遵循度_norm"]
        + 0.1 * combined["效率"]
    )

    # 汇总
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    summary = combined.groupby("模型名称").agg({
        "忠实度": "mean",
        "答案相关性": "mean",
        "指令遵循度": "mean",
        "模型请求时间(s)": "mean",
        "效率": "mean",
        "综合得分": "mean",
    }).round(4)
    print(summary.to_string())
    print("=" * 60)

    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("结果已导出至: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()