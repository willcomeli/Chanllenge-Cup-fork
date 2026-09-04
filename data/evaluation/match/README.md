# 人岗匹配评测规范

当前开发集包含10份简历和从真实图谱冻结的14个岗位。真实流程先由LLM将简历技能对齐到冻结技能库，再使用确定性公式计算全部岗位排名；选定岗位后，由LLM结合简历、命中技能和差距生成适配建议与学习路径。核心匹配评测使用人工确认画像隔离简历解析误差，端到端评测使用系统解析结果。

确定性总分同时考虑岗位覆盖率和候选技能相关率：`2 × coverage × relevance / (coverage + relevance)`。LLM不能直接修改分数或岗位排序。

Ground Truth标注每份简历的唯一最佳岗位、可接受岗位集合、最佳岗位匹配等级、已具备必备技能和缺失必备技能。GT只能依据原始简历和冻结岗位定义制作，不得查看系统排名；当前标签状态为 `draft_pending_human_review`。

综合分权重：Top-1 30%、Top-3 15%、NDCG@3 10%、等级Macro-F1 15%、缺失技能micro-F1 20%、已匹配技能micro-F1 10%。开发通过线为综合分80%、Top-1 80%、Top-3 90%。

当前复跑结果写入 `output/evaluation/match_evaluation_report_10_v1.json`：Top-1 Accuracy 100.00%、Top-3 Accuracy 100.00%、MRR 100.00%、NDCG@3 95.31%、等级Macro-F1 100.00%、综合分95.08%。当前仍为10份开发集结果，正式评测需扩充样本并完成专家复核。

```bash
python -m src.evaluation.generate_match_predictions \
  --resume-ground-truth data/evaluation/resume/resume_ground_truth_10_v1.jsonl \
  --position-pool data/evaluation/match/position_pool_v1.jsonl \
  --output output/evaluation/match_predictions_10_v1.jsonl

python -m src.evaluation.evaluate_match_predictions \
  --ground-truth data/evaluation/match/match_ground_truth_10_v1.jsonl \
  --predictions output/evaluation/match_predictions_10_v1.jsonl \
  --position-pool data/evaluation/match/position_pool_v1.jsonl \
  --output output/evaluation/match_evaluation_report_10_v1.json \
  --allow-draft
```

端到端评测使用正式简历解析结果，单独输出报告：

```bash
python -m src.evaluation.generate_match_predictions \
  --resume-predictions output/evaluation/resume_predictions_10_v1.jsonl \
  --position-pool data/evaluation/match/position_pool_v1.jsonl \
  --llm-align \
  --output output/evaluation/match_predictions_end_to_end_10_v1.jsonl

python -m src.evaluation.evaluate_match_predictions \
  --ground-truth data/evaluation/match/match_ground_truth_10_v1.jsonl \
  --predictions output/evaluation/match_predictions_end_to_end_10_v1.jsonl \
  --position-pool data/evaluation/match/position_pool_v1.jsonl \
  --output output/evaluation/match_evaluation_report_end_to_end_10_v1.json \
  --allow-draft
```
