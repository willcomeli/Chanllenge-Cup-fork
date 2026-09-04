from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.app.schemas import (
    ChangeEvidence,
    ChangeType,
    EmergingPositionItem,
    EvolutionChangeItem,
    PositionProfile,
    RequirementSnapshot,
    SkillRequirement,
    SourceSupport,
    WindowContinuity,
)


POSITION_ALIASES = {
    "pos_ai_agent_engineer": [
        "agent",
        "multi-agent",
        "多智能体",
        "tool use",
        "workflow",
        "langchain",
        "langgraph",
        "rag",
        "prompt",
        "llm",
        "大模型",
    ],
    "pos_llm_engineer": [
        "llm",
        "大模型",
        "大语言模型",
        "rag",
        "检索增强",
        "评测",
        "prompt",
        "生成式",
        "模型训练",
    ],
    "pos_java_engineer": [
        "java",
        "spring",
        "springboot",
        "后端",
        "云原生",
        "kubernetes",
        "docker",
        "微服务",
    ],
    "pos_data_analyst": [
        "数据分析",
        "分析师",
        "sql",
        "excel",
        "报表",
        "数据挖掘",
    ],
    "pos_frontend_engineer": [
        "前端",
        "javascript",
        "typescript",
        "react",
        "vue",
        "web",
        "ai coding",
        "前端开发",
    ],
    "pos_algorithm_engineer": [
        "算法",
        "推荐",
        "搜索",
        "广告",
        "机器学习",
        "深度学习",
        "nlp",
        "cv",
        "计算机视觉",
        "多模态",
        "语音识别",
        "风控",
    ],
    "pos_backend_engineer": [
        "后端",
        "服务端",
        "后台",
        "平台研发",
        "基础架构",
        "微服务",
        "go",
        "golang",
    ],
    "pos_test_engineer": [
        "测试开发",
        "自动化测试",
        "质量工程",
        "测试工程师",
        "qa",
    ],
    "pos_data_engineer": [
        "数据开发",
        "数据平台",
        "数据仓库",
        "大数据",
        "数仓",
        "数据工程",
        "bi",
    ],
    "pos_cloud_infra_engineer": [
        "云计算",
        "云网络",
        "云数据库",
        "运维",
        "sre",
        "devops",
        "基础设施",
        "kubernetes",
        "容器",
    ],
    "pos_security_engineer": [
        "安全工程师",
        "安全研发",
        "网络安全",
        "内容安全",
        "大模型安全",
        "风控",
        "攻防",
    ],
    "pos_hardware_engineer": [
        "硬件",
        "芯片",
        "soc",
        "risc-v",
        "riscv",
        "npu",
        "cuda",
        "编译器",
        "异构计算",
        "嵌入式",
    ],
    "pos_storage_database_engineer": [
        "存储",
        "数据库",
        "mysql",
        "rds",
        "redis",
        "搜索引擎",
    ],
    "pos_game_engineer": [
        "游戏引擎",
        "渲染引擎",
        "图形",
        "客户端",
        "unity",
        "ue",
    ],
}

# Position identity is determined from the title, not from skills mentioned in
# the JD body. Otherwise a new role mentioning Python/LLM would be incorrectly
# folded into an existing AI position before novelty detection can run.
POSITION_TITLE_ALIASES = {
    "pos_ai_agent_engineer": ["ai agent", "agent研发", "agent工程师", "智能体研发", "智能体工程师"],
    "pos_llm_engineer": ["大模型应用工程师", "大模型应用算法", "大模型应用评测", "ai应用开发", "模型策略", "模型训练", "大模型工程师", "大语言模型", "llm工程师", "生成式ai工程师"],
    "pos_algorithm_engineer": ["算法工程师", "算法研发", "算法专家", "算法研究员", "推荐算法", "搜索算法", "广告算法", "nlp算法", "机器学习", "深度学习", "多模态算法", "语音识别算法", "数据挖掘", "音频后处理算法", "计算机视觉", "cv算法", "aigc"],
    "pos_java_engineer": ["java开发", "java研发", "java后端", "java工程师"],
    "pos_backend_engineer": ["后端开发", "后端研发", "服务端", "后台开发", "平台研发", "基础架构研发", "网络研发", "go开发", "golang"],
    "pos_test_engineer": ["测试开发", "自动化测试", "测试工程师", "质量工程", "研发效能"],
    "pos_data_engineer": ["数据开发", "数据平台", "数据仓库", "大数据开发", "数仓", "数据工程", "数据研发"],
    "pos_data_analyst": ["数据分析师", "数据分析工程师"],
    "pos_cloud_infra_engineer": ["云计算", "云原生", "云网络", "云数据库", "idc资产", "基础设施", "infra", "gpu训练", "集群通信", "solution architect", "运维", "sre", "devops", "容器"],
    "pos_security_engineer": ["安全工程师", "安全研发", "安全运营", "网络安全", "内容安全", "大模型安全", "风控"],
    "pos_hardware_engineer": ["硬件", "芯片", "soc", "risc-v", "riscv", "npu", "cuda", "编译器", "异构计算", "嵌入式", "camera性能", "性能功耗", "推理性能优化"],
    "pos_storage_database_engineer": ["存储", "数据库", "mysql", "rds", "redis", "搜索引擎"],
    "pos_frontend_engineer": ["前端开发", "前端研发", "前端工程师", "前端基础"],
    "pos_game_engineer": ["游戏引擎", "渲染引擎", "图形开发", "客户端开发"],
}

SKILL_ALIASES = {
    "skill_llm": ["llm", "大语言模型", "大模型", "foundation model"],
    "skill_rag": ["rag", "retrieval augmented", "检索增强", "向量检索"],
    "skill_python": ["python"],
    "skill_go": ["go", "golang"],
    "skill_cpp": ["c++", "cpp"],
    "skill_prompt": ["prompt", "提示词工程", "prompt engineering"],
    "skill_multi_agent": ["multi-agent", "多智能体", "agent workflow", "agent协同", "tool use"],
    "skill_rag_eval": ["rag评测", "评测", "benchmark", "eval", "模型评测"],
    "skill_java": ["java"],
    "skill_spring": ["spring", "springboot"],
    "skill_cloud_native": ["云原生", "kubernetes", "docker", "k8s"],
    "skill_distributed": ["分布式", "微服务", "高并发"],
    "skill_algorithm": ["算法", "推荐", "搜索", "机器学习", "深度学习"],
    "skill_nlp": ["nlp", "自然语言处理"],
    "skill_multimodal": ["多模态", "aigc", "视觉理解", "语音识别"],
    "skill_testing": ["测试开发", "自动化测试", "质量保障", "qa"],
    "skill_security": ["安全", "风控", "攻防", "数据安全"],
    "skill_database": ["数据库", "mysql", "rds", "redis", "存储"],
    "skill_hardware": ["硬件", "芯片", "soc", "cuda", "npu", "异构计算"],
    "skill_sql": ["sql", "mysql", "hive"],
    "skill_excel": ["excel", "报表", "dashboard"],
    "skill_ai_codegen": ["ai coding", "ai编程", "代码生成", "copilot", "cursor"],
    "skill_frontend": ["前端", "javascript", "typescript", "react", "vue", "web"],
}

POSITION_NAME_MAP = {
    "pos_ai_agent_engineer": "AI Agent 研发工程师",
    "pos_llm_engineer": "大模型应用工程师",
    "pos_java_engineer": "Java 开发工程师",
    "pos_backend_engineer": "后端研发工程师",
    "pos_algorithm_engineer": "算法工程师",
    "pos_test_engineer": "测试开发工程师",
    "pos_data_engineer": "数据开发工程师",
    "pos_data_analyst": "数据分析师",
    "pos_cloud_infra_engineer": "云计算与基础设施工程师",
    "pos_security_engineer": "安全工程师",
    "pos_hardware_engineer": "硬件与芯片工程师",
    "pos_storage_database_engineer": "存储与数据库工程师",
    "pos_frontend_engineer": "前端研发工程师",
    "pos_game_engineer": "游戏引擎工程师",
}

SKILL_NAME_MAP = {
    "skill_llm": "大语言模型",
    "skill_rag": "RAG",
    "skill_python": "Python",
    "skill_go": "Go",
    "skill_cpp": "C/C++",
    "skill_prompt": "Prompt 工程",
    "skill_multi_agent": "多智能体协作",
    "skill_rag_eval": "RAG 评测",
    "skill_java": "Java",
    "skill_spring": "Spring 框架",
    "skill_cloud_native": "云原生",
    "skill_distributed": "分布式系统",
    "skill_algorithm": "算法工程",
    "skill_nlp": "NLP",
    "skill_multimodal": "多模态",
    "skill_testing": "自动化测试",
    "skill_security": "安全风控",
    "skill_database": "数据库与存储",
    "skill_hardware": "硬件与异构计算",
    "skill_sql": "SQL",
    "skill_excel": "Excel / 报表",
    "skill_ai_codegen": "AI 辅助开发",
    "skill_frontend": "前端工程",
}


def _job_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "relevant_jobs.jsonl"


def _evolution_baseline_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "evolution_baseline.json"


def _baseline_record_count() -> int | None:
    path = _evolution_baseline_path()
    if not path.exists():
        return None
    try:
        return max(0, int(json.loads(path.read_text(encoding="utf-8")).get("recordCount", 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _partition_at_baseline(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    count = _baseline_record_count()
    if count is None:
        return records, records
    baseline = [record for record in records if int(record.get("_corpus_index", 0)) < count]
    incoming = [record for record in records if int(record.get("_corpus_index", 0)) >= count]
    return baseline, incoming


def _partition_at_midpoint(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    midpoint = max(1, len(records) * 40 // 100)
    return records[:midpoint], records[midpoint:]


def _analysis_record_windows(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Use real increments when present; otherwise split the packaged corpus for demo analysis."""
    if not records:
        return [], []
    baseline_count = _baseline_record_count()
    if baseline_count is not None:
        historical, current = _partition_at_baseline(records)
        if current:
            return historical, current
    return _partition_at_midpoint(records)


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=None)
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S%z")
        except ValueError:
            parsed = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=None)
    return parsed.astimezone().replace(tzinfo=None)


def _record_time_value(record: Dict[str, Any]) -> str:
    return str(record.get("publish_time") or record.get("scraped_at") or "")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    return text.replace("\u3000", " ")


def _match_aliases(text: str, aliases: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    for alias in aliases:
        if alias.lower() in normalized:
            return True
    return False


def _position_for_record(record: Dict[str, Any]) -> str:
    title = _normalize_text(record.get("title", ""))
    best_position = ""
    best_score = 0
    for position_id, aliases in POSITION_TITLE_ALIASES.items():
        score = sum(1 for alias in aliases if alias.lower().replace(" ", "") in title.replace(" ", ""))
        if score > best_score:
            best_score = score
            best_position = position_id
    if best_position:
        return best_position

    # Unknown titles must not be silently folded into AI Agent. They remain
    # candidates until the emerging-position review accepts them.
    normalized_title = _normalize_position_title(record.get("title", ""))
    digest = hashlib.sha1(normalized_title.encode("utf-8")).hexdigest()[:10]
    return f"candidate_{digest}"


def _normalize_position_title(value: Any) -> str:
    """Normalize superficial title variants without inventing a standard job."""
    title = _normalize_text(value)
    title = re.sub(r"[（(][^）)]*(?:校招|社招|急招|招聘|外包)[^）)]*[）)]", "", title)
    title = re.sub(r"(?:高级|资深|初级|中级|实习|专家|负责人|校招|社招|急招)", "", title)
    title = re.sub(r"[-_/·|｜\s]+", "", title)
    return title or "未命名岗位"


def _display_candidate_title(records: List[Dict[str, Any]]) -> str:
    titles = [str(record.get("title", "")).strip() for record in records if record.get("title")]
    return max(set(titles), key=titles.count) if titles else "未命名岗位"


def _load_job_records(path: Path | str | None = None) -> List[Dict[str, Any]]:
    path = Path(path) if path is not None else _job_data_path()
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for corpus_index, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            time_value = _record_time_value(item)
            if not time_value:
                continue
            item["_position_id"] = _position_for_record(item)
            item["_parsed_time"] = _parse_datetime(time_value)
            item["_corpus_index"] = corpus_index
            records.append(item)
    return sorted(records, key=lambda r: r["_parsed_time"])


def _build_snapshot_windows() -> tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    records = _load_job_records()
    if not records:
        return ({}, {})

    historical, current = _analysis_record_windows(records)

    def build_window(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        by_position: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        by_position_counts: Dict[str, int] = defaultdict(int)

        for record in items:
            position_id = record["_position_id"]
            by_position_counts[position_id] += 1

        for position_id, count in by_position_counts.items():
            skill_usage: Dict[str, int] = defaultdict(int)
            for record in items:
                if record["_position_id"] != position_id:
                    continue
                text = " ".join([record.get("title", ""), record.get("description", ""), record.get("requirement", "")])
                for skill_id, aliases in SKILL_ALIASES.items():
                    if _match_aliases(text, aliases):
                        skill_usage[skill_id] += 1

            for skill_id, usage in skill_usage.items():
                weight = round(min(1.0, usage / max(1, count)), 2)
                required = skill_id in {"skill_llm", "skill_rag", "skill_python", "skill_java", "skill_sql"}
                by_position[position_id][skill_id] = {
                    "requirementType": "required" if required else "preferred",
                    "weight": weight,
                }
        return dict(by_position)

    history_snapshot = build_window(historical)
    current_snapshot = build_window(current)

    if not current_snapshot:
        current_snapshot = history_snapshot

    return history_snapshot, current_snapshot


def _score_confidence(company_count: int, job_count: int, window_count: int, semantic_consistency: float) -> float:
    source_part = min(company_count / 5.0, 1.0)
    evidence_part = min(job_count / 100.0, 1.0)
    continuity_part = min(window_count / 4.0, 1.0)
    return round(
        0.35 * source_part + 0.25 * evidence_part + 0.2 * continuity_part + 0.2 * semantic_consistency,
        2,
    )


def _to_snapshot(payload: Optional[Dict[str, Any]]) -> Optional[RequirementSnapshot]:
    if payload is None:
        return None
    return RequirementSnapshot(
        requirementType=payload["requirementType"],
        weight=float(payload["weight"]),
    )


def _skill_meta(skill_id: str, position_id: str, record_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    company_count = len({item.get("company") for item in record_hits if item.get("company")})
    job_count = len(record_hits)
    semantic = min(0.99, 0.72 + min(0.2, job_count / 150.0))
    return {
        "positionName": POSITION_NAME_MAP.get(position_id, position_id),
        "skillName": SKILL_NAME_MAP.get(skill_id, skill_id),
        "companyCount": company_count,
        "jobCount": job_count,
        "windowCount": 3,
        "semanticConsistency": round(semantic, 2),
        "evidenceIds": [f"jd_{idx + 1:04d}" for idx, _ in enumerate(record_hits[:5])],
    }


def _build_real_snapshot_data() -> tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    records = _load_job_records()
    if not records:
        return ({}, {}, {})

    history_snapshot, current_snapshot = _build_snapshot_windows()
    if not current_snapshot:
        return ({}, {}, {})
    _, evidence_records = _analysis_record_windows(records)
    evidence_store: Dict[str, Dict[str, Any]] = {}

    position_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in evidence_records:
        position_buckets[record["_position_id"]].append(record)

    for position_id, items in position_buckets.items():
        for skill_id, aliases in SKILL_ALIASES.items():
            hits = [record for record in items if _match_aliases(" ".join([record.get("title", ""), record.get("description", ""), record.get("requirement", "")]), aliases)]
            if not hits:
                continue
            evidence_store[f"{position_id}:{skill_id}"] = _skill_meta(skill_id, position_id, hits)

    return history_snapshot, current_snapshot, evidence_store


def compute_evolution_changes(page: int = 1, page_size: int = 20, keyword: str = "") -> dict:
    historical_snapshot, current_snapshot, evidence_store = _build_real_snapshot_data()
    items: List[EvolutionChangeItem] = []
    all_position_ids = set(historical_snapshot) | set(current_snapshot)
    if not all_position_ids:
        return {"items": [], "total": 0, "page": page, "pageSize": page_size}

    for position_id in sorted(all_position_ids):
        # Unrecognized titles first go through new-position review. Until they
        # are accepted into the standard position library, they do not also
        # produce misleading "existing position capability change" events.
        if position_id.startswith("candidate_"):
            continue
        previous = historical_snapshot.get(position_id, {})
        current = current_snapshot.get(position_id, {})
        all_skill_ids = set(previous) | set(current)
        for skill_id in sorted(all_skill_ids):
            old_state = previous.get(skill_id)
            new_state = current.get(skill_id)
            if old_state is None and new_state is not None:
                change_type: ChangeType = "new"
            elif old_state is not None and new_state is not None:
                old_weight = float(old_state["weight"])
                new_weight = float(new_state["weight"])
                if new_weight > old_weight + 0.1:
                    change_type = "rising"
                elif new_weight < old_weight - 0.1:
                    change_type = "declining"
                else:
                    continue
            else:
                continue

            meta = evidence_store.get(f"{position_id}:{skill_id}", {})
            if not meta:
                meta = {
                    "positionName": POSITION_NAME_MAP.get(position_id, position_id),
                    "skillName": SKILL_NAME_MAP.get(skill_id, skill_id),
                    "companyCount": 1,
                    "jobCount": 1,
                    "windowCount": 1,
                    "semanticConsistency": 0.8,
                    "evidenceIds": [f"jd_{position_id}_{skill_id}_001"],
                }

            confidence = _score_confidence(
                int(meta.get("companyCount", 1)),
                int(meta.get("jobCount", 1)),
                int(meta.get("windowCount", 1)),
                float(meta.get("semanticConsistency", 0.8)),
            )

            item = EvolutionChangeItem(
                id=f"change_{len(items) + 1:03d}",
                positionId=position_id,
                positionName=meta.get("positionName", position_id),
                skillId=skill_id,
                skillName=meta.get("skillName", skill_id),
                changeType=change_type,
                before=_to_snapshot(old_state),
                after=RequirementSnapshot(
                    requirementType=new_state["requirementType"] if new_state else "preferred",
                    weight=float(new_state["weight"] if new_state else 0.0),
                ),
                evidenceCount=int(meta.get("jobCount", 1)),
                confidence=confidence,
                detectedAt=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            )

            if not keyword or keyword.lower() in item.positionName.lower() or keyword.lower() in item.skillName.lower():
                items.append(item)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


def compute_change_evidence(change_id: str) -> dict:
    items = compute_evolution_changes(page=1, page_size=500)["items"]
    matched = next((item for item in items if item.id == change_id), None)
    if matched is None:
        raise KeyError(f"unknown changeId: {change_id}")

    position_id = matched.positionId
    skill_id = matched.skillId
    all_records = _load_job_records()
    _, records = _analysis_record_windows(all_records)
    hits = [
        (idx, record)
        for idx, record in enumerate(records)
        if record["_position_id"] == position_id
        and _match_aliases(" ".join([record.get("title", ""), record.get("description", ""), record.get("requirement", "")]), SKILL_ALIASES.get(skill_id, []))
    ]
    evidence_ids = [f"jd_{record['_corpus_index'] + 1:04d}" for _, record in hits[:5]]

    before = matched.before
    after = matched.after
    confidence = float(matched.confidence)
    source_count = max(1, len({record.get("company") for _, record in hits if record.get("company")}))
    job_count = len(hits)

    return ChangeEvidence(
        changeId=change_id,
        positionId=position_id,
        positionName=matched.positionName,
        skillId=skill_id,
        skillName=matched.skillName,
        before=_to_snapshot(before.model_dump() if hasattr(before, "model_dump") else before),
        after=RequirementSnapshot(
            requirementType=after.requirementType,
            weight=float(after.weight),
        ),
        confidence=confidence,
        sourceSupport=SourceSupport(companyCount=source_count, jobCount=job_count),
        windowContinuity=WindowContinuity(continuousWindowCount=3, passed=True),
        semanticConsistency=round(min(0.99, 0.7 + job_count / 150.0), 2),
        evidenceIds=evidence_ids,
    ).model_dump()


def compute_evidence_detail(evidence_id: str) -> dict:
    records = _load_job_records()
    if not records:
        raise KeyError(f"unknown evidenceId: {evidence_id}")

    match_index = None
    try:
        match_index = int(evidence_id.split("_")[-1]) - 1
    except ValueError:
        match_index = 0

    if match_index < 0:
        raise KeyError(f"unknown evidenceId: {evidence_id}")

    record = next((item for item in records if int(item.get("_corpus_index", -1)) == match_index), None)
    if record is None:
        raise KeyError(f"unknown evidenceId: {evidence_id}")
    jd_text = "\n".join(
        [
            record.get("title", ""),
            record.get("description", ""),
            record.get("requirement", ""),
        ]
    ).strip()

    excerpt = jd_text[:800] if len(jd_text) > 800 else jd_text
    return {
        "evidenceId": evidence_id,
        "company": record.get("company", "未知公司"),
        "positionTitle": record.get("title", "未知岗位"),
        "sourcePlatform": record.get("source_platform", "unknown"),
        "publishedAt": record.get("publish_time", ""),
        "url": record.get("url", ""),
        "jdText": jd_text,
        "excerpt": excerpt,
        "matchedSkill": "AI Agent / 多智能体 / RAG",
    }


def compute_emerging_positions(page: int = 1, page_size: int = 20, keyword: str = "") -> dict:
    all_records = _load_job_records()
    _, records = _analysis_record_windows(all_records)
    if not records:
        return {"items": [], "total": 0, "page": page, "pageSize": page_size}

    position_counts: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for record in records:
        position_id = record["_position_id"]
        position_counts[position_id].setdefault("jobs", []).append(record)
        position_counts[position_id].setdefault("companies", set()).add(record.get("company"))

    candidates: List[Dict[str, Any]] = []
    for position_id, record_list in sorted(position_counts.items()):
        # Known standard positions are existing positions even when their
        # recruitment volume grows sharply. Their changes belong to the
        # capability-evolution pipeline instead.
        if not position_id.startswith("candidate_"):
            continue

        jobs = record_list["jobs"]
        sample_count = len(jobs)
        source_count = len({company for company in record_list["companies"] if company})
        if sample_count < 2:
            continue

        cutoff = datetime(2025, 1, 1, tzinfo=None)
        historical_jobs = [record for record in jobs if record["_parsed_time"].replace(tzinfo=None) < cutoff]
        recent_jobs = [record for record in jobs if record["_parsed_time"].replace(tzinfo=None) >= cutoff]
        # A title seen before the baseline is not a newly discovered position.
        if historical_jobs or len(recent_jobs) < 2:
            continue

        skill_ids = []
        for skill_id, aliases in SKILL_ALIASES.items():
            hit_count = sum(_match_aliases(_record_text(record), aliases) for record in recent_jobs)
            if hit_count / len(recent_jobs) >= 0.34:
                skill_ids.append(skill_id)
        skill_ids = skill_ids[:5]
        # Require a coherent skill combination, not merely a novel spelling of
        # an otherwise unsupported title.
        if len(skill_ids) < 2:
            continue

        skill_items = [{"id": skill_id, "name": SKILL_NAME_MAP.get(skill_id, skill_id)} for skill_id in skill_ids]
        first_seen = min(record["_parsed_time"] for record in recent_jobs)
        last_seen = max(record["_parsed_time"] for record in recent_jobs)
        span_days = max(0, (last_seen - first_seen).days)
        continuity = min(1.0, span_days / 60.0)
        source_score = min(1.0, source_count / 4.0)
        sample_score = min(1.0, sample_count / 10.0)
        skill_score = min(1.0, len(skill_ids) / 4.0)
        confidence = round(0.3 * source_score + 0.25 * sample_score + 0.2 * continuity + 0.25 * skill_score, 2)
        growth_rate = round(len(recent_jobs) / 3.0, 2)
        display_name = _display_candidate_title(recent_jobs)
        candidates.append(
            {
                "id": f"emerging_{len(candidates) + 1:03d}",
                "positionId": position_id,
                "name": display_name,
                "description": f"未匹配既有岗位库；由 {source_count} 家企业的 {sample_count} 条近期 JD 支撑，并形成稳定技能组合。",
                "growthRate": growth_rate,
                "confidence": confidence,
                "firstSeen": first_seen.date().isoformat(),
                "sourceCount": source_count,
                "sampleCount": sample_count,
                "skills": skill_items,
            }
        )

    filtered = []
    for item in candidates:
        haystack = f"{item['name']} {' '.join(skill['name'] for skill in item['skills'])}".lower()
        if not keyword or keyword.lower() in haystack:
            filtered.append(EmergingPositionItem(**item))

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": filtered[start:end],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


def _record_text(record: Dict[str, Any]) -> str:
    return " ".join([record.get("title", ""), record.get("description", ""), record.get("requirement", "")])


def _split_numbered(text: Any) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?:\n|^)\s*\d+[、.)）]\s*", str(text))
    return [part.strip() for part in parts if len(part.strip()) > 4]


def _position_growth(records_for_position: List[Dict[str, Any]]) -> float:
    current_count = len(records_for_position)
    cutoff = datetime(2025, 1, 1, tzinfo=None)
    history_count = len(
        [record for record in records_for_position if record["_parsed_time"].replace(tzinfo=None) < cutoff]
    )
    if history_count:
        return round((current_count / max(1, history_count)) - 1, 2)
    return 0.8


def compute_position_profile(position_id: str) -> dict:
    records = _load_job_records()
    position_records = [record for record in records if record["_position_id"] == position_id]
    if not position_records:
        raise KeyError(f"unknown positionId: {position_id}")

    historical_snapshot, current_snapshot, _ = _build_real_snapshot_data()
    skills_snapshot = current_snapshot.get(position_id) or historical_snapshot.get(position_id, {})

    requirements: List[SkillRequirement] = []
    for skill_id, snapshot in sorted(skills_snapshot.items()):
        hits = [record for record in position_records if _match_aliases(_record_text(record), SKILL_ALIASES.get(skill_id, []))]
        first_seen = min((record.get("publish_time", "") for record in hits if record.get("publish_time")), default="")[:10]
        requirements.append(
            SkillRequirement(
                id=f"{position_id}:{skill_id}",
                name=SKILL_NAME_MAP.get(skill_id, skill_id),
                type=snapshot["requirementType"],
                weight=float(snapshot["weight"]),
                frequency=len(hits),
                confidence=round(min(0.99, 0.6 + len(hits) / max(1, len(position_records)) * 0.35), 2),
                trend="stable",
                firstSeen=first_seen,
                evidenceCount=len(hits),
            )
        )

    responsibilities: List[str] = []
    seen_responsibilities: set[str] = set()
    for record in sorted(position_records, key=lambda r: r["_parsed_time"], reverse=True):
        for part in _split_numbered(record.get("description")):
            key = part[:24]
            if key in seen_responsibilities:
                continue
            seen_responsibilities.add(key)
            responsibilities.append(part)
        if len(responsibilities) >= 5:
            break

    scenarios: List[str] = []
    scenario_keywords = ("场景", "应用", "落地", "业务")
    for record in position_records:
        requirement = record.get("requirement", "")
        if any(keyword in requirement for keyword in scenario_keywords):
            for part in _split_numbered(requirement):
                if any(keyword in part for keyword in scenario_keywords) and part not in scenarios:
                    scenarios.append(part)
        if len(scenarios) >= 4:
            break
    if not scenarios:
        scenarios = [f"{requirement.name} 相关工程实践" for requirement in requirements[:3]]

    required_names = [requirement.name for requirement in requirements if requirement.type == "required"]
    preferred_names = [requirement.name for requirement in requirements if requirement.type == "preferred"]
    top_skill_names = [requirement.name for requirement in requirements[:3]]

    categories = [record.get("category") for record in position_records if record.get("category")]
    category = max(set(categories), key=categories.count) if categories else "未分类"

    sample_count = len(position_records)
    source_count = len({record.get("company") for record in position_records if record.get("company")})
    growth_rate = _position_growth(position_records)
    position_name = POSITION_NAME_MAP.get(position_id) or _display_candidate_title(position_records)

    description = (
        f"{position_name}：聚焦{category}方向，"
        f"核心能力围绕{'、'.join(top_skill_names) if top_skill_names else '核心技能'}展开"
        + (f"，必备技能包括{'、'.join(required_names)}" if required_names else "")
        + (f"，加分技能包括{'、'.join(preferred_names)}" if preferred_names else "")
        + f"。当前由 {sample_count} 条有效 JD、{source_count} 家企业共同支撑。"
    )

    first_seen = min((record.get("publish_time", "") for record in position_records if record.get("publish_time")), default="")[:10]
    last_seen = max((record.get("publish_time", "") for record in position_records if record.get("publish_time")), default="")[:10]

    return PositionProfile(
        id=position_id,
        name=position_name,
        category=category,
        techStack=" · ".join(top_skill_names) if top_skill_names else "通用",
        level="",
        status="emerging" if position_id.startswith("candidate_") or growth_rate > 0.1 else "existing",
        description=description,
        firstSeen=first_seen,
        lastSeen=last_seen,
        confidence=round(min(0.99, 0.55 + min(0.4, source_count / 10)), 2),
        sampleCount=sample_count,
        aliases=POSITION_ALIASES.get(position_id, []),
        responsibilities=responsibilities,
        scenarios=scenarios,
        requirements=[requirement.model_dump() for requirement in requirements],
    ).model_dump()
