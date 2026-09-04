from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from typing import Any

from backend.app.services.data_sources import processed_path, read_jsonl
from backend.app.services.evolution_service import SKILL_NAME_MAP
from backend.app.services.resume_service import get_resume_task
from src.llm_client import ChatCompletionsClient


_match_reports: dict[str, dict] = {}
_skill_alignment_cache: dict[str, tuple[str, list[dict]]] = {}
LEVEL_SCORE = {"了解": 0.55, "熟悉": 0.7, "掌握": 0.85, "精通": 1.0}
INTENT_BOOST_POINTS = 45
JAVA_ROLE_ANCHOR_POINTS = 8
POSITION_INTENT_CUES: dict[str, tuple[tuple[str, float], ...]] = {
    "pos_ai_agent_engineer": (
        ("ai agent", 1.0), ("agent", 0.75), ("智能体", 1.0), ("coding agent", 0.9),
        ("rag", 0.2), ("开发", 0.2), ("工程师", 0.1),
    ),
    "pos_llm_engineer": (
        ("大模型", 0.9), ("llm", 0.9), ("模型", 0.35), ("rag", 0.25),
        ("nlp", 0.25), ("judge", 0.35), ("评测", 0.3), ("研究", 0.25),
    ),
    "pos_java_engineer": (
        ("java", 1.0), ("后端", 0.55), ("微服务", 0.45), ("spring", 0.35),
        ("服务端", 0.35), ("开发", 0.15),
    ),
    "pos_backend_engineer": (
        ("后端", 0.45), ("服务端", 0.45), ("微服务", 0.25), ("分布式", 0.35),
        ("go", 0.25),
    ),
    "pos_algorithm_engineer": (
        ("算法", 1.0), ("研究", 0.45), ("机器学习", 0.4), ("深度学习", 0.4),
        ("视觉", 0.35), ("信号处理", 0.45), ("pytorch", 0.25),
    ),
    "pos_data_analyst": (
        ("数据分析", 1.0), ("分析师", 0.6), ("产品", 0.35), ("sql", 0.35),
        ("excel", 0.35), ("bi", 0.35), ("报表", 0.35),
    ),
    "pos_data_engineer": (
        ("数据开发", 1.0), ("数据工程", 0.9), ("数仓", 0.55), ("etl", 0.55),
        ("flink", 0.35), ("spark", 0.35),
    ),
    "pos_test_engineer": (("测试", 1.0), ("质量", 0.6), ("qa", 0.6), ("自动化测试", 0.5)),
    "pos_frontend_engineer": (("前端", 1.0), ("react", 0.5), ("typescript", 0.45), ("vue", 0.45)),
    "pos_cloud_infra_engineer": (("云", 0.7), ("基础设施", 0.8), ("infra", 0.8), ("sre", 0.65), ("运维", 0.5)),
    "pos_storage_database_engineer": (("数据库", 0.9), ("存储", 0.9), ("检索", 0.5), ("mysql", 0.35), ("redis", 0.35)),
    "pos_security_engineer": (("安全", 1.0), ("风控", 0.65), ("攻防", 0.55)),
    "pos_hardware_engineer": (("硬件", 1.0), ("芯片", 0.8), ("异构", 0.65), ("嵌入式", 0.55)),
    "pos_game_engineer": (("游戏", 1.0), ("ue", 0.6), ("unity", 0.6), ("引擎", 0.55)),
}


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _match_llm_enabled() -> bool:
    return _truthy(os.environ.get("MATCH_LLM_ENABLED") or os.environ.get("LLM_MATCH_ENABLED") or "")


def _profile_fingerprint(profile: dict) -> str:
    payload = json.dumps(profile.get("skills") or [], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _intent_text(profile: dict) -> str:
    parts = [
        profile.get("targetPosition"),
        profile.get("target_position"),
        profile.get("direction"),
        profile.get("summary"),
    ]
    return " ".join(str(part or "") for part in parts)


def _norm_intent(value: Any) -> str:
    return re.sub(r"[\s·•,，。:：;；_\-/（）()]+", "", str(value or "").casefold())


def _position_intent_score(profile: dict, position: dict) -> float:
    text = _norm_intent(_intent_text(profile))
    if not text:
        return 0.0
    cues = list(POSITION_INTENT_CUES.get(str(position.get("id") or ""), ()))
    position_name = str(position.get("name") or "")
    if position_name:
        cues.append((position_name, 1.0))
    score = 0.0
    for phrase, weight in cues:
        needle = _norm_intent(phrase)
        if needle and needle in text:
            score += weight
    return min(1.0, score / 1.2)


def _role_anchor_bonus(profile: dict, position_id: str) -> int:
    text = _norm_intent(_intent_text(profile))
    if position_id == "pos_java_engineer" and "java" in text:
        return JAVA_ROLE_ANCHOR_POINTS
    return 0


def _match_level(score: int, required_score: int, missing_required_count: int) -> str:
    if required_score >= 80 and missing_required_count == 0:
        return "高"
    if score >= 95:
        return "高"
    if score >= 50 or required_score >= 35:
        return "中"
    return "低"


def _fallback_skill_alignment(profile: dict) -> list[dict]:
    id_by_name = {name.casefold(): skill_id for skill_id, name in SKILL_NAME_MAP.items()}
    aligned = []
    for skill in profile.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        raw_id, raw_name = str(skill.get("id") or ""), str(skill.get("name") or "").strip()
        skill_id = raw_id if raw_id in SKILL_NAME_MAP else id_by_name.get(raw_name.casefold(), "")
        if skill_id:
            aligned.append({"rawName": raw_name, "standardSkillId": skill_id, "standardSkillName": SKILL_NAME_MAP[skill_id], "level": str(skill.get("level") or "熟悉"), "confidence": float(skill.get("confidence") or .7), "source": "deterministic_fallback"})
    return aligned


def _align_resume_skills(profile: dict, client: Any | None = None) -> list[dict]:
    """Use the LLM only for semantic alignment to the frozen skill vocabulary."""
    fallback = _fallback_skill_alignment(profile)
    raw_skills = [item for item in profile.get("skills") or [] if isinstance(item, dict) and item.get("name")]
    if not raw_skills:
        return []
    try:
        judge = client or ChatCompletionsClient.from_env()
        response = judge.complete_json(
            "你负责将简历技能严格对齐到给定标准技能库。只能选择技能库中的ID；语义不等价则不输出，不能编造。返回JSON：{\"alignments\":[{\"rawName\":\"\",\"standardSkillId\":\"\",\"confidence\":0到1,\"reason\":\"\"}]}。",
            {"skillCatalog": [{"id": key, "name": value} for key, value in sorted(SKILL_NAME_MAP.items())], "resumeSkills": [{"name": str(item.get("name") or ""), "level": str(item.get("level") or "熟悉"), "evidence": str(item.get("evidenceText") or item.get("source") or "")} for item in raw_skills]},
        )
        raw_by_name = {str(item.get("name") or "").strip().casefold(): item for item in raw_skills}
        aligned, seen = [], set()
        for item in response.get("alignments") or []:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("rawName") or "").strip()
            skill_id = str(item.get("standardSkillId") or "")
            source_skill = raw_by_name.get(raw_name.casefold())
            if not source_skill or skill_id not in SKILL_NAME_MAP or skill_id in seen:
                continue
            aligned.append({"rawName": raw_name, "standardSkillId": skill_id, "standardSkillName": SKILL_NAME_MAP[skill_id], "level": str(source_skill.get("level") or "熟悉"), "confidence": max(0.0, min(1.0, float(item.get("confidence") or .7))), "reason": str(item.get("reason") or ""), "source": "llm"})
            seen.add(skill_id)
        return aligned or fallback
    except Exception:
        return fallback


def _aligned_skills_for_task(resume_task_id: str, profile: dict) -> list[dict]:
    fingerprint = _profile_fingerprint(profile)
    cached = _skill_alignment_cache.get(resume_task_id)
    if cached and cached[0] == fingerprint:
        return cached[1]
    aligned = _align_resume_skills(profile) if _match_llm_enabled() else _fallback_skill_alignment(profile)
    _skill_alignment_cache[resume_task_id] = (fingerprint, aligned)
    return aligned


def _position_requirements(position_id: str) -> tuple[dict, list[dict]]:
    nodes = [node for node in read_jsonl(processed_path("graph_nodes.jsonl")) if node.get("mode") == "panorama"]
    edges = [edge for edge in read_jsonl(processed_path("graph_edges.jsonl")) if edge.get("mode") == "panorama"]
    node_map = {node["id"]: node for node in nodes}
    position = node_map.get(position_id)
    if not position or position.get("type") != "position":
        raise KeyError(f"unknown position: {position_id}")
    requirements = []
    for edge in edges:
        if edge.get("source") != position_id or edge.get("relationship") != "REQUIRES":
            continue
        skill = node_map.get(edge.get("target"))
        if skill:
            requirements.append({"id": skill["id"], "name": skill["name"], "type": edge.get("requirementType", "required"), "weight": float(edge.get("weight", skill.get("weight", 0.65)))})
    return position, requirements


def _learning_path(gaps: list[dict]) -> list[dict]:
    if not gaps:
        return [{"stage": 1, "title": "强化优势并形成作品", "duration": "1–2 周", "skills": ["项目复盘"], "goal": "将已有能力整理为可验证的项目证据"}]
    groups = [gaps[index:index + 2] for index in range(0, min(len(gaps), 6), 2)]
    titles = ["补齐高优先级基础", "完成岗位场景实践", "形成综合项目证据"]
    return [{"stage": index + 1, "title": titles[min(index, 2)], "duration": f"{index + 1}–{index + 2} 周", "skills": [gap["name"] for gap in group], "goal": f"掌握{'、'.join(gap['name'] for gap in group)}并完成可验证练习"} for index, group in enumerate(groups)]


def _resume_learning_path(profile: dict) -> list[dict]:
    path = []
    raw_items = profile.get("learningSuggestions", [])
    if not isinstance(raw_items, list):
        return path
    for index, item in enumerate(raw_items[:6]):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        skills = [
            str(skill).strip()
            for skill in item.get("skills", [])
            if str(skill).strip()
        ][:6]
        actions = [
            str(action).strip()
            for action in item.get("actions", [])
            if str(action).strip()
        ][:3]
        goal = str(item.get("expectedOutcome") or item.get("goal") or item.get("reason") or "").strip()
        if actions:
            goal = f"{goal}；行动：{'、'.join(actions)}" if goal else f"行动：{'、'.join(actions)}"
        path.append(
            {
                "stage": int(item.get("stage") or index + 1),
                "title": title,
                "duration": str(item.get("duration") or "1–2 周"),
                "skills": skills,
                "goal": goal or "完成可验证练习并补充项目证据",
            }
        )
    return path


def _llm_fit_guidance(profile: dict, report: dict, client: Any | None = None) -> dict:
    """Generate guidance only after deterministic scoring has selected a position."""
    fallback = {"summary": report["summary"], "suggestions": report["suggestions"], "learningPath": _learning_path(report["gaps"])}
    try:
        judge = client or ChatCompletionsClient.from_env()
        response = judge.complete_json(
            "你是职业发展顾问。根据结构化简历、已选岗位、确定性技能匹配结果和差距生成建议，不得修改匹配分数或虚构经历。返回JSON：{\"summary\":\"岗位适配结论\",\"suggestions\":[\"具体建议\"],\"learningPath\":[{\"stage\":1,\"title\":\"\",\"duration\":\"\",\"skills\":[\"\"],\"goal\":\"可验证成果\"}]}。学习路径2至4阶段。",
            {
                "resume": {"direction": profile.get("direction"), "summary": profile.get("summary"), "skills": profile.get("skills"), "experiences": profile.get("experiences")},
                "selectedPosition": {"id": report["positionId"], "name": report["positionName"], "score": report["overallScore"]},
                "matchedSkills": report["strengths"],
                "skillGaps": report["gaps"],
            },
        )
        suggestions = [str(item).strip() for item in response.get("suggestions") or [] if str(item).strip()][:6]
        path = []
        for index, item in enumerate(response.get("learningPath") or []):
            if not isinstance(item, dict) or not str(item.get("title") or "").strip():
                continue
            path.append({"stage": index + 1, "title": str(item["title"]).strip(), "duration": str(item.get("duration") or "1–2 周"), "skills": [str(skill).strip() for skill in item.get("skills") or [] if str(skill).strip()][:6], "goal": str(item.get("goal") or "完成可验证项目成果").strip()})
        return {"summary": str(response.get("summary") or fallback["summary"]).strip(), "suggestions": suggestions or fallback["suggestions"], "learningPath": path or fallback["learningPath"], "guidanceSource": "llm", "guidanceModel": judge.model}
    except Exception:
        return {**fallback, "guidanceSource": "deterministic_fallback", "guidanceModel": ""}


def _build_match_report(resume_task_id: str, profile: dict, position_id: str, persist: bool = True, aligned_skills: list[dict] | None = None) -> dict:
    position, requirements = _position_requirements(position_id)
    alignment = aligned_skills if aligned_skills is not None else _fallback_skill_alignment(profile)
    resume_skills = {item["standardSkillId"]: item for item in alignment}
    total_weight = sum(max(0.1, item["weight"]) for item in requirements) or 1.0
    covered_weight = required_total = required_covered = preferred_total = preferred_covered = 0.0
    missing_required_count = 0
    strengths, gaps = [], []
    for item in sorted(requirements, key=lambda value: value["weight"], reverse=True):
        weight = max(0.1, item["weight"])
        resume_skill = resume_skills.get(item["id"])
        required = item["type"] == "required"
        if required: required_total += weight
        else: preferred_total += weight
        if resume_skill:
            factor = LEVEL_SCORE.get(resume_skill.get("level", "熟悉"), 0.7)
            covered_weight += weight * factor
            strengths.append(item["name"])
            if required: required_covered += weight * factor
            else: preferred_covered += weight * factor
        else:
            missing_required_count += int(required)
            importance = round(min(100, weight * 100))
            gaps.append({"name": item["name"], "priority": "高" if required or importance >= 80 else "中", "requirement": "必备技能" if required else "加分技能", "current": "未识别", "weight": importance})
    coverage = covered_weight / total_weight if requirements else 0.0
    resume_weight = sum(LEVEL_SCORE.get(item.get("level", "熟悉"), .7) for item in resume_skills.values()) or 1.0
    relevant_weight = sum(LEVEL_SCORE.get(item.get("level", "熟悉"), .7) for skill_id, item in resume_skills.items() if any(requirement["id"] == skill_id for requirement in requirements))
    relevance = relevant_weight / resume_weight
    skill_score = round((2 * coverage * relevance / (coverage + relevance) if coverage + relevance else 0.0) * 100)
    intent_score = _position_intent_score(profile, position)
    role_anchor_bonus = _role_anchor_bonus(profile, position_id)
    score = min(100, skill_score + round(intent_score * INTENT_BOOST_POINTS) + role_anchor_bonus)
    required_score = round(required_covered / required_total * 100) if required_total else 100
    preferred_score = round(preferred_covered / preferred_total * 100) if preferred_total else 100
    experiences = profile.get("experiences", [])
    project_score = min(100, len(experiences) * 30 + len(strengths) * 8)
    skills = profile.get("skills", [])
    depth_score = round(sum(LEVEL_SCORE.get(item.get("level", "熟悉"), 0.7) for item in skills) / max(1, len(skills)) * 100)
    match_level = _match_level(score, required_score, missing_required_count)
    match_id = f"match_{uuid.uuid4().hex[:10]}"
    report = {
        "matchId": match_id, "resumeTaskId": resume_task_id, "positionId": position_id, "positionName": position["name"], "candidateName": profile.get("candidateName", "未识别姓名"),
        "overallScore": score, "matchLevel": match_level, "fitLevel": "高度匹配" if match_level == "高" else "中度匹配" if match_level == "中" else "有待提升", "benchmarkRank": "暂无排名", "benchmarkSampleCount": 0,
        "summary": f"技能库对齐后覆盖 {len(strengths)}/{len(requirements)} 项岗位技能，岗位覆盖率 {round(coverage * 100)}%，候选技能相关率 {round(relevance * 100)}%，岗位意向相似度 {round(intent_score * 100)}%。",
        "matchedSkillCount": len(strengths), "totalRequirementCount": len(requirements), "gapCount": len(gaps),
        "dimensions": [{"name": "必备技能", "value": required_score, "color": "#6ee7f9"}, {"name": "加分技能", "value": preferred_score, "color": "#a78bfa"}, {"name": "岗位意向", "value": round(intent_score * 100), "color": "#5ee7a8"}, {"name": "项目经验", "value": project_score, "color": "#5ee7a8"}, {"name": "技能深度", "value": depth_score, "color": "#fbbf73"}],
        "strengths": strengths, "gaps": gaps[:8], "skillAlignment": alignment, "evidence": {"skillEvidenceCount": len(skills), "projectEvidenceCount": len(experiences), "jobSampleCount": int(position.get("sampleCount", 0))},
        "suggestions": [f"优先补齐{gap['name']}，该能力在目标岗位中的重要度为 {gap['weight']}%" for gap in gaps[:3]] or ["技能覆盖较完整，建议补充可量化的项目成果与岗位证据"],
        "learningPath": _resume_learning_path(profile) or _learning_path(gaps),
    }
    if persist:
        _match_reports[match_id] = report
    return report


def create_match(resume_task_id: str, position_id: str) -> dict:
    task = get_resume_task(resume_task_id)
    profile = task.get("result") or {}
    report = _build_match_report(resume_task_id, profile, position_id, aligned_skills=_aligned_skills_for_task(resume_task_id, profile))
    guidance = _llm_fit_guidance(profile, report) if _match_llm_enabled() else {
        "summary": report["summary"],
        "suggestions": report["suggestions"],
        "learningPath": report["learningPath"],
        "guidanceSource": "deterministic_fallback",
        "guidanceModel": "",
    }
    report.update(guidance)
    _match_reports[report["matchId"]] = report
    return report


def rank_matches(resume_task_id: str, limit: int = 50) -> dict:
    task = get_resume_task(resume_task_id)
    profile = task.get("result") or {}
    aligned_skills = _aligned_skills_for_task(resume_task_id, profile)
    positions = [
        node for node in read_jsonl(processed_path("graph_nodes.jsonl"))
        if node.get("mode") == "panorama" and node.get("type") == "position"
    ]
    reports = [_build_match_report(resume_task_id, profile, position["id"], persist=False, aligned_skills=aligned_skills) for position in positions]
    reports.sort(
        key=lambda report: (
            -report["overallScore"],
            -next((dimension["value"] for dimension in report["dimensions"] if dimension["name"] == "必备技能"), 0),
            report["positionName"],
        )
    )
    items = [
        {
            "positionId": report["positionId"],
            "positionName": report["positionName"],
            "score": report["overallScore"],
            "fitLevel": report["fitLevel"],
            "matchedSkillCount": report["matchedSkillCount"],
            "totalSkillCount": report["totalRequirementCount"],
            "strengths": report["strengths"][:3],
            "gapCount": report["gapCount"],
        }
        for report in reports[:max(1, min(limit, 100))]
    ]
    return {
        "resumeTaskId": resume_task_id,
        "bestPositionId": items[0]["positionId"] if items else "",
        "bestPositionName": items[0]["positionName"] if items else "暂无可匹配岗位",
        "bestScore": items[0]["score"] if items else 0,
        "skillAlignment": aligned_skills,
        "items": items,
    }


def get_learning_path(match_id: str) -> dict:
    if match_id not in _match_reports:
        raise KeyError(f"unknown match: {match_id}")
    return {"matchId": match_id, "items": _match_reports[match_id]["learningPath"]}
