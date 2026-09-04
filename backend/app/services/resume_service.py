from __future__ import annotations

import re
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.app.demo_data import PANORAMA_NODES, RESUME_TASK, SKILL_REVERSE_NODES, fresh
from backend.app.services.data_sources import processed_path, read_jsonl
from backend.app.services.evolution_service import SKILL_ALIASES, SKILL_NAME_MAP
from backend.app.services.resume_llm_service import (
    analyze_resume_images_with_llm,
    analyze_resume_with_llm,
    build_rule_learning_suggestions,
)
from backend.app.services.resume_text import ResumeTextError, extract_resume_content
from src.llm_client import ChatCompletionsClient, JsonChatClient, load_llm_config


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PROFICIENCY_LEVELS = ("了解", "熟悉", "掌握", "精通")

PROFICIENCY_MARKERS = {
    "精通": ("精通", "expert", "深入", "主导", "负责核心", "专家"),
    "掌握": ("掌握", "熟练", "熟练掌握", "proficient", "advanced", "独立"),
    "熟悉": ("熟悉", "familiar", "基础", "参与"),
    "了解": ("了解", "入门", "basic", "课程", "学习"),
}

FALLBACK_SKILL_ALIASES = {
    "skill_langchain": ("LangChain", ["langchain"]),
    "skill_vector_db": ("向量数据库", ["向量数据库", "vector database", "milvus", "faiss", "pinecone"]),
    "skill_fastapi": ("FastAPI", ["fastapi"]),
    "skill_pytorch": ("PyTorch", ["pytorch"]),
    "skill_tensorflow": ("TensorFlow", ["tensorflow"]),
    "skill_react": ("React", ["react"]),
    "skill_typescript": ("TypeScript", ["typescript", "ts"]),
    "skill_spark": ("Spark", ["spark"]),
    "skill_flink": ("Flink", ["flink"]),
    "skill_linux": ("Linux", ["linux"]),
    "skill_git": ("Git", ["git"]),
    "skill_cpp": ("C++", ["c++", "cpp"]),
    "skill_vlm": ("视觉语言模型", ["视觉语言模型", "vlm", "多模态"]),
}

EDUCATION_KEYWORDS = {
    "博士": ("博士", "PhD", "Ph.D", "Doctor"),
    "硕士": ("硕士", "研究生", "Master", "M.S."),
    "本科": ("本科", "学士", "Bachelor", "B.S.", "B.E."),
    "大专": ("大专", "专科", "Associate"),
}

INTENTION_PATTERNS = (
    re.compile(r"(?:求职意向|目标岗位|意向岗位|应聘岗位)\s*[:：]?\s*([^\n，。;；]{2,40})"),
    re.compile(r"(?:Target Position|Position Desired)\s*[:：]?\s*([^\n，。;；]{2,60})", re.IGNORECASE),
)
NAME_LABEL = re.compile(r"(?:姓名|姓\s*名|Name)\s*[:：]\s*([\u4e00-\u9fff·]{2,6}|[A-Za-z][A-Za-z ]{1,29})", re.IGNORECASE)
CHINESE_NAME_LINE = re.compile(r"^[\u4e00-\u9fff·]{2,6}$")
YEARS_PATTERN = re.compile(r"(\d{1,2})\s*(?:\+|余)?\s*年(?:相关)?(?:工作|项目|开发|从业)?经验")
TIME_RANGE = re.compile(
    r"(\d{4}[./年-]\s*\d{1,2}\s*月?)\s*[-—~到至]+\s*(至今|now|present|\d{4}[./年-]\s*\d{1,2}\s*月?)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class SkillDefinition:
    id: str
    name: str
    aliases: list[str]


@dataclass(slots=True)
class ResumeTask:
    taskId: str
    filename: str
    fileSize: int
    status: str
    progress: int
    createdAt: str
    updatedAt: str
    error: str = ""
    result: dict[str, Any] | None = None
    userEdits: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "taskId": self.taskId,
            "id": self.taskId,
            "filename": self.filename,
            "fileSize": self.fileSize,
            "status": self.status,
            "progress": self.progress,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
            "error": self.error,
        }
        if self.result is not None:
            result = fresh(self.result)
            if self.userEdits.get("skills"):
                result["skills"] = fresh(self.userEdits["skills"])
            payload["result"] = result
        return payload


class ResumeTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, ResumeTask] = {}
        self._lock = threading.Lock()

    def create(
        self,
        filename: str = "",
        content: bytes = b"",
        llm_client: JsonChatClient | None = None,
        *,
        background: bool = False,
    ) -> ResumeTask:
        task_id = f"resume_{uuid.uuid4().hex[:10]}"
        now = _now()
        if not content:
            demo = fresh(RESUME_TASK)
            demo["taskId"] = task_id
            demo["id"] = task_id
            demo["filename"] = filename or "demo_resume"
            demo["fileSize"] = 0
            demo["createdAt"] = now
            demo["updatedAt"] = now
            task = _task_from_dict(demo)
            with self._lock:
                self._tasks[task_id] = task
            return task

        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("简历不能超过 10 MB")

        task = ResumeTask(
            taskId=task_id,
            filename=filename or "resume",
            fileSize=len(content),
            status="processing",
            progress=10,
            createdAt=now,
            updatedAt=now,
        )
        with self._lock:
            self._tasks[task_id] = task

        llm_enabled = llm_client is not None or _resume_llm_enabled()
        if background and not llm_enabled and self._complete_text_task_with_rules(task, content):
            return task

        if background:
            threading.Thread(
                target=self._process,
                args=(task, content, llm_client),
                daemon=True,
                name=f"resume-parser-{task_id}",
            ).start()
        else:
            self._process(task, content, llm_client, raise_errors=True)
        return task

    def _complete_text_task_with_rules(self, task: ResumeTask, content: bytes) -> bool:
        try:
            resume_content = extract_resume_content(task.filename, content, allow_vision=False)
            task.progress = 45
            task.updatedAt = _now()
            task.result = parse_resume_text(task.filename, resume_content.text)
            task.status = "completed"
            task.progress = 100
            task.updatedAt = _now()
            return True
        except Exception:
            return False

    def _process(
        self,
        task: ResumeTask,
        content: bytes,
        llm_client: JsonChatClient | None,
        *,
        raise_errors: bool = False,
    ) -> None:
        try:
            use_multimodal = llm_client is not None or _resume_llm_enabled()
            resume_content = extract_resume_content(task.filename, content, allow_vision=use_multimodal)
            task.progress = 45
            task.updatedAt = _now()
            if resume_content.mode == "vision":
                task.result = analyze_resume_images(
                    task.filename,
                    resume_content.images,
                    mime_type=resume_content.mime_type,
                    llm_client=llm_client,
                )
            else:
                task.result = analyze_resume_text(
                    task.filename,
                    resume_content.text,
                    llm_client=llm_client,
                    allow_fallback=not use_multimodal,
                )
            task.status = "completed"
            task.progress = 100
            task.updatedAt = _now()
        except Exception as exc:
            task.status = "failed"
            task.progress = 100
            task.error = str(exc)
            task.updatedAt = _now()
            if raise_errors:
                raise

    def get(self, task_id: str) -> ResumeTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def apply_full_skill_list(self, task_id: str, skills: list[dict[str, Any]]) -> ResumeTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.userEdits["skills"] = [_normalize_skill_payload(skill) for skill in skills]
            task.updatedAt = _now()
            return task

    def apply_skill_patch(
        self,
        task_id: str,
        *,
        added: list[dict[str, Any]] | None = None,
        removed: list[str] | None = None,
        updated: list[dict[str, Any]] | None = None,
    ) -> ResumeTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            current = _current_skills(task)
            if removed:
                remove_set = {
                    key
                    for item in removed
                    for key in _skill_identity_keys({"id": item, "name": item})
                }
                current = [
                    item
                    for item in current
                    if not _skill_identity_keys(item) & remove_set
                ]
            if updated:
                for change in updated:
                    keys = _skill_identity_keys(change)
                    if not keys:
                        continue
                    for item in current:
                        if _skill_identity_keys(item) & keys:
                            item.update(
                                _normalize_skill_payload(
                                    {**item, **{k: v for k, v in change.items() if k in {"id", "name", "level", "source", "confidence"}}}
                                )
                            )
                            break
            if added:
                existing = {
                    str(value).casefold()
                    for item in current
                    for value in (item.get("id"), item.get("name"))
                    if value
                }
                for skill in added:
                    normalized = _normalize_skill_payload(skill)
                    key = str(normalized.get("id") or normalized.get("name")).casefold()
                    if key and key not in existing:
                        current.append(normalized)
                        existing.add(key)
            task.userEdits["skills"] = current
            task.updatedAt = _now()
            return task


_store = ResumeTaskStore()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _task_from_dict(payload: dict[str, Any]) -> ResumeTask:
    return ResumeTask(
        taskId=payload["taskId"],
        filename=payload.get("filename", ""),
        fileSize=int(payload.get("fileSize", 0)),
        status=payload.get("status", "completed"),
        progress=int(payload.get("progress", 100)),
        createdAt=payload.get("createdAt", _now()),
        updatedAt=payload.get("updatedAt", _now()),
        error=payload.get("error", ""),
        result=payload.get("result"),
    )


def _current_skills(task: ResumeTask) -> list[dict[str, Any]]:
    if task.userEdits.get("skills"):
        return fresh(task.userEdits["skills"])
    if task.result:
        return fresh(task.result.get("skills", []))
    return []


def _normalize_skill_payload(skill: dict[str, Any]) -> dict[str, Any]:
    name = str(skill.get("name") or skill.get("id") or "自定义技能").strip()
    level = skill.get("level") if skill.get("level") in PROFICIENCY_LEVELS else "掌握"
    confidence = skill.get("confidence", 1.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0
    return {
        "id": str(skill.get("id") or f"custom_{uuid.uuid4().hex[:8]}"),
        "name": name,
        "level": level,
        "source": str(skill.get("source") or "用户修正"),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
    }


def _canonical_skill_definition(skill: dict[str, Any]) -> SkillDefinition | None:
    values = [
        str(skill.get("id") or "").strip(),
        str(skill.get("skillId") or "").strip(),
        str(skill.get("name") or "").strip(),
    ]
    lookup = {value.casefold() for value in values if value}
    if not lookup:
        return None
    for definition in _build_skill_catalog():
        candidates = {definition.id.casefold(), definition.name.casefold()}
        candidates.update(alias.casefold() for alias in definition.aliases if alias)
        if lookup & candidates:
            return definition
    return None


def _skill_identity_keys(skill: dict[str, Any]) -> set[str]:
    keys = {
        str(value).casefold()
        for value in (skill.get("id"), skill.get("skillId"), skill.get("name"))
        if value
    }
    definition = _canonical_skill_definition(skill)
    if definition:
        keys.add(definition.id.casefold())
        keys.add(definition.name.casefold())
        keys.update(alias.casefold() for alias in definition.aliases if alias)
    return keys


def _build_skill_catalog() -> list[SkillDefinition]:
    by_id: dict[str, SkillDefinition] = {}
    by_name: dict[str, str] = {}
    for skill_id, skill_name in SKILL_NAME_MAP.items():
        by_id[skill_id] = SkillDefinition(skill_id, skill_name, [skill_name, *SKILL_ALIASES.get(skill_id, [])])
        by_name[skill_name.casefold()] = skill_id

    graph_nodes = read_jsonl(processed_path("graph_nodes.jsonl")) or []
    for node in [*PANORAMA_NODES, *SKILL_REVERSE_NODES, *graph_nodes]:
        if node.get("type") != "skill" or not node.get("name"):
            continue
        name = str(node["name"])
        skill_id = by_name.get(name.casefold(), str(node.get("id") or name))
        existing = by_id.get(skill_id)
        aliases = [name, *node.get("aliases", [])]
        by_id[skill_id] = SkillDefinition(skill_id, name, sorted(set((existing.aliases if existing else []) + aliases)))
        by_name[name.casefold()] = skill_id

    for skill_id, (name, aliases) in FALLBACK_SKILL_ALIASES.items():
        existing_id = by_name.get(name.casefold())
        if existing_id:
            existing = by_id[existing_id]
            by_id[existing_id] = SkillDefinition(existing.id, existing.name, sorted(set(existing.aliases + [name, *aliases])))
        else:
            by_id[skill_id] = SkillDefinition(skill_id, name, [name, *aliases])
            by_name[name.casefold()] = skill_id
    return list(by_id.values())


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    if alias.isascii() and any(char.isalnum() for char in alias):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _find_alias(text: str, aliases: list[str]) -> tuple[str, re.Match[str]] | None:
    ordered_aliases = sorted({alias.strip() for alias in aliases if alias and alias.strip()}, key=len, reverse=True)
    for alias in ordered_aliases:
        match = _alias_pattern(alias).search(text)
        if match:
            return alias, match
    return None


def _line_context(text: str, match: re.Match[str]) -> str:
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    if end == -1:
        end = len(text)
    context = text[start:end].strip()
    if context:
        return context[:120]
    return text[max(0, match.start() - 45):match.end() + 45].strip()[:120]


def _detect_level(context: str) -> str:
    folded = context.casefold()
    for level in ("精通", "掌握", "熟悉", "了解"):
        if any(marker.casefold() in folded for marker in PROFICIENCY_MARKERS[level]):
            return level
    return "掌握"


def _skill_confidence(alias: str, context: str, level: str) -> float:
    confidence = 0.78 + min(0.14, len(alias) / 80)
    if level != "掌握":
        confidence += 0.04
    if any(marker in context for marker in ("技能", "项目", "负责", "开发", "经验")):
        confidence += 0.04
    return round(min(0.98, confidence), 3)


def _extract_skills(text: str) -> list[dict[str, Any]]:
    skills = []
    for definition in _build_skill_catalog():
        found = _find_alias(text, definition.aliases)
        if not found:
            continue
        alias, match = found
        context = _line_context(text, match)
        level = _detect_level(context)
        skills.append(
            {
                "id": definition.id,
                "name": definition.name,
                "level": level,
                "source": context or alias,
                "confidence": _skill_confidence(alias, context, level),
            }
        )
    skills.sort(key=lambda item: (-PROFICIENCY_LEVELS.index(item["level"]), -item["confidence"], item["name"]))
    return skills


def _extract_name(text: str, filename: str) -> str:
    match = NAME_LABEL.search(text)
    if match:
        return match.group(1).strip()
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if CHINESE_NAME_LINE.match(stripped) and not any(word in stripped for word in ("简历", "求职", "教育")):
            return stripped
    cleaned = re.sub(r"(?:简历|resume|求职|[_-].*)", "", filename.rsplit(".", 1)[0], flags=re.IGNORECASE).strip()
    return cleaned or "未识别姓名"


def _extract_education(text: str) -> str:
    for label, keywords in EDUCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                index = text.find(keyword)
                window = text[max(0, index - 24):index + 48]
                major = re.search(r"(计算机|软件|电子|信息|数学|自动化|通信|人工智能|数据科学)[\u4e00-\u9fffA-Za-z]*", window)
                return f"{label} · {major.group(0)}" if major else label
    return "未识别学历"


def _extract_experience_years(text: str) -> int:
    years = []
    for match in YEARS_PATTERN.finditer(text):
        try:
            years.append(int(match.group(1)))
        except ValueError:
            continue
    return max(years, default=0)


def _extract_target_position(text: str) -> str:
    for pattern in INTENTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return "待选择目标岗位"


def _extract_experiences(text: str, skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines()]
    experiences = []
    for index, line in enumerate(lines):
        match = TIME_RANGE.search(line)
        if not match:
            continue
        period = f"{match.group(1)} — {match.group(2)}"
        title = line[match.end():].strip(" -—:：|")
        cursor = index + 1
        while not title and cursor < len(lines):
            title = lines[cursor].strip()
            cursor += 1
        detail_lines = []
        while cursor < len(lines) and len(detail_lines) < 3:
            candidate = lines[cursor].strip()
            if candidate and TIME_RANGE.search(candidate):
                break
            if candidate:
                detail_lines.append(candidate)
            cursor += 1
        description = " ".join(detail_lines)
        experiences.append(_experience_payload(period, title, description, skills))
        if len(experiences) >= 5:
            break

    if not experiences:
        candidates = [line for line in lines if any(word in line for word in ("项目", "实习", "工作经历", "负责"))]
        for line in candidates[:3]:
            experiences.append(_experience_payload("简历原文", line[:30], line[:180], skills))
    return experiences


def _experience_payload(period: str, title: str, description: str, skills: list[dict[str, Any]]) -> dict[str, Any]:
    haystack = f"{title} {description}".casefold()
    tags = []
    for skill in skills:
        name = skill["name"]
        if str(name).casefold() in haystack and name not in tags:
            tags.append(name)
        if len(tags) >= 5:
            break
    return {
        "period": period,
        "title": title[:80] or "未命名经历",
        "description": description[:220],
        "detail": description[:220],
        "skills": tags,
        "tags": tags,
    }


def _direction(skills: list[dict[str, Any]], target_position: str) -> str:
    names = {skill["name"] for skill in skills}
    if names & {"大语言模型", "RAG", "LangChain", "FastAPI", "多智能体协作"} or "AI" in target_position.upper():
        return "AI 与算法方向"
    if names & {"SQL", "Spark", "Flink", "Excel / 报表"}:
        return "数据工程与分析方向"
    if names & {"React", "TypeScript", "前端工程"}:
        return "前端与产品工程方向"
    return "软件与数据工程方向"


def _completeness(profile: dict[str, Any]) -> int:
    score = 0
    if profile["candidateName"] != "未识别姓名":
        score += 15
    if profile["targetPosition"] != "待选择目标岗位":
        score += 15
    if profile["education"] != "未识别学历":
        score += 15
    if profile["experienceYears"]:
        score += 10
    score += min(20, len(profile["experiences"]) * 10)
    score += min(25, len(profile["skills"]) * 5)
    return min(100, max(35, score))


def parse_resume_text(filename: str, text: str) -> dict[str, Any]:
    if len(text.strip()) < 20:
        raise ValueError("没有从简历中提取到足够文本；扫描版 PDF 请确认 OCR 配置是否可用")

    candidate_name = _extract_name(text, filename)
    target_position = _extract_target_position(text)
    education = _extract_education(text)
    experience_years = _extract_experience_years(text)
    skills = _extract_skills(text)
    experiences = _extract_experiences(text, skills)
    direction = _direction(skills, target_position)
    profile = {
        "candidateName": candidate_name,
        "name": candidate_name,
        "targetPosition": target_position,
        "intendedPosition": target_position,
        "education": education,
        "experienceYears": experience_years,
        "direction": direction,
        "summary": f"{direction} · {experience_years} 年相关经验" if experience_years else direction,
        "skills": skills,
        "experiences": experiences,
    }
    profile["completeness"] = _completeness(profile)
    profile["learningSuggestions"] = build_rule_learning_suggestions(profile)
    profile["resumeOptimizationSuggestions"] = _rule_resume_optimization_suggestions(profile)
    profile["abilityProfile"] = _rule_ability_profile(profile)
    profile["analysisSource"] = "rule"
    profile["llmAnalysis"] = {"enabled": False, "status": "not_enabled"}
    return profile


def _rule_ability_profile(profile: dict[str, Any]) -> dict[str, Any]:
    skills = [skill.get("name") for skill in profile.get("skills", []) if isinstance(skill, dict)]
    experiences = profile.get("experiences", [])
    strengths = skills[:5]
    weaknesses = []
    if len(skills) < 4:
        weaknesses.append("技能证据偏少")
    if len(experiences) < 2:
        weaknesses.append("项目或实习经历证据不足")
    if profile.get("targetPosition") == "待选择目标岗位":
        weaknesses.append("目标岗位不明确")
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "projectEvidenceLevel": "较充分" if len(experiences) >= 2 else "待补充",
        "engineeringMaturity": "可从项目经历进一步判断" if experiences else "证据不足",
        "targetRelevance": "已识别目标岗位" if profile.get("targetPosition") != "待选择目标岗位" else "待选择目标岗位",
        "riskNotes": weaknesses,
        "summary": profile.get("summary", ""),
    }


def _rule_resume_optimization_suggestions(profile: dict[str, Any]) -> list[str]:
    suggestions = []
    if profile.get("targetPosition") == "待选择目标岗位":
        suggestions.append("补充明确的求职意向或目标岗位，方便系统做更稳定的人岗匹配。")
    if len(profile.get("experiences", [])) < 2:
        suggestions.append("补充项目经历中的个人贡献、技术难点和量化结果。")
    if len(profile.get("skills", [])) < 4:
        suggestions.append("为核心技能补充上下文证据，避免只有关键词列表。")
    return suggestions or ["简历结构较完整，建议继续补充可量化项目成果。"]


def _resume_llm_enabled() -> bool:
    return load_llm_config().resume_enabled


def analyze_resume_text(
    filename: str,
    text: str,
    *,
    llm_client: JsonChatClient | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    rule_profile = parse_resume_text(filename, text)
    if llm_client is None and not _resume_llm_enabled():
        return rule_profile

    try:
        client = llm_client or ChatCompletionsClient.from_env()
        profile = analyze_resume_with_llm(filename, text, client, fallback_profile=rule_profile)
        profile["llmAnalysis"] = {
            "enabled": True,
            "status": "completed",
            "model": client.model,
            "inputMode": "text",
            "fallbackSource": "rule",
        }
        return profile
    except Exception as exc:
        if not allow_fallback:
            raise
        rule_profile["llmAnalysis"] = {
            "enabled": True,
            "status": "degraded",
            "error": str(exc),
            "fallbackSource": "rule",
        }
        return rule_profile


def analyze_resume_images(
    filename: str,
    images: list[bytes],
    *,
    mime_type: str = "image/png",
    llm_client: JsonChatClient | None = None,
) -> dict[str, Any]:
    if llm_client is None and not _resume_llm_enabled():
        raise ResumeTextError("扫描版 PDF 需要启用支持视觉输入的简历分析模型。")
    try:
        config = load_llm_config()
        client = llm_client or ChatCompletionsClient.from_env(model=config.vision_model)
        profile = analyze_resume_images_with_llm(filename, images, client, mime_type=mime_type)
        profile["llmAnalysis"] = {
            "enabled": True,
            "status": "completed",
            "model": client.model,
            "inputMode": "vision",
            "pageCount": len(images),
            "fallbackSource": "none",
        }
        return profile
    except Exception as exc:
        raise ResumeTextError(
            f"扫描版 PDF 多模态解析失败，请确认当前模型支持图片输入：{exc}"
        ) from exc


def create_resume_task(
    filename: str = "",
    content: bytes = b"",
    *,
    llm_client: JsonChatClient | None = None,
    background: bool = False,
) -> dict:
    try:
        task = _store.create(filename, content, llm_client=llm_client, background=background)
    except ResumeTextError as exc:
        raise ValueError(str(exc)) from exc
    return {"taskId": task.taskId, "status": task.status, "progress": task.progress}


def get_resume_task(task_id: str) -> dict:
    task = _store.get(task_id)
    if task is not None:
        return task.as_dict()
    if task_id == "demo_resume_task":
        return fresh(RESUME_TASK)
    raise KeyError(f"unknown resume task: {task_id}")


def update_resume_skills(task_id: str, skills: list[dict[str, Any]]) -> dict:
    task = _store.apply_full_skill_list(task_id, skills)
    if task is None and task_id == "demo_resume_task":
        demo = _store.create()
        demo.taskId = task_id
        _store._tasks[task_id] = demo
        task = _store.apply_full_skill_list(task_id, skills)
    if task is None:
        raise KeyError(f"unknown resume task: {task_id}")
    return {"taskId": task.taskId, "skills": fresh(_current_skills(task))}


def patch_resume_skills(
    task_id: str,
    *,
    added: list[dict[str, Any]] | None = None,
    removed: list[str] | None = None,
    updated: list[dict[str, Any]] | None = None,
) -> dict:
    task = _store.apply_skill_patch(task_id, added=added, removed=removed, updated=updated)
    if task is None:
        raise KeyError(f"unknown resume task: {task_id}")
    return {"taskId": task.taskId, "skills": fresh(_current_skills(task)), "result": deepcopy(task.as_dict().get("result", {}))}


def get_resume_task_store() -> ResumeTaskStore:
    return _store
