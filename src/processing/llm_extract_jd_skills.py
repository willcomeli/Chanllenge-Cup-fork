from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.app.services.data_sources import project_root, read_jsonl, write_jsonl
from backend.app.services.evolution_service import POSITION_NAME_MAP, SKILL_NAME_MAP, _position_for_record
from src.llm_client import ChatCompletionsClient, JsonChatClient
from src.processing.extract_jd_predictions import predict_jd_label, record_id


PROMPT_VERSION = "jd-extraction-v1"
PARSER_VERSION = f"llm-{PROMPT_VERSION}"
DEFAULT_INPUT = project_root() / "data" / "processed" / "splits" / "graph_train_200.jsonl"
DEFAULT_OUTPUT = project_root() / "data" / "processed" / "extractions" / "llm_jd_extraction_predictions.jsonl"
DEFAULT_TEST_OUTPUT = project_root() / "data" / "processed" / "evaluation" / "llm_jd_test_predictions.jsonl"
DEFAULT_SPLITS = {
    "graph_train": project_root() / "data" / "processed" / "splits" / "graph_train_200.jsonl",
    "jd_test": project_root() / "data" / "processed" / "splits" / "jd_test_set_100.jsonl",
    "holdout": project_root() / "data" / "processed" / "splits" / "jd_holdout_336.jsonl",
}
_CUSTOM_POSITION_NAME_MAP: dict[str, str] | None = None


def _position_name_map() -> dict[str, str]:
    return _CUSTOM_POSITION_NAME_MAP or POSITION_NAME_MAP


def _load_position_vocabulary(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = payload.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("position vocabulary must contain a non-empty positions list")
    result = {}
    for item in positions:
        if not isinstance(item, dict):
            continue
        position_id, name = str(item.get("id") or "").strip(), str(item.get("name") or "").strip()
        if position_id and name:
            result[position_id] = name
    if len(result) != len(positions):
        raise ValueError("every vocabulary position requires a unique id and name")
    return result


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _catalog_prompt() -> str:
    positions = [{"id": key, "name": value} for key, value in sorted(_position_name_map().items())]
    skills = [{"id": key, "name": value} for key, value in sorted(SKILL_NAME_MAP.items())]
    return (
        "你是岗位 JD 结构化解析助手。请只基于给定 JD 原文抽取信息，不要编造没有证据的内容。\n"
        "岗位 position.id 只能从 positions 中选择；如果无法归入已有岗位，使用 candidate_other。\n"
        "技能 skill.id 只能从 skills 中选择，最多输出 8 个最明确技能；无法归一的技能放入 newSkillCandidates。\n"
        "每个技能必须带 required 或 preferred、原文证据 evidenceText 和 0-1 confidence。\n"
        "职责 responsibilities 和场景 scenarios 必须来自原文，不要泛泛总结。\n"
        "返回严格 JSON，格式为 {\"items\":[...]}，不要 Markdown。\n"
        "item 字段包括 sourceId、scope、position、similarPositions、skills、newSkillCandidates、"
        "responsibilities、scenarios、isNewPositionCandidate、reviewReasons、confidence。\n"
        f"positions={json.dumps(positions, ensure_ascii=False)}\n"
        f"skills={json.dumps(skills, ensure_ascii=False)}"
    )


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sourceId": record_id(record),
        "sourcePlatform": record.get("source_platform") or record.get("sourcePlatform") or "",
        "sourceJobId": record.get("source_job_id") or record.get("sourceJobId") or "",
        "company": record.get("company") or "",
        "title": record.get("title") or "",
        "category": record.get("category") or "",
        "publishTime": record.get("publish_time") or record.get("publishTime") or "",
        "description": str(record.get("description") or "")[:500],
        "requirement": str(record.get("requirement") or "")[:500],
        "url": record.get("url") or "",
    }


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return default


def _text_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("name") or item.get("evidenceText") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in result:
            result.append(text[:240])
        if len(result) >= limit:
            break
    return result


def _normalize_position(raw: Any, record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raw = {}
    position_id = str(raw.get("id") or raw.get("positionId") or raw.get("expectedPositionId") or "")
    position_names = _position_name_map()
    raw_name = str(raw.get("name") or record.get("title") or "候选新岗位")
    if position_id == "candidate_other":
        return {
            "id": position_id,
            "name": raw_name,
            "confidence": _clamp_confidence(raw.get("confidence"), 0.55),
            "source": "llm",
            "evidenceText": str(raw.get("evidenceText") or record.get("title") or "")[:240],
        }
    if position_id not in position_names:
        if _CUSTOM_POSITION_NAME_MAP is None:
            position_id = _position_for_record(record)
        else:
            position_id = "candidate_other"
    is_candidate = position_id.startswith("candidate_") or position_id == "candidate_other"
    return {
        "id": position_id,
        "name": position_names.get(position_id, raw_name),
        "confidence": _clamp_confidence(raw.get("confidence"), 0.55 if is_candidate else 0.75),
        "source": "llm",
        "evidenceText": str(raw.get("evidenceText") or record.get("title") or "")[:240],
    }


def _normalize_skills(raw_skills: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_skills, list):
        return []
    skills = []
    seen: set[str] = set()
    for raw in raw_skills:
        if not isinstance(raw, dict):
            continue
        skill_id = str(raw.get("id") or raw.get("skillId") or "")
        if skill_id not in SKILL_NAME_MAP or skill_id in seen:
            continue
        requirement_type = "required" if raw.get("type") == "required" or raw.get("requirementType") == "required" else "preferred"
        skills.append(
            {
                "id": skill_id,
                "name": SKILL_NAME_MAP[skill_id],
                "type": requirement_type,
                "requirementType": requirement_type,
                "confidence": _clamp_confidence(raw.get("confidence"), 0.7),
                "source": "llm",
                "matchedAlias": str(raw.get("rawName") or raw.get("matchedAlias") or raw.get("name") or "")[:80],
                "evidenceText": str(raw.get("evidenceText") or raw.get("evidence") or "")[:300],
                "importance": _clamp_confidence(raw.get("importance"), 0.5),
            }
        )
        seen.add(skill_id)
    return sorted(skills, key=lambda item: (item["type"] != "required", -item["confidence"], item["id"]))


def _normalize_similar_positions(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, list):
        return []
    results = []
    for raw in raw_value:
        if isinstance(raw, str):
            raw = {"id": raw}
        if not isinstance(raw, dict):
            continue
        position_id = str(raw.get("id") or raw.get("positionId") or "")
        position_names = _position_name_map()
        if position_id not in position_names:
            continue
        results.append(
            {
                "id": position_id,
                "name": position_names[position_id],
                "confidence": _clamp_confidence(raw.get("confidence"), 0.5),
                "reason": str(raw.get("reason") or "")[:160],
            }
        )
        if len(results) >= 5:
            break
    return results


def normalize_llm_item(raw: dict[str, Any], record: dict[str, Any], *, split: str, generated_at: str, model: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    source_id = record_id(record)
    raw_position = raw.get("position") or {
        "id": raw.get("positionId") or raw.get("predictedPositionId") or raw.get("expectedPositionId"),
        "name": raw.get("positionName") or raw.get("predictedPositionName") or raw.get("expectedPositionName"),
        "confidence": raw.get("positionConfidence"),
        "evidenceText": raw.get("positionEvidenceText"),
    }
    position = _normalize_position(raw_position, record)
    skills = _normalize_skills(raw.get("skills"))
    rule_fallback = predict_jd_label(record, split=split, generated_at=generated_at)
    if not skills:
        skills = rule_fallback["skills"]
    if not position.get("id"):
        position = rule_fallback["position"]

    new_skill_candidates = []
    for candidate in raw.get("newSkillCandidates") or []:
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("name") or "").strip()
        evidence = str(candidate.get("evidenceText") or candidate.get("evidence") or "").strip()
        if name:
            new_skill_candidates.append(
                {
                    "name": name[:80],
                    "aliases": [str(alias)[:80] for alias in candidate.get("aliases", []) if alias][:5],
                    "suggestedCluster": str(candidate.get("suggestedCluster") or "")[:80],
                    "evidenceText": evidence[:240],
                    "confidence": _clamp_confidence(candidate.get("confidence"), 0.5),
                }
            )

    confidence = _clamp_confidence(raw.get("confidence"), position.get("confidence", 0.6))
    review_reasons = _text_list(raw.get("reviewReasons"), limit=6)
    if position["id"].startswith("candidate_") and "新岗位候选" not in review_reasons:
        review_reasons.append("新岗位候选")
    if new_skill_candidates and "存在新技能候选" not in review_reasons:
        review_reasons.append("存在新技能候选")
    if confidence < 0.7 and "LLM 置信度较低" not in review_reasons:
        review_reasons.append("LLM 置信度较低")

    return {
        "schemaVersion": "1.1",
        "sourceId": source_id,
        "evaluation_id": source_id,
        "split": split,
        "sourcePlatform": record.get("source_platform") or "",
        "sourceJobId": record.get("source_job_id") or "",
        "contentHash": record.get("content_hash") or "",
        "company": record.get("company") or "",
        "title": record.get("title") or "",
        "publishTime": record.get("publish_time") or "",
        "scrapedAt": record.get("scraped_at") or "",
        "scope": str(raw.get("scope") or ("review" if position["id"].startswith("candidate_") or position["id"] == "candidate_other" else "in_scope")),
        "position": position,
        "positionId": position["id"],
        "positionName": position["name"],
        "predictedPositionId": position["id"],
        "predictedPositionName": position["name"],
        "similarPositions": _normalize_similar_positions(raw.get("similarPositions")),
        "skills": skills,
        "predictedSkills": skills,
        "responsibilities": _text_list(raw.get("responsibilities"), limit=6),
        "scenarios": _text_list(raw.get("scenarios"), limit=6),
        "newSkillCandidates": new_skill_candidates,
        "isNewPositionCandidate": bool(raw.get("isNewPositionCandidate") or position["id"].startswith("candidate_")),
        "reviewReasons": review_reasons,
        "confidence": confidence,
        "parserVersion": PARSER_VERSION,
        "promptVersion": PROMPT_VERSION,
        "model": model,
        "generatedAt": generated_at,
    }


def extract_jd_with_llm(
    records: list[dict[str, Any]],
    client: JsonChatClient,
    *,
    split: str = "",
    batch_size: int = 5,
    generated_at: str | None = None,
    verbose: bool = False,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    timestamp = generated_at or _generated_at()
    predictions_by_id: dict[str, dict[str, Any]] = {}
    step = max(1, batch_size)
    for start in range(0, len(records), step):
        batch = records[start : start + step]
        result = client.complete_json(_catalog_prompt(), {"jobs": [_compact_record(record) for record in batch]})
        raw_items = result.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("LLM JD extraction response missing items list")
        raw_by_id = {
            str(item.get("sourceId") or item.get("source_id") or item.get("id") or ""): item
            for item in raw_items
            if isinstance(item, dict)
        }
        batch_predictions = []
        for index, record in enumerate(batch):
            source_id = record_id(record)
            raw_item = raw_by_id.get(source_id)
            if raw_item is None and index < len(raw_items) and isinstance(raw_items[index], dict):
                raw_item = raw_items[index]
            prediction = normalize_llm_item(
                raw_item or {},
                record,
                split=split,
                generated_at=timestamp,
                model=client.model,
            )
            predictions_by_id[source_id] = prediction
            batch_predictions.append(prediction)
        if on_batch is not None:
            on_batch(batch_predictions)
        if verbose:
            print(f"LLM JD extraction [{split or 'default'}]: {min(start + step, len(records))}/{len(records)}", flush=True)
    return [predictions_by_id[record_id(record)] for record in records if record_id(record) in predictions_by_id]


def extract_file_with_llm(
    input_path: Path,
    output_path: Path,
    client: JsonChatClient,
    *,
    split: str = "",
    batch_size: int = 5,
    limit: int | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    records = read_jsonl(input_path)
    if limit is not None:
        records = records[:limit]
    predictions = extract_jd_with_llm(records, client, split=split, batch_size=batch_size, verbose=verbose)
    write_jsonl(output_path, predictions)
    return predictions


def extract_default_splits_with_llm(
    output_path: Path,
    client: JsonChatClient,
    *,
    test_output_path: Path | None = DEFAULT_TEST_OUTPUT,
    batch_size: int = 5,
    limit_per_split: int | None = None,
    verbose: bool = False,
    splits: list[str] | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    timestamp = _generated_at()
    selected_splits = splits or list(DEFAULT_SPLITS)
    unknown_splits = [split for split in selected_splits if split not in DEFAULT_SPLITS]
    if unknown_splits:
        raise ValueError(f"unknown splits: {', '.join(unknown_splits)}")

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in selected_splits:
        path = DEFAULT_SPLITS[split]
        records = read_jsonl(path)
        if limit_per_split is not None:
            records = records[:limit_per_split]
        records_by_split[split] = records

    result_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if resume:
        for item in read_jsonl(output_path):
            split = str(item.get("split") or "")
            source_id = str(item.get("sourceId") or item.get("evaluation_id") or "")
            if split and source_id:
                result_by_key[(split, source_id)] = item

    def ordered_predictions() -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        for current_split in selected_splits:
            for record in records_by_split[current_split]:
                item = result_by_key.get((current_split, record_id(record)))
                if item is not None:
                    ordered.append(item)
        return ordered

    def write_checkpoint() -> None:
        ordered = ordered_predictions()
        write_jsonl(output_path, ordered)
        if test_output_path is not None and "jd_test" in selected_splits:
            write_jsonl(test_output_path, [item for item in ordered if item.get("split") == "jd_test"])

    for split in selected_splits:
        records = records_by_split[split]
        pending_records = [record for record in records if (split, record_id(record)) not in result_by_key]
        if verbose and len(pending_records) != len(records):
            print(f"LLM JD extraction [{split}]: resume skip {len(records) - len(pending_records)}/{len(records)}", flush=True)

        def checkpoint(batch_items: list[dict[str, Any]]) -> None:
            for item in batch_items:
                result_by_key[(split, str(item.get("sourceId") or item.get("evaluation_id") or ""))] = item
            write_checkpoint()

        extract_jd_with_llm(
            pending_records,
            client,
            split=split,
            batch_size=batch_size,
            generated_at=timestamp,
            verbose=verbose,
            on_batch=checkpoint,
        )
        write_checkpoint()
        if verbose:
            complete_count = sum((split, record_id(record)) in result_by_key for record in records)
            print(f"LLM JD extraction split complete: {split} ({complete_count} records)", flush=True)
    return ordered_predictions()


def main() -> None:
    global _CUSTOM_POSITION_NAME_MAP
    parser = argparse.ArgumentParser(description="Extract structured JD fields with an LLM-compatible chat API.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--split", default="graph_train")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--splits", default="", help="Comma-separated split names: graph_train,jd_test,holdout.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--position-vocabulary", type=Path, help="Optional evaluation-only position vocabulary JSON.")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--default-splits", action="store_true", help="Run graph_train, jd_test, and holdout splits.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--write-empty", action="store_true", help="Create an empty output without calling an LLM.")
    args = parser.parse_args()

    if args.position_vocabulary is not None:
        _CUSTOM_POSITION_NAME_MAP = _load_position_vocabulary(args.position_vocabulary)
        print(f"loaded position vocabulary: {len(_CUSTOM_POSITION_NAME_MAP)} positions", flush=True)

    if args.write_empty:
        write_jsonl(args.output, [])
        print(f"wrote empty LLM JD extraction output -> {args.output}")
        return

    client = ChatCompletionsClient.from_env(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        retries=args.retries,
    )
    if args.default_splits or args.input is None:
        predictions = extract_default_splits_with_llm(
            args.output,
            client,
            test_output_path=args.test_output,
            batch_size=args.batch_size,
            limit_per_split=args.limit_per_split,
            verbose=args.verbose,
            splits=[item.strip() for item in args.splits.split(",") if item.strip()] or None,
            resume=not args.no_resume,
        )
    else:
        predictions = extract_file_with_llm(
            args.input,
            args.output,
            client,
            split=args.split,
            batch_size=args.batch_size,
            limit=args.limit,
            verbose=args.verbose,
        )
    print(f"wrote LLM JD extraction predictions: {len(predictions)} records -> {args.output}")


if __name__ == "__main__":
    main()
