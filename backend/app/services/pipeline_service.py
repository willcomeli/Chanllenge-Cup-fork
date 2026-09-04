from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.services.data_sources import processed_path, read_jsonl, write_jsonl
from backend.app.services.evolution_service import (
    POSITION_NAME_MAP,
    SKILL_ALIASES,
    SKILL_NAME_MAP,
    _match_aliases,
    _normalize_position_title,
    _position_for_record,
    _record_text,
)
from src.llm_client import ChatCompletionsClient, load_llm_config
from src.processing.clean_multisource_jobs import classify_records, read_jsonl as read_raw_jsonl


DB_PATH = processed_path("career_prism.db")
BATCH_ROOT = processed_path("batches")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS batches (
          id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
          input_count INTEGER NOT NULL DEFAULT 0, valid_count INTEGER NOT NULL DEFAULT 0,
          rejected_count INTEGER NOT NULL DEFAULT 0, new_position_count INTEGER NOT NULL DEFAULT 0,
          change_count INTEGER NOT NULL DEFAULT 0, no_change_count INTEGER NOT NULL DEFAULT 0,
          pending_review_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, applied_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pipeline_reviews (
          id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, type TEXT NOT NULL, target_id TEXT NOT NULL,
          title TEXT NOT NULL, description TEXT NOT NULL, confidence REAL NOT NULL,
          sources_json TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
          note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, reviewed_at TEXT, applied_at TEXT
        );
        CREATE TABLE IF NOT EXISTS processed_sources (
          source_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, content_hash TEXT NOT NULL, processed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS position_registry (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, normalized_name TEXT UNIQUE NOT NULL,
          aliases_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL
        );
        """
    )
    return connection


def _seed_registry(connection: sqlite3.Connection) -> None:
    for position_id, name in POSITION_NAME_MAP.items():
        connection.execute(
            "INSERT OR IGNORE INTO position_registry(id,name,normalized_name,aliases_json,created_at) VALUES(?,?,?,?,?)",
            (position_id, name, _normalize_position_title(name), "[]", _now()),
        )
    connection.commit()


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    mapping = {
        "input_count": "inputCount", "valid_count": "validCount", "rejected_count": "rejectedCount",
        "new_position_count": "newPositionCount", "change_count": "changeCount",
        "no_change_count": "noChangeCount", "pending_review_count": "pendingReviewCount",
        "created_at": "createdAt", "applied_at": "appliedAt",
    }
    return {mapping.get(key, key): value for key, value in item.items()}


def list_batches(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_dict(row) for row in rows]


def get_batch(batch_id: str) -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown batch: {batch_id}")
    return _row_dict(row)


def _graph_position_skills() -> dict[str, set[str]]:
    nodes = [node for node in read_jsonl(processed_path("graph_nodes.jsonl")) if node.get("mode") == "panorama"]
    edges = [edge for edge in read_jsonl(processed_path("graph_edges.jsonl")) if edge.get("mode") == "panorama"]
    node_map = {node["id"]: node for node in nodes}
    result: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source, target = node_map.get(edge.get("source")), node_map.get(edge.get("target"))
        if source and target and source.get("type") == "position" and target.get("type") == "skill" and edge.get("relationship") == "REQUIRES":
            result[_normalize_position_title(source["name"])].add(target["name"])
    return result


def _record_skills(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skills = []
    for skill_id, aliases in SKILL_ALIASES.items():
        hits = sum(_match_aliases(_record_text(record), aliases) for record in records)
        if hits:
            skills.append({"id": skill_id, "name": SKILL_NAME_MAP.get(skill_id, skill_id), "frequency": hits, "weight": round(hits / len(records), 2)})
    return sorted(skills, key=lambda item: (-item["frequency"], item["name"]))


def _source_id(record: dict[str, Any]) -> str:
    return str(record.get("source_id") or f"{record.get('source_platform', '')}:{record.get('source_job_id', '')}")


def _append_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    existing = {str(item.get("sourceId") or item.get("evaluation_id") or ""): item for item in read_jsonl(path)}
    for item in predictions:
        source_id = str(item.get("sourceId") or item.get("evaluation_id") or "")
        if source_id:
            existing[source_id] = item
    write_jsonl(path, list(existing.values()))


def _extract_batch_with_llm(records: list[dict[str, Any]], batch_dir: Path) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    if not load_llm_config().jd_enabled:
        return {}

    from src.processing.llm_extract_jd_skills import extract_jd_with_llm

    checkpoint_path = batch_dir / "llm_jd_extraction_predictions.jsonl"
    client = ChatCompletionsClient.from_env()

    def checkpoint(batch_items: list[dict[str, Any]]) -> None:
        _append_predictions(checkpoint_path, batch_items)

    predictions = extract_jd_with_llm(
        records,
        client,
        split="batch_upload",
        batch_size=5,
        on_batch=checkpoint,
    )
    write_jsonl(checkpoint_path, predictions)
    _append_predictions(processed_path("extractions/llm_jd_extraction_predictions.jsonl"), predictions)
    return {str(item.get("sourceId") or item.get("evaluation_id") or ""): item for item in predictions}


def _prediction_position(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> tuple[str | None, str]:
    if not prediction:
        matched = _registry_match(connection, record["title"])
        return matched if matched else (None, record["title"])

    position = prediction.get("position") if isinstance(prediction.get("position"), dict) else {}
    position_id = str(position.get("id") or prediction.get("positionId") or prediction.get("predictedPositionId") or "")
    position_name = str(position.get("name") or prediction.get("positionName") or prediction.get("predictedPositionName") or record["title"])
    if position_id and position_id not in {"candidate_other"} and not position_id.startswith("candidate_"):
        row = connection.execute("SELECT id,name FROM position_registry WHERE id=? AND status='active'", (position_id,)).fetchone()
        if row is not None:
            return row["id"], row["name"]
        if position_id in POSITION_NAME_MAP:
            return position_id, POSITION_NAME_MAP[position_id]
    return None, position_name or record["title"]


def _skills_for_records(
    records: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not predictions_by_id:
        return _record_skills(records)

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        prediction = predictions_by_id.get(_source_id(record)) or {}
        for skill in prediction.get("skills") or []:
            if not isinstance(skill, dict):
                continue
            skill_id = str(skill.get("id") or "")
            if skill_id not in SKILL_NAME_MAP:
                continue
            item = grouped.setdefault(
                skill_id,
                {
                    "id": skill_id,
                    "name": SKILL_NAME_MAP[skill_id],
                    "frequency": 0,
                    "_confidence_total": 0.0,
                },
            )
            item["frequency"] += 1
            item["_confidence_total"] += float(skill.get("confidence") or 0.7)

    if not grouped:
        return _record_skills(records)
    result = []
    for item in grouped.values():
        frequency = int(item["frequency"])
        result.append(
            {
                "id": item["id"],
                "name": item["name"],
                "frequency": frequency,
                "weight": round(frequency / max(1, len(records)), 2),
                "confidence": round(float(item["_confidence_total"]) / max(1, frequency), 3),
                "source": "llm",
            }
        )
    return sorted(result, key=lambda item: (-item["frequency"], item["name"]))


def _update_unchanged_position_stats(updates: list[tuple[str, list[dict[str, Any]]]]) -> None:
    if not updates:
        return
    path = processed_path("graph_nodes.jsonl")
    nodes = read_jsonl(path)
    for position_name, records in updates:
        latest = max((record.get("publish_time", "") for record in records), default="")[:10]
        normalized = _normalize_position_title(position_name)
        for node in nodes:
            if node.get("type") != "position" or _normalize_position_title(node.get("name")) != normalized:
                continue
            node["sampleCount"] = int(node.get("sampleCount", 0)) + len(records)
            if latest:
                node["lastSeen"] = max(str(node.get("lastSeen", "")), latest)
    write_jsonl(path, nodes)


def _registry_match(connection: sqlite3.Connection, title: str) -> tuple[str, str] | None:
    normalized = _normalize_position_title(title)
    rows = connection.execute("SELECT id,name,normalized_name,aliases_json FROM position_registry WHERE status='active'").fetchall()
    for row in rows:
        aliases = json.loads(row["aliases_json"])
        if normalized == row["normalized_name"] or normalized in {_normalize_position_title(alias) for alias in aliases}:
            return row["id"], row["name"]
    fallback = _position_for_record({"title": title})
    if not fallback.startswith("candidate_"):
        return fallback, POSITION_NAME_MAP.get(fallback, title)
    return None


def process_batch(filename: str, content: bytes) -> dict[str, Any]:
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    batch_dir = BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    raw_path = batch_dir / "raw.jsonl"
    raw_path.write_bytes(content)
    raw_records, parse_rejections = read_raw_jsonl(raw_path)
    relevant, review_boundary, rejected, report = classify_records(raw_records)
    rejected_count = len(parse_rejections) + len(rejected) + len(review_boundary)
    predictions_by_id = _extract_batch_with_llm(relevant, batch_dir)
    write_jsonl(batch_dir / "relevant_jobs.jsonl", relevant)
    write_jsonl(batch_dir / "rejected_jobs.jsonl", parse_rejections + rejected)

    with _connect() as connection:
        _seed_registry(connection)
        connection.execute(
            "INSERT INTO batches(id,filename,status,input_count,valid_count,rejected_count,created_at) VALUES(?,?,?,?,?,?,?)",
            (batch_id, filename, "classified", len(raw_records) + len(parse_rejections), len(relevant), rejected_count, _now()),
        )
        unseen = []
        for record in relevant:
            if connection.execute("SELECT 1 FROM processed_sources WHERE source_id=?", (record["source_id"],)).fetchone() is None:
                unseen.append(record)

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        group_meta: dict[str, tuple[str | None, str]] = {}
        for record in unseen:
            prediction = predictions_by_id.get(_source_id(record))
            position_id, position_name = _prediction_position(connection, record, prediction)
            if position_id:
                key = f"existing:{position_id}"
                group_meta[key] = (position_id, position_name)
            else:
                normalized = _normalize_position_title(position_name)
                key = f"candidate:{normalized}"
                group_meta[key] = (None, position_name)
            groups[key].append(record)

        graph_skills = _graph_position_skills()
        new_count = change_count = no_change_count = 0
        review_rows = []
        no_change_rows = []
        unchanged_updates: list[tuple[str, list[dict[str, Any]]]] = []
        for key, records in groups.items():
            position_id, position_name = group_meta[key]
            skills = _skills_for_records(records, predictions_by_id)
            companies = sorted({record.get("company", "未知公司") for record in records})
            evidence_ids = [record["source_id"] for record in records]
            if position_id is None:
                if len(records) >= 3 and len(companies) >= 2 and len(skills) >= 2:
                    new_count += 1
                    target_id = f"candidate_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:12]}"
                    payload = {"positionName": position_name, "normalizedTitle": key.split(":", 1)[1], "skills": skills, "companies": companies, "evidenceIds": evidence_ids, "sampleCount": len(records)}
                    review_rows.append((f"review_{batch_id}_new_{new_count}", "新岗位", target_id, position_name, f"未匹配标准岗位库；由 {len(companies)} 家企业、{len(records)} 条 JD 和 {len(skills)} 项稳定技能支撑。", min(0.96, 0.55 + len(companies) * 0.08 + len(records) * 0.03), companies, payload))
                else:
                    no_change_count += len(records)
                    no_change_rows.extend(records)
                continue

            old_skills = graph_skills.get(_normalize_position_title(position_name), set())
            current_skills = {skill["name"] for skill in skills}
            added = sorted(current_skills - old_skills)
            if added:
                change_count += 1
                payload = {"positionId": position_id, "positionName": position_name, "addedSkills": [skill for skill in skills if skill["name"] in added], "evidenceIds": evidence_ids, "sampleCount": len(records)}
                review_rows.append((f"review_{batch_id}_change_{change_count}", "能力变更", position_id, f"{position_name}新增能力", f"检测到新增技能：{'、'.join(added)}。", min(0.95, 0.65 + len(records) * 0.04), companies, payload))
            else:
                no_change_count += len(records)
                no_change_rows.extend(records)
                unchanged_updates.append((position_name, records))

        for review_id, review_type, target_id, title, description, confidence, sources, payload in review_rows:
            connection.execute(
                "INSERT INTO pipeline_reviews(id,batch_id,type,target_id,title,description,confidence,sources_json,payload_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,'pending',?)",
                (review_id, batch_id, review_type, target_id, title, description, confidence, json.dumps(sources, ensure_ascii=False), json.dumps(payload, ensure_ascii=False), _now()),
            )
        for record in unseen:
            connection.execute("INSERT OR IGNORE INTO processed_sources(source_id,batch_id,content_hash,processed_at) VALUES(?,?,?,?)", (record["source_id"], batch_id, record["content_hash"], _now()))
        connection.execute("UPDATE batches SET status=?,new_position_count=?,change_count=?,no_change_count=?,pending_review_count=?,applied_at=? WHERE id=?", ("reviewing" if review_rows else "applied", new_count, change_count, no_change_count, len(review_rows), None if review_rows else _now(), batch_id))
        connection.commit()

    # Evidence is appended idempotently to the global processed corpus.
    global_jobs = {item.get("source_id"): item for item in read_jsonl(processed_path("relevant_jobs.jsonl"))}
    for record in unseen:
        global_jobs[record["source_id"]] = record
    write_jsonl(processed_path("relevant_jobs.jsonl"), list(global_jobs.values()))
    write_jsonl(batch_dir / "no_change_jobs.jsonl", no_change_rows)
    _update_unchanged_position_stats(unchanged_updates)
    return get_batch(batch_id)


def pipeline_reviews(status: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM pipeline_reviews"
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status=?"
        params = ("applied" if status == "approved" else status,)
    query += " ORDER BY created_at DESC"
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [{"id": row["id"], "type": row["type"], "title": row["title"], "description": row["description"], "confidence": row["confidence"], "sources": json.loads(row["sources_json"]), "createdAt": row["created_at"], "status": "approved" if row["status"] == "applied" else row["status"], "targetId": row["target_id"], "note": row["note"], "batchId": row["batch_id"]} for row in rows]


def _apply_payload(review_type: str, target_id: str, payload: dict[str, Any]) -> None:
    from src.processing.build_graph_seed import merge_graph_data
    generated_at = _now()
    nodes, edges = [], []
    if review_type == "新岗位":
        position_id = f"pos_{target_id.removeprefix('candidate_')}"
        name = payload["positionName"]
        nodes += [
            {"mode": "panorama", "id": position_id, "name": name, "type": "position", "trend": "new", "sampleCount": payload["sampleCount"], "confidence": 0.9, "generatedAt": generated_at},
            {"mode": "panorama", "id": "approved_cluster_emerging", "name": "新兴技术岗位簇", "type": "cluster", "generatedAt": generated_at},
            {"mode": "skill_reverse", "id": f"reverse_{position_id}", "name": name, "type": "position", "trend": "new", "sampleCount": payload["sampleCount"], "confidence": 0.9, "generatedAt": generated_at},
            {"mode": "skill_reverse", "id": "approved_skill_cluster_emerging", "name": "新兴岗位核心技能簇", "type": "cluster", "generatedAt": generated_at},
        ]
        edges += [{"mode": "panorama", "source": position_id, "target": "approved_cluster_emerging", "relationship": "BELONGS_TO"}]
        skills = payload["skills"]
    else:
        name = payload["positionName"]
        existing_positions = [node for node in read_jsonl(processed_path("graph_nodes.jsonl")) if node.get("mode") == "panorama" and node.get("type") == "position"]
        matched_node = next((node for node in existing_positions if _normalize_position_title(node.get("name")) == _normalize_position_title(name)), None)
        position_id = matched_node["id"] if matched_node else payload["positionId"]
        reverse_positions = [node for node in read_jsonl(processed_path("graph_nodes.jsonl")) if node.get("mode") == "skill_reverse" and node.get("type") == "position"]
        reverse_matched = next((node for node in reverse_positions if _normalize_position_title(node.get("name")) == _normalize_position_title(name)), None)
        reverse_position_id = reverse_matched["id"] if reverse_matched else f"reverse_{position_id}"
        nodes += [
            {"mode": "skill_reverse", "id": "approved_skill_cluster_emerging", "name": "新兴岗位核心技能簇", "type": "cluster", "generatedAt": generated_at},
        ]
        skills = payload["addedSkills"]

    for skill in skills:
        skill_id = f"pipeline_{skill['id']}"
        reverse_skill_id = f"reverse_{skill_id}"
        common = {"name": skill["name"], "type": "skill", "weight": skill["weight"], "sampleCount": skill["frequency"], "confidence": 0.88, "generatedAt": generated_at}
        nodes += [{"mode": "panorama", "id": skill_id, **common}, {"mode": "skill_reverse", "id": reverse_skill_id, **common}]
        edges.append({"mode": "panorama", "source": position_id, "target": skill_id, "relationship": "REQUIRES", "requirementType": "required" if skill["weight"] >= 0.6 else "preferred", "weight": skill["weight"], "confidence": 0.88})
        if review_type == "新岗位":
            edges += [{"mode": "skill_reverse", "source": reverse_skill_id, "target": "approved_skill_cluster_emerging", "relationship": "BELONGS_TO"}, {"mode": "skill_reverse", "source": f"reverse_{position_id}", "target": reverse_skill_id, "relationship": "REQUIRES", "requirementType": "required" if skill["weight"] >= 0.6 else "preferred", "weight": skill["weight"], "confidence": 0.88}]
        else:
            edges += [{"mode": "skill_reverse", "source": reverse_skill_id, "target": "approved_skill_cluster_emerging", "relationship": "BELONGS_TO"}, {"mode": "skill_reverse", "source": reverse_position_id, "target": reverse_skill_id, "relationship": "REQUIRES", "requirementType": "required" if skill["weight"] >= 0.6 else "preferred", "weight": skill["weight"], "confidence": 0.88}]

    node_path, edge_path = processed_path("graph_nodes.jsonl"), processed_path("graph_edges.jsonl")
    merged_nodes, merged_edges = merge_graph_data(read_jsonl(node_path), read_jsonl(edge_path), nodes, edges)
    write_jsonl(node_path, merged_nodes)
    write_jsonl(edge_path, merged_edges)


def decide_pipeline_review(review_id: str, status: str, note: str = "") -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM pipeline_reviews WHERE id=?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown review: {review_id}")
        if row["status"] in {"applied", "rejected"}:
            return {"id": review_id, "status": "approved" if row["status"] == "applied" else row["status"], "note": row["note"], "graphUpdated": row["status"] == "applied"}
        if status == "rejected":
            connection.execute("UPDATE pipeline_reviews SET status='rejected',note=?,reviewed_at=? WHERE id=?", (note, _now(), review_id))
            final_status, graph_updated = "rejected", False
        else:
            payload = json.loads(row["payload_json"])
            _apply_payload(row["type"], row["target_id"], payload)
            applied_at = _now()
            connection.execute("UPDATE pipeline_reviews SET status='applied',note=?,reviewed_at=?,applied_at=? WHERE id=?", (note, applied_at, applied_at, review_id))
            if row["type"] == "新岗位":
                position_id = f"pos_{row['target_id'].removeprefix('candidate_')}"
                connection.execute("INSERT OR IGNORE INTO position_registry(id,name,normalized_name,aliases_json,created_at) VALUES(?,?,?,?,?)", (position_id, payload["positionName"], payload["normalizedTitle"], json.dumps([payload["positionName"]], ensure_ascii=False), applied_at))
            final_status, graph_updated = "applied", True
        pending = connection.execute("SELECT COUNT(*) FROM pipeline_reviews WHERE batch_id=? AND status='pending'", (row["batch_id"],)).fetchone()[0]
        connection.execute("UPDATE batches SET pending_review_count=?,status=?,applied_at=? WHERE id=?", (pending, "applied" if pending == 0 else "reviewing", _now() if pending == 0 else None, row["batch_id"]))
        connection.commit()
    return {"id": review_id, "status": "approved" if final_status == "applied" else final_status, "note": note, "graphUpdated": graph_updated}
