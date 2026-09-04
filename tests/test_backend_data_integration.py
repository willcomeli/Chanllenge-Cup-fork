from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.data_sources import processed_path, read_jsonl, write_jsonl


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _expected_source_count() -> int:
    raw_paths = [
        path
        for path in (processed_path("relevant_jobs.jsonl").parents[1] / "raw").glob("*jobs.jsonl")
        if not path.name.startswith("demo_")
    ]
    if raw_paths:
        return sum(_count_jsonl(path) for path in raw_paths)
    cleaning_report = json.loads(processed_path("cleaning_report.json").read_text(encoding="utf-8"))
    if "input_records" in cleaning_report:
        return int(cleaning_report["input_records"])
    return int(cleaning_report["inputRecords"])


def test_dashboard_summary_uses_real_local_data_files() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/dashboard")

    assert response.status_code == 200
    summary = response.json()["data"]
    expected_valid = _count_jsonl(processed_path("relevant_jobs.jsonl"))
    expected_source = _expected_source_count()
    expected_changes = client.get("/api/v1/evolution/changes?page=1&page_size=1").json()["data"]["total"]
    split_report = json.loads(processed_path("splits/split_report.json").read_text(encoding="utf-8"))

    assert summary["validCount"] == expected_valid
    assert summary["sourceCount"] == expected_source
    assert summary["graphTrainCount"] == split_report["graphTrainCount"]
    assert summary["jdTestCount"] == split_report["jdTestCount"]
    assert summary["holdoutCount"] == split_report["holdoutCount"]
    assert summary["metrics"][0]["sampleCount"] == split_report["jdTestCount"]
    assert summary["changedCount"] == expected_changes
    assert summary["metrics"][0]["source"] in {"backend_demo_metric", "deepseek_gold_evaluation"}
    assert all(metric["source"] == "backend_demo_metric" for metric in summary["metrics"][1:])


def test_evaluation_summary_counts_current_pending_reviews() -> None:
    client = TestClient(app)

    pending = client.get("/api/v1/reviews?status=pending").json()["data"]
    evaluation = client.get("/api/v1/evaluations/summary").json()["data"]

    assert evaluation["pendingReviewCount"] == len(pending)
    assert evaluation["highPriorityReviewCount"] == sum(1 for item in pending if item["confidence"] >= 0.9)


def test_graph_endpoint_uses_processed_jsonl_graph_data() -> None:
    client = TestClient(app)
    nodes = read_jsonl(processed_path("graph_nodes.jsonl"))
    edges = read_jsonl(processed_path("graph_edges.jsonl"))

    panorama = client.get("/api/v1/graph?mode=panorama").json()["data"]
    reverse = client.get("/api/v1/graph?mode=skill_reverse").json()["data"]

    assert len(panorama["nodes"]) == sum(1 for node in nodes if node.get("mode") == "panorama")
    assert len(reverse["nodes"]) == sum(1 for node in nodes if node.get("mode") == "skill_reverse")
    assert len(panorama["edges"]) == sum(1 for edge in edges if edge.get("mode") == "panorama")
    assert len(reverse["edges"]) == sum(1 for edge in edges if edge.get("mode") == "skill_reverse")


def test_jd_batch_upload_is_visible_in_existing_batch_list(tmp_path, monkeypatch) -> None:
    import backend.app.services.pipeline_service as pipeline_service

    def temp_processed_path(filename: str) -> Path:
        return tmp_path / filename

    monkeypatch.setattr(pipeline_service, "DB_PATH", tmp_path / "career_prism.db")
    monkeypatch.setattr(pipeline_service, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(pipeline_service, "processed_path", temp_processed_path)

    record = {
        "source_platform": "test_jobs",
        "company": "测试公司",
        "recruit_type": "社会招聘",
        "source_job_id": "demo-001",
        "job_id": "demo-001",
        "title": "Java 后端开发工程师",
        "locations": "上海",
        "employment_type": "全职",
        "category": "技术",
        "publish_time": "2026-07-08",
        "description": "负责 Java 后端与云原生平台研发",
        "requirement": "精通 Java，熟悉 Spring Boot、Docker、Kubernetes",
        "url": "https://example.com/jobs/demo-001",
        "scraped_at": "2026-08-01 10:00:00+08:00",
    }
    content = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    client = TestClient(app)

    created = client.post("/api/v1/jd-batches", files={"file": ("batch.jsonl", content, "application/jsonl")})
    listed = client.get("/api/v1/jd-batches")

    assert created.status_code == 200
    assert listed.status_code == 200
    assert any(batch["id"] == created.json()["data"]["id"] for batch in listed.json()["data"])


def test_jd_batch_upload_uses_llm_extraction_and_approval_dedupes_generated_review(tmp_path, monkeypatch) -> None:
    import backend.app.services.pipeline_service as pipeline_service
    import backend.app.services.review_service as review_service

    class FakeJdClient:
        model = "fake-jd-llm"

        def complete_json(self, system_prompt: str, user_payload: dict) -> dict:
            assert "岗位 JD 结构化解析助手" in system_prompt
            return {
                "items": [
                    {
                        "sourceId": item["sourceId"],
                        "scope": "review",
                        "position": {
                            "id": "candidate_other",
                            "name": "认知数字孪生编排工程师",
                            "evidenceText": item["title"],
                            "confidence": 0.91,
                        },
                        "skills": [
                            {"id": "skill_python", "requirementType": "required", "confidence": 0.92, "evidenceText": "Python"},
                            {"id": "skill_rag", "requirementType": "required", "confidence": 0.9, "evidenceText": "RAG"},
                        ],
                        "newSkillCandidates": [],
                        "responsibilities": ["负责认知数字孪生平台研发"],
                        "scenarios": ["认知数字孪生"],
                        "isNewPositionCandidate": True,
                        "reviewReasons": ["新岗位候选"],
                        "confidence": 0.9,
                    }
                    for item in user_payload["jobs"]
                ]
            }

    def temp_processed_path(filename: str) -> Path:
        return tmp_path / filename

    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_JD_ENABLED", "true")
    monkeypatch.setattr(pipeline_service, "DB_PATH", tmp_path / "career_prism.db")
    monkeypatch.setattr(pipeline_service, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(pipeline_service, "processed_path", temp_processed_path)
    monkeypatch.setattr(review_service, "processed_path", temp_processed_path)
    monkeypatch.setattr(pipeline_service.ChatCompletionsClient, "from_env", lambda: FakeJdClient())

    write_jsonl(temp_processed_path("graph_nodes.jsonl"), [])
    write_jsonl(temp_processed_path("graph_edges.jsonl"), [])

    records = []
    for index, company in enumerate(["甲公司", "乙公司", "丙公司"], start=1):
        records.append(
            {
                "source_platform": "test_jobs",
                "company": company,
                "recruit_type": "校园招聘",
                "source_job_id": f"llm-{index}",
                "job_id": f"llm-{index}",
                "title": "Java 后端开发工程师",
                "locations": "长春",
                "employment_type": "全职",
                "category": "技术",
                "publish_time": "2026-09-03",
                "description": "负责 Java 后端系统，也包含 Python、RAG、认知数字孪生平台研发。",
                "requirement": "熟悉 Python、RAG、Java、Spring Boot。",
                "url": f"https://example.com/jobs/llm-{index}",
                "scraped_at": "2026-09-03 10:00:00+08:00",
            }
        )
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records).encode("utf-8")

    batch = pipeline_service.process_batch("llm-batch.jsonl", content)
    duplicate_generated_review = {
        "id": "review_position_candidate_duplicate",
        "type": "新岗位",
        "title": "认知数字孪生编排工程师",
        "description": "动态计算出的同名候选岗位",
        "confidence": 0.8,
        "sources": ["动态演化服务"],
        "createdAt": "2026-09-03",
        "status": "pending",
        "targetId": "candidate_duplicate",
        "note": "",
    }
    monkeypatch.setattr(review_service, "_build_emerging_reviews", lambda: [duplicate_generated_review])
    monkeypatch.setattr(review_service, "_build_change_reviews", lambda: [])
    pending = review_service.get_reviews(status="pending", review_type="新岗位")

    assert batch["newPositionCount"] == 1
    assert batch["changeCount"] == 0
    assert len(read_jsonl(tmp_path / "batches" / batch["id"] / "llm_jd_extraction_predictions.jsonl")) == 3
    assert [item["title"] for item in pending] == ["认知数字孪生编排工程师"]

    decision = review_service.decide_review(pending[0]["id"], "approved")
    remaining = review_service.get_reviews(status="pending", review_type="新岗位")

    assert decision["graphUpdated"] is True
    assert remaining == []
    assert any(node.get("name") == "认知数字孪生编排工程师" for node in read_jsonl(temp_processed_path("graph_nodes.jsonl")))
