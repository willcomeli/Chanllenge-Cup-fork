from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.data_sources import processed_path, read_jsonl


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def test_dashboard_summary_uses_real_local_data_files() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/dashboard")

    assert response.status_code == 200
    summary = response.json()["data"]
    expected_valid = _count_jsonl(processed_path("relevant_jobs.jsonl"))
    raw_paths = [
        path
        for path in (processed_path("relevant_jobs.jsonl").parents[1] / "raw").glob("*jobs.jsonl")
        if not path.name.startswith("demo_")
    ]
    expected_source = sum(_count_jsonl(path) for path in raw_paths)
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

    class FakeJdClient:
        model = "fake-jd-model"

        def complete_json(self, _system_prompt, user_payload):
            return {
                "items": [
                    {
                        "sourceId": job["sourceId"],
                        "scope": "in_scope",
                        "position": {"id": "pos_java_engineer", "name": "Java 开发工程师"},
                        "skills": [
                            {"id": "skill_java", "name": "Java", "type": "required", "evidenceText": "精通 Java"}
                        ],
                    }
                    for job in user_payload["jobs"]
                ]
            }

    def temp_processed_path(filename: str) -> Path:
        return tmp_path / filename

    monkeypatch.setattr(pipeline_service, "DB_PATH", tmp_path / "career_prism.db")
    monkeypatch.setattr(pipeline_service, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(pipeline_service, "processed_path", temp_processed_path)
    monkeypatch.setattr(
        pipeline_service.ChatCompletionsClient,
        "from_env",
        classmethod(lambda cls, **kwargs: FakeJdClient()),
    )

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
    batch_dir = tmp_path / "batches" / created.json()["data"]["id"]
    llm_rows = read_jsonl(batch_dir / "llm_predictions.jsonl")
    assert len(llm_rows) == 1
    assert llm_rows[0]["model"] == "fake-jd-model"
    assert llm_rows[0]["positionId"] == "pos_java_engineer"
