from __future__ import annotations

import pytest
import time
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ocr import CallableOcrProvider, NullOcrProvider, OcrError, _extract_text_from_response, set_ocr_provider
from backend.app.services.resume_service import analyze_resume_text, create_resume_task, get_resume_task, parse_resume_text
from backend.app.services.resume_text import ResumeTextError, extract_resume_content, extract_resume_text


SAMPLE_RESUME = """姓名：陈小雨
求职意向：AI Agent 研发工程师
硕士 · 计算机科学
3 年相关工作经验

专业技能
- 精通 Python，熟练掌握 FastAPI
- 掌握 大语言模型 与 RAG 检索链路
- 熟悉 LangChain、向量数据库、Docker

项目经历
2025.03 - 至今  企业知识库智能问答系统
负责 RAG 链路、向量检索与模型服务化，离线评测准确率提升 18%。

2024.06 - 2025.01  多轮对话助手
参与提示词工程、会话状态管理及工具调用模块开发。
"""


def test_parse_resume_text_links_skills_to_graph_catalog() -> None:
    profile = parse_resume_text("陈小雨_简历.txt", SAMPLE_RESUME)

    assert profile["candidateName"] == "陈小雨"
    assert profile["targetPosition"] == "AI Agent 研发工程师"
    assert profile["experienceYears"] == 3
    assert profile["completeness"] >= 80

    skills = {skill["name"]: skill for skill in profile["skills"]}
    assert {"Python", "RAG", "FastAPI", "LangChain", "云原生"} <= set(skills)
    assert skills["Python"]["level"] == "精通"
    assert skills["云原生"]["id"] == "skill_cloud_native"
    assert profile["experiences"]
    assert profile["analysisSource"] == "rule"
    assert profile["learningSuggestions"]
    assert profile["resumeOptimizationSuggestions"]


class FakeResumeLlmClient:
    model = "fake-resume-llm"

    def complete_json(self, system_prompt: str, user_payload: dict) -> dict:
        assert "简历分析与学习建议助手" in system_prompt
        assert "resumeText" in user_payload
        return {
            "profile": {
                "candidateName": "陈小雨",
                "targetPosition": "AI Agent 研发工程师",
                "education": "硕士 · 计算机科学",
                "experienceYears": 3,
                "direction": "AI 应用工程方向",
                "summary": "具备 RAG 和智能体应用项目经验",
                "skills": [
                    {
                        "id": "skill_python",
                        "name": "Python",
                        "level": "精通",
                        "evidenceText": "精通 Python",
                        "confidence": 0.92,
                    },
                    {
                        "id": "skill_rag",
                        "name": "RAG",
                        "level": "掌握",
                        "evidenceText": "负责 RAG 链路",
                        "confidence": 0.9,
                    },
                ],
                "experiences": [
                    {
                        "period": "2025.03 - 至今",
                        "title": "企业知识库智能问答系统",
                        "description": "负责 RAG 链路、向量检索与模型服务化",
                        "skills": ["RAG", "Python"],
                        "confidence": 0.88,
                    }
                ],
                "abilityProfile": {
                    "strengths": ["RAG 工程落地"],
                    "weaknesses": ["多智能体协作证据不足"],
                    "projectEvidenceLevel": "较充分",
                    "engineeringMaturity": "具备独立模块交付经验",
                    "targetRelevance": "较高",
                },
                "learningSuggestions": [
                    {
                        "stage": 1,
                        "category": "短期补齐",
                        "priority": "高",
                        "title": "补齐多智能体协作证据",
                        "reason": "目标岗位需要 Agent 编排能力",
                        "duration": "1-2 周",
                        "skills": ["多智能体协作"],
                        "actions": ["完成工具调用工作流 Demo"],
                        "expectedOutcome": "形成可演示项目证据",
                    }
                ],
                "resumeOptimizationSuggestions": ["补充智能体项目的量化指标"],
                "confidence": 0.87,
            }
        }


class FakeVisionResumeLlmClient(FakeResumeLlmClient):
    def complete_json_with_images(
        self,
        system_prompt: str,
        user_payload: dict,
        images: list[bytes],
        *,
        mime_type: str = "image/png",
    ) -> dict:
        assert "页面图片" in system_prompt
        assert user_payload["inputMode"] == "vision"
        assert user_payload["pageCount"] == len(images) == 1
        assert mime_type == "image/png"
        return self.complete_json(system_prompt, {"resumeText": "vision"})


def test_analyze_resume_text_uses_llm_client_and_returns_learning_suggestions() -> None:
    profile = analyze_resume_text("陈小雨_简历.txt", SAMPLE_RESUME, llm_client=FakeResumeLlmClient())

    assert profile["analysisSource"] == "llm"
    assert profile["llmAnalysis"]["status"] == "completed"
    assert profile["skills"][0]["id"] == "skill_python"
    assert profile["abilityProfile"]["strengths"] == ["RAG 工程落地"]
    assert profile["learningSuggestions"][0]["title"] == "补齐多智能体协作证据"


def test_create_resume_task_accepts_injected_llm_client() -> None:
    created = create_resume_task(
        filename="resume.txt",
        content=SAMPLE_RESUME.encode("utf-8"),
        llm_client=FakeResumeLlmClient(),
    )
    task = get_resume_task(created["taskId"])

    assert task["result"]["analysisSource"] == "llm"
    assert task["result"]["llmAnalysis"]["model"] == "fake-resume-llm"


def test_background_resume_task_uses_llm_when_enabled() -> None:
    created = create_resume_task(
        filename="resume.txt",
        content=SAMPLE_RESUME.encode("utf-8"),
        llm_client=FakeResumeLlmClient(),
        background=True,
    )
    deadline = time.time() + 2
    task = get_resume_task(created["taskId"])
    while task["status"] == "processing" and time.time() < deadline:
        time.sleep(0.01)
        task = get_resume_task(created["taskId"])

    assert task["status"] == "completed"
    assert task["result"]["analysisSource"] == "llm"
    assert task["result"]["llmAnalysis"]["status"] == "completed"


def test_resume_task_lifecycle_supports_upload_and_delta_skill_patch() -> None:
    client = TestClient(app)

    created = client.post(
        "/api/v1/resume-tasks",
        files={"file": ("resume.txt", SAMPLE_RESUME.encode("utf-8"), "text/plain")},
    )
    assert created.status_code == 200
    task_id = created.json()["data"]["taskId"]

    fetched = client.get(f"/api/v1/resume-tasks/{task_id}")
    assert fetched.status_code == 200
    assert fetched.json()["data"]["result"]["candidateName"] == "陈小雨"

    patched = client.patch(
        f"/api/v1/resume-tasks/{task_id}/skills",
        json={
            "added": [{"name": "Kubernetes", "level": "熟悉", "source": "用户补充"}],
            "removed": ["skill_fastapi"],
            "updated": [{"id": "skill_python", "level": "掌握", "source": "人工修正"}],
        },
    )
    assert patched.status_code == 200
    edited = patched.json()["data"]["skills"]
    names = {skill["name"] for skill in edited}
    assert "Kubernetes" in names
    assert "FastAPI" not in names
    python = next(skill for skill in edited if skill["id"] == "skill_python")
    assert python["level"] == "掌握"
    assert python["source"] == "人工修正"


def test_extract_text_from_plain_bytes_supports_chinese_encodings() -> None:
    text = extract_resume_text("resume.txt", "姓名：陈小雨\n熟悉 SQL".encode("gb18030"))

    assert "陈小雨" in text
    assert "SQL" in text


def test_ocr_response_extractor_handles_common_json_shapes() -> None:
    assert _extract_text_from_response({"text": "hi"}) == "hi"
    assert _extract_text_from_response({"data": {"text": "hi"}}) == "hi"
    assert _extract_text_from_response({"result": [{"text": "a"}, {"text": "b"}]}) == "a\nb"
    assert _extract_text_from_response({"nope": 1}) == ""


def _build_scanned_pdf() -> bytes:
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(50, 50, 545, 792), color=(0.9, 0.9, 0.9), fill=(1, 1, 1))
    data = document.tobytes()
    document.close()
    return data


def test_scanned_pdf_routes_through_configured_ocr_provider() -> None:
    calls: list[str] = []

    def fake_ocr(image: bytes, mime: str) -> str:
        calls.append(mime)
        return "姓名：陈小雨\n精通 Python 与 RAG 检索链路"

    set_ocr_provider(CallableOcrProvider(fake_ocr, name="fake"))
    try:
        text = extract_resume_text("scanned.pdf", _build_scanned_pdf())
    finally:
        set_ocr_provider(None)

    assert "陈小雨" in text
    assert calls == ["image/png"]


def test_scanned_pdf_without_ocr_provider_reports_clear_error() -> None:
    set_ocr_provider(NullOcrProvider())
    try:
        with pytest.raises(ResumeTextError, match="OCR"):
            extract_resume_text("scanned.pdf", _build_scanned_pdf())
    finally:
        set_ocr_provider(None)


def test_scanned_pdf_uses_multimodal_model_without_ocr() -> None:
    set_ocr_provider(NullOcrProvider())
    try:
        prepared = extract_resume_content("scanned.pdf", _build_scanned_pdf(), allow_vision=True)
        created = create_resume_task(
            filename="scanned.pdf",
            content=_build_scanned_pdf(),
            llm_client=FakeVisionResumeLlmClient(),
        )
    finally:
        set_ocr_provider(None)

    task = get_resume_task(created["taskId"])
    assert prepared.mode == "vision"
    assert len(prepared.images) == 1
    assert task["result"]["analysisSource"] == "llm"
    assert task["result"]["llmAnalysis"]["inputMode"] == "vision"
    assert task["result"]["llmAnalysis"]["pageCount"] == 1


def test_null_ocr_provider_raises_on_direct_use() -> None:
    provider = NullOcrProvider()

    assert provider.available is False
    with pytest.raises(OcrError):
        provider.recognize(b"image")
