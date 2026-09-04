import unittest
from datetime import datetime
from unittest.mock import patch

from backend.app.services.evolution_service import (
    _position_for_record,
    compute_evolution_changes,
    compute_evidence_detail,
    compute_emerging_positions,
)


class EvolutionServiceTest(unittest.TestCase):
    def test_packaged_baseline_falls_back_to_demo_evolution_changes(self):
        payload = compute_evolution_changes(page=1, page_size=20)

        self.assertGreater(payload["total"], 0)
        self.assertGreater(len(payload["items"]), 0)

    def test_returns_evidence_detail_for_real_jd_record(self):
        payload = compute_evidence_detail("jd_0001")

        self.assertIn("evidenceId", payload)
        self.assertIn("company", payload)
        self.assertIn("positionTitle", payload)
        self.assertIn("jdText", payload)

    def test_returns_emerging_positions_without_datetime_type_error(self):
        payload = compute_emerging_positions(page=1, page_size=10)

        self.assertIn("items", payload)
        self.assertIn("total", payload)
        self.assertIn("page", payload)
        self.assertIn("pageSize", payload)

    def test_known_position_growth_is_not_a_new_position(self):
        records = [
            {
                "title": "Java 后端工程师",
                "company": f"公司{index % 3}",
                "description": "Java Spring 微服务开发",
                "requirement": "熟悉 Java、Spring 和 Docker",
                "publish_time": f"2026-0{index + 1}-01T00:00:00",
                "_parsed_time": datetime(2026, index + 1, 1),
                "_position_id": "pos_java_engineer",
            }
            for index in range(5)
        ]
        with patch("backend.app.services.evolution_service._load_job_records", return_value=records), patch(
            "backend.app.services.evolution_service._baseline_record_count", return_value=None
        ):
            payload = compute_emerging_positions()
        self.assertEqual(payload["total"], 0)

    def test_unknown_recent_multisource_title_is_a_new_position_candidate(self):
        records = [
            {
                "title": "具身智能数据工程师",
                "company": f"公司{index}",
                "description": "负责机器人数据和多模态大模型训练",
                "requirement": "熟悉 Python、大模型、Docker 与数据处理",
                "publish_time": f"2026-0{index * 2 + 1}-01T00:00:00",
                "_parsed_time": datetime(2026, index * 2 + 1, 1),
                "_position_id": "candidate_embodied_data",
            }
            for index in range(3)
        ]
        with patch("backend.app.services.evolution_service._load_job_records", return_value=records), patch(
            "backend.app.services.evolution_service._baseline_record_count", return_value=None
        ):
            payload = compute_emerging_positions()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["items"][0].name, "具身智能数据工程师")

    def test_unmatched_title_is_not_folded_into_agent_position(self):
        position_id = _position_for_record(
            {
                "title": "量子计算系统工程师",
                "description": "使用 Python 和大语言模型构建研发工具",
                "requirement": "熟悉 LLM 与 RAG",
            }
        )
        self.assertTrue(position_id.startswith("candidate_"))

    def test_common_technical_titles_are_normalized_to_standard_positions(self):
        cases = {
            "推荐算法工程师-国际电商": "pos_algorithm_engineer",
            "后端研发工程师": "pos_backend_engineer",
            "测试开发工程师": "pos_test_engineer",
            "云计算 SRE 工程师": "pos_cloud_infra_engineer",
            "硬件芯片工程师": "pos_hardware_engineer",
        }

        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(_position_for_record({"title": title}), expected)


if __name__ == "__main__":
    unittest.main()
