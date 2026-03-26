import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

# Allow running directly from testing/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eceasy_server.app import create_app
from eceasy_server.config import get_model_name_for_provider
from eceasy_server.llm import resolve_runtime_llm_config
from eceasy_server.schemas import QueryRequest
from eceasy_server.services import stream_response


class TestServerSplit(unittest.TestCase):
    def test_create_app_routes_exist(self):
        app = create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/query", paths)
        self.assertIn("/", paths)
        self.assertIn("/frontpage", paths)

    def test_llm_invalid_provider_raises(self):
        req = QueryRequest(query="q", search_uuid="s", llm_provider="invalid")
        with self.assertRaises(HTTPException) as cm:
            resolve_runtime_llm_config(req)
        self.assertEqual(cm.exception.status_code, 400)

    def test_llm_remote_without_key_raises(self):
        req = QueryRequest(
            query="q",
            search_uuid="s",
            llm_provider="openai",
            api_key="",
            use_server_key=False,
        )
        with self.assertRaises(HTTPException) as cm:
            resolve_runtime_llm_config(req)
        self.assertEqual(cm.exception.status_code, 400)

    @patch("eceasy_server.streaming.shelve.open")
    @patch("eceasy_server.streaming.get_related_questions", return_value=["Q1?"])
    @patch("eceasy_server.streaming.search_with_duckduckgo", return_value=[])
    @patch("eceasy_server.streaming.get_rag_context", return_value=[{"snippet": "ctx", "url": "u", "name": "n"}])
    def test_stream_response_emits_all_markers_with_images(
        self,
        _mock_rag,
        _mock_web,
        _mock_related,
        mock_shelve_open,
    ):
        # Make shelve context manager safe for tests.
        mock_db = {}
        mock_shelve_open.return_value.__enter__.return_value = mock_db
        mock_shelve_open.return_value.__exit__.return_value = False

        # Fake streaming chunks from OpenAI client.
        fake_chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]),
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = iter(fake_chunks)

        def fake_image_suggester(_query, _text, _retriever):
            return [
                {
                    "source_relpath": "Sample_ELEC_Study_Pattern.png",
                    "description": "Sample ELEC Study Pattern",
                    "doc_type": "general",
                }
            ]

        out = "".join(
            stream_response(
                query="study pattern",
                search_uuid="uuid-1",
                generate_related_questions=True,
                client=fake_client,
                model_name="gpt-test",
                image_retriever=object(),
                image_suggester=fake_image_suggester,
            )
        )

        self.assertIn("__LLM_RESPONSE__", out)
        self.assertIn("__RELATED_QUESTIONS__", out)
        self.assertIn("__SUGGESTED_IMAGES__", out)
        self.assertIn("Sample_ELEC_Study_Pattern.png", out)

    @patch("eceasy_server.llm.openai.OpenAI")
    def test_llm_user_model_selection_allowed(self, mock_openai_ctor):
        mock_openai_ctor.return_value = MagicMock()
        req = QueryRequest(
            query="q",
            search_uuid="s",
            llm_provider="openai",
            api_key="test-user-key",
            use_server_key=False,
            llm_model="gpt-4o",
        )
        _client, provider, model_name, using_server_key = resolve_runtime_llm_config(req)
        self.assertEqual(provider, "openai")
        self.assertEqual(model_name, "gpt-4o")
        self.assertFalse(using_server_key)

    @patch("eceasy_server.llm.openai.OpenAI")
    def test_llm_user_model_selection_rejects_invalid_model(self, mock_openai_ctor):
        mock_openai_ctor.return_value = MagicMock()
        req = QueryRequest(
            query="q",
            search_uuid="s",
            llm_provider="openai",
            api_key="test-user-key",
            use_server_key=False,
            llm_model="gpt-invalid-model",
        )
        with self.assertRaises(HTTPException) as cm:
            resolve_runtime_llm_config(req)
        self.assertEqual(cm.exception.status_code, 400)

    @patch("eceasy_server.llm.OPENAI_API_KEY", "server-test-key")
    @patch("eceasy_server.llm.openai.OpenAI")
    def test_llm_server_key_mode_forces_env_model(self, mock_openai_ctor):
        mock_openai_ctor.return_value = MagicMock()
        req = QueryRequest(
            query="q",
            search_uuid="s",
            llm_provider="openai",
            use_server_key=True,
            llm_model="gpt-invalid-model",
        )
        _client, provider, model_name, using_server_key = resolve_runtime_llm_config(req)
        self.assertEqual(provider, "openai")
        self.assertTrue(using_server_key)
        self.assertEqual(model_name, get_model_name_for_provider("openai"))


if __name__ == "__main__":
    unittest.main()

