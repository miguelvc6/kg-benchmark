import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guardian.model_provider import OpenAIChatProvider


class OpenAIChatProviderTests(unittest.TestCase):
    def test_loads_api_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env").write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                previous_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    provider = OpenAIChatProvider()
                finally:
                    os.chdir(previous_cwd)

            self.assertEqual(provider.api_key, "test-key")
            self.assertEqual(provider.model, "test-model")
            self.assertEqual(provider.base_url, "https://api.openai.com/v1")


if __name__ == "__main__":
    unittest.main()
