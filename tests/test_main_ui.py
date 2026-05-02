import unittest

from fastapi.testclient import TestClient

from app.main import app


class TestMainUI(unittest.TestCase):
    def test_homepage_served(self) -> None:
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("本地知识库智能问答", response.text)


if __name__ == "__main__":
    unittest.main()
