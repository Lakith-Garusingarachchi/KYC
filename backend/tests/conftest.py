import os
import sys
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet

# Settings are read once, so the test environment must be in place before `app` is imported.
_TMP = Path(tempfile.mkdtemp(prefix="kyc-tests-"))
os.environ.update({
    "DATABASE_URL": f"sqlite+aiosqlite:///{_TMP / 'test.db'}",
    "UPLOAD_DIR": str(_TMP / "uploads"),
    "FILE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "AUTH_SECRET_KEY": "test-secret-key-test-secret-key-0123456789",
    "LLM_PROVIDER": "gemini",
    "GEMINI_API_KEY": "test-key",
    "GOOGLE_CLIENT_ID": "",
    "GOOGLE_CLIENT_SECRET": "",
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fakeredis  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.database import init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.redis_client import set_redis  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest_asyncio.fixture
async def client():
    set_redis(fakeredis.FakeAsyncRedis(decode_responses=True))
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def models_available():
    if not (MODELS_DIR / "face_recognition_sface_2021dec.onnx").is_file():
        pytest.skip("face models not downloaded")
