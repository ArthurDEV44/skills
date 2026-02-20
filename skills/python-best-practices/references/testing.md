# Testing with pytest

## Test Structure and Naming

```python
# tests/test_user.py
import pytest
from myapp.models import User, UserNotFoundError

class TestUserCreation:
    """Group related tests in classes (no __init__ needed)."""

    def test_creates_user_with_valid_email(self) -> None:
        user = User(name="Alice", email="alice@example.com")
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_rejects_invalid_email(self) -> None:
        with pytest.raises(ValueError, match="Invalid email"):
            User(name="Alice", email="not-an-email")

    def test_normalizes_email_to_lowercase(self) -> None:
        user = User(name="Alice", email="Alice@Example.COM")
        assert user.email == "alice@example.com"
```

**Naming conventions:**
- Test files: `test_<module>.py`
- Test functions: `test_<what_it_does>` -- descriptive, not `test_1`, `test_user`
- Test classes: `Test<Subject>` -- group related tests

## Fixtures

### Basic Fixtures

```python
@pytest.fixture
def sample_user() -> User:
    return User(name="Alice", email="alice@example.com")

@pytest.fixture
def admin_user() -> User:
    return User(name="Admin", email="admin@example.com", role="admin")

def test_user_display_name(sample_user: User) -> None:
    assert sample_user.display_name == "Alice"
```

### Fixture with Teardown (yield)

```python
@pytest.fixture
def db_connection() -> Iterator[Connection]:
    conn = create_connection("sqlite:///:memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    yield conn
    conn.close()

@pytest.fixture
def temp_directory() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)
```

### Fixture Scopes

```python
@pytest.fixture(scope="session")
def database() -> Iterator[Database]:
    """Created once for the entire test session."""
    db = Database.create()
    yield db
    db.drop()

@pytest.fixture(scope="module")
def api_client() -> ApiClient:
    """Created once per test module."""
    return ApiClient(base_url="http://test")

@pytest.fixture  # scope="function" is the default
def clean_state(database: Database) -> Iterator[None]:
    """Reset state before each test."""
    yield
    database.clear_all()
```

### conftest.py for Shared Fixtures

```python
# tests/conftest.py -- automatically discovered by pytest
import pytest

@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applied to every test automatically."""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

@pytest.fixture
def mock_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external API calls."""
    def fake_fetch(url: str) -> dict:
        return {"status": "ok", "data": []}
    monkeypatch.setattr("myapp.api.fetch", fake_fetch)
```

## Parametrize

### Basic Parametrization

```python
@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
    ("123", "123"),
])
def test_uppercase(input_val: str, expected: str) -> None:
    assert input_val.upper() == expected
```

### Parametrize with IDs and Marks

```python
@pytest.mark.parametrize("email,valid", [
    pytest.param("user@example.com", True, id="valid-email"),
    pytest.param("user@sub.example.com", True, id="subdomain"),
    pytest.param("no-at-sign", False, id="missing-at"),
    pytest.param("", False, id="empty"),
    pytest.param("a@b", False, id="no-tld", marks=pytest.mark.xfail(reason="TLD validation pending")),
])
def test_email_validation(email: str, valid: bool) -> None:
    assert is_valid_email(email) == valid
```

### Stacked Parametrize (Cartesian Product)

```python
@pytest.mark.parametrize("method", ["GET", "POST", "PUT"])
@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
def test_request_handling(method: str, content_type: str) -> None:
    """Runs 6 combinations (3 methods x 2 content types)."""
    response = make_request(method=method, content_type=content_type)
    assert response.status_code == 200
```

## Mocking with monkeypatch

```python
def test_fetch_user_handles_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(url: str) -> Response:
        return Response(status_code=404, body=b"")

    monkeypatch.setattr("myapp.client.http_get", mock_get)

    with pytest.raises(UserNotFoundError):
        fetch_user("nonexistent-id")

def test_uses_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "test-key-123")
    client = create_client()
    assert client.api_key == "test-key-123"
```

### unittest.mock for Complex Mocking

```python
from unittest.mock import AsyncMock, MagicMock, patch, call

def test_sends_notification() -> None:
    sender = MagicMock()
    service = NotificationService(sender=sender)

    service.notify("user-1", "Hello!")

    sender.send.assert_called_once_with("user-1", "Hello!")

# Async mock
async def test_async_fetch() -> None:
    mock_fetch = AsyncMock(return_value={"id": 1, "name": "Alice"})
    with patch("myapp.client.fetch", mock_fetch):
        result = await get_user(1)
        assert result.name == "Alice"

# patch as decorator
@patch("myapp.email.send_email")
def test_registration_sends_email(mock_send: MagicMock) -> None:
    register_user("alice@example.com")
    mock_send.assert_called_once_with(
        to="alice@example.com",
        subject="Welcome!",
    )
```

## Exception Testing

```python
def test_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"must be positive"):
        create_item(quantity=-1)

def test_raises_with_attributes() -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate({"name": ""})
    assert exc_info.value.field == "name"
    assert "required" in str(exc_info.value)
```

## Temporary Files and Directories

```python
def test_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    generate_report(output)

    data = json.loads(output.read_text())
    assert data["status"] == "complete"
    assert len(data["items"]) == 5

def test_reads_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[server]\nport = 8080\n')

    config = load_config(config_file)
    assert config.server.port == 8080
```

## Markers

```python
# Register custom markers in pyproject.toml
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow",
#     "integration: marks integration tests",
# ]

@pytest.mark.slow
def test_full_pipeline() -> None:
    """Run with: pytest -m slow"""
    ...

@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_unix_permissions() -> None:
    ...

@pytest.mark.xfail(reason="Known bug #123")
def test_edge_case() -> None:
    ...
```

## Async Testing

```python
import pytest

# With pytest-asyncio
@pytest.mark.asyncio
async def test_async_fetch() -> None:
    result = await fetch_data("https://api.example.com")
    assert result["status"] == "ok"

# Async fixture
@pytest.fixture
async def async_client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient() as client:
        yield client

@pytest.mark.asyncio
async def test_with_client(async_client: AsyncClient) -> None:
    response = await async_client.get("/health")
    assert response.status_code == 200
```

## pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = [
    "--strict-markers",        # error on unknown markers
    "--strict-config",         # error on config issues
    "-ra",                     # show summary of all non-passing tests
    "--tb=short",              # shorter tracebacks
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]
filterwarnings = [
    "error",                   # treat warnings as errors
    "ignore::DeprecationWarning:third_party_lib.*",
]
```

## Testing Best Practices

- **Arrange-Act-Assert** -- clear structure for each test
- **One assertion per concept** -- test one behavior, may have multiple asserts for that behavior
- **Descriptive names** -- `test_rejects_expired_token` not `test_token_3`
- **Don't test implementation details** -- test behavior, not internal state
- **Use factories** -- helper functions to create test objects with sensible defaults
- **Prefer monkeypatch over mock** -- simpler, less coupling to implementation
- **Test edge cases** -- empty inputs, None, boundary values, Unicode, large inputs
- **Avoid test interdependence** -- each test should pass in isolation
