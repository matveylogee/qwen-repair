import os
import re
import secrets
import stat
from pathlib import Path


def validate_token(token: str) -> str:
    if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token) is None:
        raise ValueError("HTTP executor token must contain 32 to 256 URL-safe ASCII characters")
    return token


def load_or_create_token(path: Path) -> str:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            token = secrets.token_urlsafe(32)
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write((token + "\n").encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            return token
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600):
                raise ValueError("HTTP executor token file must be an owned regular file with mode 0600")
            raw = stream.read(258)
        return validate_token(raw.removesuffix(b"\n").decode("ascii"))
    except (OSError, UnicodeError):
        raise ValueError("Cannot read or create a secure HTTP executor token file") from None
