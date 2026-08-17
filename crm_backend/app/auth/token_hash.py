from hashlib import sha256


def hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()
