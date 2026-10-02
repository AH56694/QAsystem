import argparse
import getpass
import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import load_settings

ROUNDS = 600000


class Accounts:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY, "
                       "salt BLOB NOT NULL, digest BLOB NOT NULL, rounds INTEGER NOT NULL, "
                       "role TEXT NOT NULL CHECK(role IN ('user','admin')))")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, name, password, role="user"):
        name = name.strip().casefold()
        if not re.fullmatch(r"[a-z0-9_]{3,32}", name):
            raise ValueError("用户名需为 3～32 位英文字母、数字或下划线")
        if not 10 <= len(password) <= 128:
            raise ValueError("密码长度需为 10～128 个字符")
        if role not in {"user", "admin"}:
            raise ValueError("未知角色")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ROUNDS)
        try:
            with closing(self.connect()) as db, db:
                db.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                           (name, salt, digest, ROUNDS, role))
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已存在") from exc

    def authenticate(self, name, password):
        if len(password) > 128:
            return None
        with closing(self.connect()) as db:
            row = db.execute("SELECT name, salt, digest, rounds, role FROM users WHERE name=?",
                             (name.strip().casefold(),)).fetchone()
        salt, expected, rounds = (row[1], row[2], row[3]) if row else (b"0" * 16, b"0" * 32, ROUNDS)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, rounds)
        if hmac.compare_digest(actual, expected) and row:
            return {"name": row[0], "role": row[4]}
        return None


def main():
    parser = argparse.ArgumentParser(description="本机创建管理员，网页注册只创建普通用户")
    parser.add_argument("name")
    args = parser.parse_args()
    password = getpass.getpass("管理员密码（输入不回显）：")
    if password != getpass.getpass("再次输入："):
        raise SystemExit("两次密码不一致")
    Accounts(load_settings().accounts).create(args.name, password, role="admin")
    print("管理员已创建")


if __name__ == "__main__":
    main()