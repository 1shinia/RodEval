"""Recover an existing active administrator from an interactive server terminal."""
import argparse
import getpass
import sqlite3
import sys
import warnings
from pathlib import Path

from werkzeug.security import generate_password_hash


def _validate_database(conn: sqlite3.Connection) -> None:
    """Reject unsupported or incomplete databases without migrating them."""
    version = conn.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
    if version != 28:
        raise ValueError('数据库 schema 必须为 28；请先使用当前版本正常启动服务完成迁移。')
    columns = {row[1]: row for row in conn.execute('PRAGMA table_info(users)')}
    if not {'id', 'username', 'password_hash', 'role', 'deleted_at', 'auth_version'} <= columns.keys():
        raise ValueError('数据库 users schema 不完整。')
    column = columns['auth_version']
    if column[2].upper() != 'INTEGER' or column[3] != 1 or column[4] != '0':
        raise ValueError('数据库 auth_version schema 不兼容。')
    conn.execute('SELECT token, user_id FROM password_reset_tokens LIMIT 0')


def _active_admin(conn: sqlite3.Connection, username: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, auth_version FROM users WHERE username = ? AND role = 'admin' AND deleted_at IS NULL",
        (username,),
    ).fetchone()
    if row is None:
        raise ValueError('指定账号不存在、已删除或不是活动管理员。')
    return row


def main(argv: list[str] | None = None) -> int:
    """Read a hidden password twice and atomically revoke the target's credentials."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--database', required=True, help='现存元数据库的绝对路径')
    parser.add_argument('--username', required=True, help='已有活动管理员用户名')
    args = parser.parse_args(argv)
    conn = None
    try:
        path = Path(args.database)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('--database 必须是现存数据库文件的绝对路径。')
        # mode=rw prevents a typo or a concurrently removed file from creating a DB.
        conn = sqlite3.connect(path.resolve().as_uri() + '?mode=rw', uri=True, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        _validate_database(conn)
        target = _active_admin(conn, args.username)
        if not sys.stdin.isatty():
            raise ValueError('必须在交互式服务器终端运行，禁止从 stdin/管道读取密码。')
        print(f'数据库：{path.resolve()}\n管理员：{args.username} (id={target["id"]})')
        # Never let getpass fall back to echoed stdin when a terminal is unavailable.
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            password = getpass.getpass('新密码（至少6个字符）：').strip()
            confirmation = getpass.getpass('再次输入新密码：').strip()
        if password != confirmation:
            raise ValueError('两次密码不一致，未修改。')
        if len(password) < 6:
            raise ValueError('新密码至少6个字符，未修改。')
        pw_hash = generate_password_hash(password)
        # Do not hold a write lock while waiting for terminal input. Revalidate
        # identity and generation under the lock to reject concurrent changes.
        conn.execute('BEGIN IMMEDIATE')
        current = _active_admin(conn, args.username)
        if tuple(current) != tuple(target):
            raise ValueError('账号在输入期间已变化，请重新运行命令。')
        conn.execute('UPDATE users SET password_hash = ?, auth_version = auth_version + 1 WHERE id = ?',
                     (pw_hash, target['id']))
        conn.execute('DELETE FROM password_reset_tokens WHERE user_id = ?', (target['id'],))
        conn.commit()
        print('管理员密码已更新；该账号的旧登录会话及密码重置链接已失效。')
        return 0
    except (KeyboardInterrupt, EOFError):
        print('\n已取消，未修改。', file=sys.stderr)
        return 1
    except (ValueError, sqlite3.Error, OSError, getpass.GetPassWarning) as exc:
        print(f'恢复失败：{exc}', file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            if conn.in_transaction:
                conn.rollback()
            conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
