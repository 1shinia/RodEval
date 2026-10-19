# 服务器终端恢复管理员密码

适用于已有活动管理员忘记密码，且操作员拥有部署服务器终端访问权限的情况。该命令不会创建、恢复已删除或提升普通账号。

## 操作

1. 升级至包含 schema 28 的服务版本并正常启动一次，使常规数据库迁移完成。恢复命令自身不迁移、不初始化数据库，也不运行管理员 bootstrap。
2. 确认实际部署使用的元数据库绝对路径（默认位于输出目录 `evalscope_meta.db`），按部署流程保留备份。
3. 在服务器交互终端运行，替换数据库路径和已有管理员用户名：

   ```bash
   cd /tmp
   PYTHONPATH=/data/proj/eval/evalperf /root/anaconda3/envs/hermes/bin/python \
     -m evalscope.service.reset_admin_password \
     --database /ABSOLUTE/OUTPUT_DIR/evalscope_meta.db \
     --username admin
   ```

4. 核对屏幕显示的数据库、账号和账号 ID，输入新密码两遍。输入使用 `getpass`，不回显；密码首尾空白会按现有网页登录规则去除，处理后至少 6 个字符。可用 Ctrl-C 取消。
5. 使用新密码重新登录。该账号的旧密码、旧 JWT 登录会话（含认证 Cookie）和已有密码重置链接失效。其他账号及业务记录不受该命令影响。恢复操作本身无需重启服务。

禁止通过命令行参数传递密码；本命令不提供 `--password` 参数，也不支持管道、文件或非交互 stdin 回退。无可用隐藏输入终端时直接失败。不要删除数据库或依赖 `EVALSCOPE_ADMIN_PASSWORD` 覆盖已有活动管理员。

## 失败处理与事务

数据库参数必须显式提供、为绝对路径且文件存在。错误路径不会创建数据库，旧版/不完整 schema、无目标管理员、普通用户及已删除账号均在提示密码前拒绝。两次输入不符、短密码或取消不会写入。

密码哈希更新、账号 `auth_version` 加一和删除该账号全部 reset tokens 在同一 SQLite 事务提交；任一语句失败均回滚。输入密码期间不占用写锁，提交前再次检查活动管理员身份及版本，账号发生变化则退出。数据库被占用超时会报错，可稍后重试。

服务认证对 `auth_version` 单独实时读取，不复用用户信息缓存。历史 JWT 缺少此字段时按 0 处理，因此常规 schema 升级保留原有会话，只有执行恢复的目标账号被撤销。版本检查在全局 `require_auth` 执行；`verify_token` 仅解析 JWT，不能独立作为新增路由的完整授权入口。

## 隔离回归

从 `/tmp` 运行，避免读写部署输出目录：

```bash
PYTHONPATH=/data/proj/eval/evalperf /root/anaconda3/envs/hermes/bin/pytest \
  /data/proj/eval/evalperf/tests/test_reset_admin_password.py \
  /data/proj/eval/evalperf/tests/test_password_reset.py \
  /data/proj/eval/evalperf/tests/test_login_lockout.py \
  /data/proj/eval/evalperf/tests/test_db_backup_bootstrap.py \
  /data/proj/eval/evalperf/tests/test_db_migrations.py -q
```
