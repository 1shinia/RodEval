#!/bin/bash
# 重启 evalperf 后端服务(9002) + 前端 Vite(5173)
# 用法: bash /data/proj/eval/restart_services.sh
set -u
APP_DIR=/data/proj/eval/evalperf
HERMES_PY=/root/anaconda3/envs/hermes/bin/python
LOG_DIR=$APP_DIR/outputs
# 后端必须带 CONDA_NO_PLUGINS=1 与 PYTHONPATH=.，否则 conda 插件报错 / 找不到本地 evalscope
export CONDA_NO_PLUGINS=1
export PYTHONPATH=.

# ── 停旧进程 ─────────────────────────────────────────────
echo "== stopping old processes =="
# 后端
pkill -f "evalscope.service.app --port 9002" 2>/dev/null && echo "  backend: old process signalled" || echo "  backend: none running"
# 前端 vite (按端口匹配, 避免误杀其他 vite)
for pid in $(pgrep -f "vite --host 0.0.0.0 --port 5173"); do
  kill "$pid" 2>/dev/null && echo "  vite: killed $pid"
done
sleep 2

# ── 起新进程 (setsid 脱离会话, 日志追加) ─────────────────
echo "== starting backend on :9002 =="
cd "$APP_DIR" || exit 1
setsid nohup "$HERMES_PY" -m evalscope.service.app --port 9002 \
  >> "$LOG_DIR/evalscope_service.log" 2>&1 < /dev/null &
echo "  backend pid: $!"

echo "== starting frontend on :5173 =="
cd "$APP_DIR/evalscope/web" || exit 1
setsid nohup npx vite --host 0.0.0.0 --port 5173 \
  >> "$LOG_DIR/vite.log" 2>&1 < /dev/null &
echo "  vite pid: $!"

# ── 就绪探活 ─────────────────────────────────────────────
echo "== waiting for readiness =="
code=000
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9002/ 2>/dev/null)
  [ "$code" != "000" ] && break
  sleep 1
done
echo "backend : HTTP $code (401 = 正常, 需登录)"
code_fe=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5173/ 2>/dev/null)
echo "frontend: HTTP $code_fe"
echo "== 日志: $LOG_DIR/evalscope_service.log / $LOG_DIR/vite.log =="
