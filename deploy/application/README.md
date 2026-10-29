# 应用镜像交付

面向 Linux x86_64，Python 3.12。前端静态文件与 Waitress 后端同端口提供。
不含历史数据库、任务、密钥、模型权重、CUDA、本地推理引擎。
保留 CPU PyTorch 等 MTEB 评测依赖；源代码中的本地模型入口暂未移除，不能视为已禁用的产品能力。

构建：在 hermes 环境执行 `python deploy/application/prepare.py`，然后
`docker build -t evalperf/application:delivery-20261023 /data/evalperf-build/context`。
requirements.lock 从当前已安装依赖递归锁定；missing_in_hermes 记录原环境也未安装的声明项。

部署：导入交付的镜像归档 `docker load -i application-image.tar`。
复制 compose.yaml，在旁边创建 .env，填写 EVALSCOPE_ADMIN_PASSWORD（独立强密码）。
默认 APP_PORT=9000、BIND_ADDRESS=127.0.0.1。需外部访问时配置 BIND_ADDRESS，推荐经过 HTTPS 反向代理。
执行 `docker compose up -d`，访问配置的地址。初始用户名 admin。
outputs 保存数据库、随机登录签名密钥、任务和日志，必须持久化并备份。
CACHE_DIR 默认 /root/.cache；恢复数据集后管理员需扫描、确认并发布。
缓存挂载为可写，以支持管理员准备数据及缓存锁；普通用户仍由服务策略强制离线。

本应用镜像包含 Docker 客户端与 Python Docker SDK，但默认部署没有赋予 Docker 管理权限。
代码沙箱还需要宿主机 Docker socket、独立沙箱镜像，以及宿主机与应用容器一致的工作目录挂载。
沙箱镜像属于下一份独立交付。仅挂载 socket 不能保证所有代码基准都能执行，尤其宿主路径和回连地址必须单独验收。
不要默认使用 privileged，也不要开启历史全局容器清理。

升级前备份 outputs（运行时数据库应使用系统备份功能或停机备份），固定镜像版本。
数据库迁移后回滚须同时考虑数据库兼容性。
