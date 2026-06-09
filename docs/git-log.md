# Git 日志说明

本项目已使用 Git 进行版本管理，主分支为 `main`，远程仓库为：

```text
https://github.com/zoieee424-Gyh/demo-finance.git
```

## 提交记录

```text
0a10732 docs: add knowledge base and architecture notes
51df534 feat: add Vue frontend workspace
bf6c89d feat: implement FastAPI backend and DeepAgent agents
8376c07 chore: add project docs and startup scripts
```

## 提交含义

| Commit | 类型 | 说明 |
| --- | --- | --- |
| `8376c07` | 项目初始化 | 添加 README、启动脚本、脚本目录说明和基础忽略规则。 |
| `bf6c89d` | 后端核心 | 实现 FastAPI 后端、五类 DeepAgent 智能体、RAG、MySQL 元数据、SSE 推理链、测试和本地 vendor 依赖兜底。 |
| `51df534` | 前端页面 | 添加 Vue3 + Element Plus 前端工作台，支持五智能体切换、自动路由、报告展示、证据展示和推理链展示。 |
| `0a10732` | 知识库与文档 | 添加预置知识库材料、策略配置、RAG 评估材料、架构文档和接口文档。 |

## 查看命令

```bash
git log --oneline --graph --decorate --all
```

## 当前状态

```text
main == origin/main
```

说明：本项目提交历史用于课程验收和面试展示，能够体现从项目初始化、后端智能体能力、前端交互页面到知识库与文档材料的开发过程。
