import uvicorn


def main() -> None:
    """启动企业知识库 RAG 问答系统 FastAPI 服务"""
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
