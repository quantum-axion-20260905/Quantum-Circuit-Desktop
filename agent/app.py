from qc_agent.server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8788, reload=False, log_level="debug")

