# One-line Docker run

Build and run:
```bash
docker build -t model-validation-assistant:latest . && docker run --rm -p 8000:8000 -e OPENAI_API_KEY="${OPENAI_API_KEY}" model-validation-assistant:latest
```

Once running, open http://localhost:8000/docs to interact with the demo endpoints.
