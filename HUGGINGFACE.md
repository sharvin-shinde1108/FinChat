# Deploying to Hugging Face Spaces

Quick steps to deploy this project as a Gradio Space.

1. Create a new Space on Hugging Face: choose the `Gradio` SDK (or `Python`) and public/private as desired.
2. In the Space settings, add repository secrets for any API keys used (e.g., `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`, etc.).
   - Go to the Space → Settings → Repository secrets and add keys.
3. Push this repository (or the selected files) to the Space repository. This project includes:
   - `requirements.txt` — Python deps for Spaces to install
   - `start_gradio.py` — entrypoint that launches the Gradio `demo` in `gradio_app.py`

How to push from your local machine:

```bash
git add requirements.txt start_gradio.py HUGGINGFACE.md
git commit -m "Add Hugging Face deployment files"
git push origin main
```

How Spaces runs it:
- The Space installer will install packages from `requirements.txt`.
- The default run command can be `python start_gradio.py` (the UI will be served on port 7860).

Notes and tips:
- Do not commit secrets. Use Repository secrets in the Space for environment variables.
- If your app requires larger CPU/memory, pick a paid Space or optimize dependencies.
- If Gradio fails to launch, check the Space logs for errors and ensure `PORT` is not hard-coded.
