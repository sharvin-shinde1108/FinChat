import os
from gradio_app import demo

def main():
    port = int(os.environ.get("PORT", 7860))
    # Bind to all interfaces for Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
