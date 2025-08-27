import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from app import create_app  # assuming __init__.py defines create_app()

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
