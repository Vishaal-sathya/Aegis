from flask import Flask
from .config import Config
from .load_model import load_model

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Load configs
    app.config.from_object(Config)

    # Load model once at startup
    app.model = load_model(
        checkpoint_path=app.config["MODEL_PATH"],
        device=app.config["DEVICE"],
        num_classes=app.config["NUM_CLASSES"],
    )
    app.device = app.config["DEVICE"]

    # Register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app
