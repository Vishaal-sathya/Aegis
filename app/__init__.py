from flask import Flask
from .config import Config
from .load_model import load_model_pt

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_object(Config)

    app.model = load_model_pt(
        checkpoint_path=app.config["MODEL_PATH"],
        device=app.config["DEVICE"],
        num_classes=app.config["NUM_CLASSES"],
    )
    app.device = app.config["DEVICE"]

    from .routes.main import main_bp
    app.register_blueprint(main_bp)

    from .routes.model_routes import model_bp
    app.register_blueprint(model_bp)

    from .routes.pad_routes import pad_bp
    app.register_blueprint(pad_bp)  # 👈 now PAD is registered

    return app
