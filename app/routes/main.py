from flask import Blueprint, render_template, current_app, request, jsonify


main_bp = Blueprint("main", __name__)

@main_bp.route('/')
def main():
    pad_mode = 2   # or 2, based on your requirement / config
    return render_template("index.html", pad_mode=pad_mode)