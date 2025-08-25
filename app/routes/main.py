from flask import Blueprint, render_template, current_app, request, jsonify


main_bp = Blueprint("main", __name__)

@main_bp.route('/')
def main():
    return render_template('index.html')