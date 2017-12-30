from flask import Blueprint

api = Blueprint('api', __name__)

from . import login, user, decorator, notification, tweet, repost, comment, follow
