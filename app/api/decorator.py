from functools import wraps

from flask import abort
from flask import redirect
from flask import session
from flask import url_for

from ..models import User
from .treelog import log


# 通过 session 来获取当前登录的用户
def current_user():
    user_id = session['user_id']
    log('user__id', user_id)
    user = User.query.filter_by(id=user_id).first()
    log('user', user)
    return user


def requires_login(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        # f 是被装饰的函数
        # 所以下面两行会先于被装饰的函数内容调用
        user = current_user()
        if user is None:
            return redirect(url_for('api.login_view'))
        return f(*args, **kwargs)

    return wrapped


def requires_admin(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        # f 是被装饰的函数
        # 所以下面两行会先于被装饰的函数内容调用
        print('debug, requires_login')
        user = current_user()
        if user.role != 1:
            # return abort(401)
            return abort(404)
        return f(*args, **kwargs)

    return wrapped
