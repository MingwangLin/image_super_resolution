from flask import redirect
from flask import render_template
from flask import url_for

from ..models import User, Follow
from . import api
from .decorator import requires_login, current_user
from .treelog import log

# 处理关注用户函数
@api.route('/follow/<user_id>')
@requires_login
def user_follow(user_id):
    u = current_user()
    if current_user() is None:
        return redirect(url_for('login_view'))
    f = Follow(follower_id=u.id, followed_id=user_id)
    f.save()
    return redirect(url_for('api.timeline_view', username=user.username))

# 处理取消关注用户函数
@api.route('/unfollow/<user_id>')
@requires_login
def user_unfollow(user_id):
    u = current_user()
    user = User.query.filter_by(id=user_id).first()
    f = Follow.query.filter_by(follower_id=u.id).first()
    f.delete()
    return redirect(url_for('api.timeline_view', username=user.username))


# 显示所有用户信息
@api.route('/admin/users')
def users_view():
    users = User.query.all()
    return render_template('users_view.html', users=users)
