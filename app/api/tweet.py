from flask import abort
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

from .treelog import log
from ..models import Tweet, TweetImg
from . import api
from .decorator import requires_login, current_user
from .notification import save_notification, user_notified
import json


# 显示单条微博的界面
@api.route('/tweet/<tweet_id>')
@requires_login
def tweet_view(tweet_id):
    t = Tweet.query.filter_by(id=tweet_id).first()
    u = current_user()
    return render_template('single_tweet_view.html', t=t, u=u)


# 处理添加微博的函数  POST
@api.route('/tweet/add', methods=['POST'])
def tweet_add():
    user = current_user()
    form = request.get_json()
    t = Tweet(form)
    log('t', t)
    # 设置是谁发的
    t.user = user
    # 保存到数据库
    t.save()
    imgs_url = form.get('imgs_url')
    for i in imgs_url:
        s = TweetImg(img_url=i)
        s.tweet = t
        s.save()
    tweet = Tweet.query.filter_by(id=t.id).first()
    r = {
        'success': True,
        'tweet': tweet.json(),
        'user': user.json(),
    }
    log('r', r)
    # 获取微博中@的用户名, 生成相应的At实例, 存入数据库
    if '@' in t.content:
        name_lst = user_notified(t.content)
        save_notification(lst=name_lst, tweet=t)
    return jsonify(r)


# 显示 更新 微博的界面
@api.route('/tweet/update/<tweet_id>')
def tweet_update_view(tweet_id):
    t = Tweet.query.filter_by(id=tweet_id).first()
    if t is None:
        abort(404)
    # 获取当前登录的用户, 如果用户没登录或者用户不是这条微博的主人, 就返回 401 错误
    user = current_user()
    if user is None or user.id != t.user_id:
        abort(401)
    else:
        return render_template('tweet_edit.html', tweet=t)


# 处理 更新 微博的请求
@api.route('/tweet/update/<tweet_id>', methods=['POST'])
def tweet_update(tweet_id):
    t = Tweet.query.filter_by(id=tweet_id).first()
    if t is None:
        abort(404)
    # 获取当前登录的用户, 如果用户没登录或者用户不是这条微博的主人, 就返回 401 错误
    user = current_user()
    if user is None or user.id != t.user_id:
        abort(401)
    else:
        t.content = request.form.get('content', '')
        t.save()
        return redirect(url_for('api.timeline_view', username=user.username))


# 处理 删除 微博的请求
@api.route('/tweet/delete/<tweet_id>')
def tweet_delete(tweet_id):
    t = Tweet.query.filter_by(id=tweet_id).first()
    if t is None:
        abort(404)
    # 获取当前登录的用户, 如果用户没登录或者用户不是这条微博的主人, 就返回 401 错误
    user = current_user()
    if user is None or user.id != t.user_id:
        abort(401)
    else:
        t.delete()
        return redirect(url_for('api.timeline_view', username=user.username))
