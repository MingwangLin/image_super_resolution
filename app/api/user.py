from flask import abort
from flask import jsonify
from flask import session

from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

from .treelog import log
from .login import hash_password, fake_tweet, string_generator

from ..models import At, Comment, User, Tweet, Follow
from . import api
from .decorator import requires_login, requires_admin, current_user

from .notification import save_notification, user_notified
from .sr import process_img
from werkzeug.utils import secure_filename
import os


# 显示某个用户的主页  GET
@api.route('/timeline/<username>')
@requires_login
def timeline_view(username):
    # 查找 username 对应的用户
    visitor = current_user()
    host = User.query.filter_by(username=username).first()
    if host is None:
        abort(404)
    else:
        args = {
            'visitor': visitor,
            'host': host,
        }
        return render_template('timeline_bs.html', **args)


# 用ajax显示某个用户的时间线  GET
@api.route('/tweets/<username>')
@requires_login
def timeline_ajax(username):
    # 查找 username 对应的用户
    visitor = current_user()
    host = User.query.filter_by(username=username).first()
    if host is None:
        abort(404)
    else:
        # args = request.args
        # offset = args.get('offset', 0)
        # offset = int(offset)
        # limit = args.get('limit', 20)
        # limit = int(limit)
        # tweets = host.tweets
        # tweets.sort(key=lambda t: t.created_time, reverse=True)
        # tweets = tweets[offset:offset + limit]
        # tweets = [t.json() for t in tweets]
        args = request.args
        page = args.get('page', 1, type=int)
        log('p', request.args)
        pagination = Tweet.query.filter_by(user_id=host.id).order_by(
            Tweet.created_time.desc()).paginate(
            page, error_out=False)
        my_tweets = pagination.items
        tweets = [i.json() for i in my_tweets]
        # 每条tweet json都加上转发的原创微博
        for i in range(len(tweets)):
            tweets[i]['original_tweet'] = [j.reposted.json() for j in my_tweets[i].reposted.all()]
        log('tweets', tweets)
        tweets_perpage = {
            'success': True,
            'host': host.json(),
            'visitor': visitor.json(),
            'tweets': tweets,

        }
        # log('filtered_tweets', filtered_tweets)
        return jsonify(tweets_perpage)


# 用ajax显示关注人微博时间线  GET
@api.route('/followedtweets/<username>')
@requires_login
def timeline_followed(username):
    # 查找 username 对应的用户
    visitor = current_user()
    host = User.query.filter_by(username=username).first()
    if host is None:
        abort(404)
    else:
        args = request.args
        page = args.get('page', 1, type=int)
        # log('p', request.args)
        pagination = Tweet.query. \
            join(Follow, Follow.followed_id == Tweet.user_id) \
            .filter(Follow.follower_id == visitor.id).order_by(
            Tweet.created_time.desc()).paginate(
            page, per_page=10, error_out=False)
        followed_tweets = pagination.items
        # log('f', followed_tweets)
        tweets = [i.json() for i in followed_tweets]
        # 每条tweet json都加上转发的原创微博
        for i in range(len(tweets)):
            tweets[i]['original_tweet'] = [j.reposted.json() for j in followed_tweets[i].reposted.all()]
        log('tweets', tweets)
        tweets_perpage = {
            'success': True,
            'host': host.json(),
            'visitor': visitor.json(),
            'tweets': tweets

        }
        return jsonify(tweets_perpage)

# 用ajax上传用户头像
@api.route('/upload/avatars', methods=['POST'])
@requires_login
def upload_avatars():
    user = current_user()
    # uploaded 是上传时候的文件名
    file = request.files.get('uploaded')
    log('upload, ', request.files)
    data = {}
    if file:
        filename = file.filename
        filename = secure_filename(file.filename)
        log('filename, ', filename)
        path = '/static/avatars/' + filename
        abs_path = '/var/www/twitter/app' + path
        # abs_path = '/Users/linmingwang/twitter/app' + path
        # abs_path = os.path.join(path, filename)
        # log('abs', abs_path)
        file.save(abs_path)
        # file.save(path)
        user.avatar = path
        user.save()
        log('user', user.username)
        url = '/timeline/'+ user.username
        data = {
            'success': True,
            'url': url,
        }
    else:
        data['success'] = False
    return jsonify(data)

# 用ajax上传微博图片
@api.route('/upload/picture', methods=['POST'])
@requires_login
def upload_picture():
    user = current_user()
    file = request.files.get('uploaded')
    log('upload, ', request.files)
    log('file', file.filename)
    data = {}
    if file:
        img_format = file.filename.split('.')[1]
        filename = string_generator(size=8) + '.' + img_format
        log('filename, ', filename)
        dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log ('dirname', dir_name)
        img_path = '/static/tweets_picture/' + filename
        abs_path = os.path.join(dir_name, img_path)
        log('abs', abs_path)
        file.save(abs_path)
        process_img(abs_path)
        # file.save(path)
        url = img_path # for online apache
        url = abs_path # for local pc debug
        data = {
            'success': True,
            'url': url,
        }

    else:
        data['success'] = False
    return jsonify(data)


# 删除用户
@api.route('/admin/users/delete/<user_id>')
@requires_admin
def user_delete(user_id):
    user = current_user()
    u = User.query.filter_by(id=user_id).first()
    u.delete()
    return redirect(url_for('users_view'))


##显示更新用户的界面
@api.route('/admin/users/update/<user_id>')
@requires_login
@requires_admin
def user_update_view(user_id):
    user = current_user()
    u = User.query.filter_by(id=user_id).first()
    return render_template('user_edit.html', user=u)


# 处理更新用户的请求
@api.route('/admin/users/update/<user_id>', methods=['POST'])
@requires_login
@requires_admin
def user_update(user_id):
    user = current_user()
    u = User.query.filter_by(id=user_id).first()
    u.password = request.form.get('password', '')
    u.save()
    return redirect(url_for('users_view'))
