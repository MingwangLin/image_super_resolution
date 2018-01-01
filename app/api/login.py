import hashlib
import random
import string

from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from .treelog import log
from ..models import User, Follow, Tweet, TweetImg
from . import api
from .notification import save_notification, user_notified


def hash_password(pwd):
    m = hashlib.md5()
    m.update(pwd.encode('utf-8'))
    result = m.hexdigest()
    return result


@api.route('/')
def index():
    return redirect(url_for('api.login_view'))


# 显示登录界面的函数  GET
@api.route('/login')
def login_view():
    return render_template('login.html')


# 处理登录请求  POST
@api.route('/login', methods=['POST'])
def login():
    account = request.get_json()
    log('account', account)
    u = User(account)
    u.password = hash_password(u.password)
    user = User.query.filter_by(username=u.username).first()
    r = {
        'success': True,
        'message': '登录成功',
        'code': '成功',
        'data': {

        }
    }
    if user.validate(u):
        log("用户登录成功")
        # 用 make_response 生成响应 并且设置 cookie
        session['user_id'] = user.id
        r['data'] = '/timeline/{}'.format(user.username)
    else:
        r['message'] = '登录失败'
    return jsonify(r)


# 处理注册的请求  POST
@api.route('/register', methods=['POST'])
def register():
    account = request.get_json()

    data = request.get_data()
    '''
    json_string = data.decode('utf-8')
    d = json.loads(json_string)
    log('json_string', json_string)
    log('d', d)
    '''
    log('account', account)
    log('data', data)

    u = User(account)
    usr = User.query.filter_by(username=u.username).first()
    r = {
        'success': True,
        'message': '注册成功',
        'code': '成功',
        'data': {

        }
    }
    if u.valid() and usr is None:
        log("用户注册成功")
        u.password = hash_password(u.password)
        # 保存到数据库
        u.save()
        user = User.query.filter_by(username=u.username).first()
        user_id = user.id
        session['user_id'] = user_id
        r['data'] = '/timeline/{}'.format(u.username)
        # 自动关注之前所有人
        for i in range(1, user_id):
            f = Follow(user_id, i)
            f.save()
        admin_id = 1
        admin = User.query.filter_by(id=admin_id).first()
        # 向该用户发送@通知供测试
        content = '@' + user.username + ' ' + 'test'
        fake_tweet(content, user_id=admin_id)
        content = 'test'
        fake_tweet(content, user_id=user_id)
    else:
        r['message'] = '注册失败'
    return jsonify(r)


@api.route('/signout')
def signout():
    session['user_id'] = None
    log('session', session)
    return redirect(url_for('api.login_view'))


# 自动创建用户
@api.route('/testuser', methods=['GET'])
def fake_user():
    form = {
        'username': 'visitor' + string_generator(size=4),
        'password': string_generator(size=6),
    }
    user = User(form)
    # 写入关注人信息
    user.password = hash_password(user.password)
    # 保存到数据库
    user.save()
    user = User.query.filter_by(username=user.username).first()
    # 自动关注之前所有人
    for i in range(1, user.id):
        f = Follow(user.id, i)
        f.save()
    admin_id = 1
    admin = User.query.filter_by(id=admin_id).first()
    session['user_id'] = user.id
    content = '@' + user.username + ' ' + 'test'
    fake_tweet(content, user_id=admin_id)
    content = 'test'
    user_id = user.id
    fake_tweet(content, user_id=user_id)
    created_user = {
        'success': True,
        'user': user.json(),
        'message': '登录成功',
    }
    return jsonify(created_user)


# 自动创建微博
def fake_tweet(content, user_id):
    # 向该用户发送@通知供测试
    form = {
        'content': content
    }
    t = Tweet(form)
    t.user_id = user_id
    t.save()
    for i in range(1, 10):
        img_url = '/static/tweets_picture/' + str(i) + '.jpg'
        s = TweetImg(img_url)
        s.tweet = t
        s.save()
    # 根据解析微博得到的@的用户名数组, 生成相应的At实例, 存入数据库
    name_lst = user_notified(t.content)
    save_notification(lst=name_lst, tweet=t)
    return


def string_generator(size):
    # chars = string.ascii_uppercase + string.digits
    chars = string.digits
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))
