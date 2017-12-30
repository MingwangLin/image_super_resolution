from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import jsonify
from ..models import Comment, Tweet
from . import api
from .decorator import requires_login, current_user
from .notification import save_notification_in_comment, user_notified
from .treelog import log


# # 显示发送评论的界面
# @api.route('/tweet/comment/<tweet_id>')
# def comment_add_view(tweet_id):
#     user = current_user()
#     tweet = Tweet.query.filter_by(id=tweet_id).first()
#     if user is None:
#         return redirect(url_for('login_view'))
#     else:
#         return render_template('comment_add.html', tweet=tweet)


# 处理发送评论的函数
@api.route('/comment/add/<tweet_id>', methods=['POST'])
@requires_login
def comment_add(tweet_id):
    user = current_user()
    form = request.get_json()
    c = Comment(form)
    log('c', c)
    # 设置是谁发的
    c.user = user
    # 保存到数据库
    tweet = Tweet.query.filter_by(id=tweet_id).first()
    c.tweet = tweet
    c.save()

    r = {
        'success': True,
        'comment': c.json(),
        'user': user.json(),
    }
    log('r', r)
    # 获取评论中@的用户名, 生成相应的At实例, 存入数据库
    if '@' in c.content:
        name_lst = user_notified(c.content)
        save_notification_in_comment(lst=name_lst, comment=c)
    return jsonify(r)


# 用 ajax 返回某条微博的评论
@api.route('/tweet/comments/<tweet_id>', methods=['GET'])
@requires_login
def comments(tweet_id):
    visitor = current_user()
    tweet = Tweet.query.filter_by(id=tweet_id).first()
    args = request.args
    page = args.get('page', 1, type=int)
    log('page', page)
    pagination = Comment.query.filter_by(tweet_id=tweet.id).order_by(
        Comment.created_time.desc()).paginate(
        page, error_out=False)

    comments = pagination.items

    log('items', comments)
    comments = [i.json() for i in comments]
    comments = {
        'success': True,
        'visitor': visitor.json(),
        'comments': comments,

    }
    # log('filtered_tweets', filtered_tweets)
    return jsonify(comments)


# 显示发送回复的界面
@api.route('/tweet/reply/<comment_id>')
@requires_login
def reply_add_view(comment_id):
    user = current_user()
    comment = Comment.query.filter_by(id=comment_id).first()
    return render_template('reply_add.html', comment=comment)


# 处理发送回复的函数
@api.route('/tweet/reply/<comment_id>', methods=['POST'])
@requires_login
def reply_add(comment_id):
    user = current_user()
    comment = Comment.query.filter_by(id=comment_id).first()
    tweet = comment.tweet
    c = Comment(request.form)
    # 谁发的
    c.user = user
    c.user_id = user.id
    # 发到哪条微博
    c.tweet = tweet
    c.tweet_id = tweet.id
    # 回复哪条评论
    c.comment_replied = comment_id
    # 回复哪位用户
    c.user_replied = comment.user.username
    c.save()
    if '@' in c.content:
        name_lst = user_notified(c.content)
        save_notification_in_comment(lst=name_lst, comment=c)
    return redirect(url_for('api.tweet_view', tweet_id=tweet.id))
