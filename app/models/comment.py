from . import db
from . import ReprMixin
from .user import User
import time

class Comment(db.Model, ReprMixin):
    __tablename__ = 'comments'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String())
    created_time = db.Column(db.Integer(), default=0)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    tweet_id = db.Column(db.Integer, db.ForeignKey('tweets.id'))
    comment_replied = db.Column(db.Integer())
    user_replied = db.Column(db.String())
    reply_viewed = db.Column(db.Integer(), default=0)
    ats = db.relationship('At', backref='comment')

    def __init__(self, form):
        super(Comment, self).__init__()
        self.content = form.get('content', '')
        self.created_time = int(time.time())

    def __repr__(self):
        class_name = self.__class__.__name__
        return u'<{}: {}>'.format(class_name, self.id)

    def json(self):
        extra = dict(
            tweet_id = self.tweet_id,
            user_name=User.query.filter_by(id=self.sender_id).first().username
        )
        d = {k: v for k, v in self.__dict__.items() if k not in self.blacklist()}
        d.update(extra)
        return d

    def blacklist(self):
        b = [
            '_sa_instance_state',
            'id',
            'ats',
            'comment_replied',
            'user_replied',
            'reply_viewed',
            'sender_id',
        ]
        return b

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()
