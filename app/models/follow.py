from . import db
from . import ReprMixin
import time

class Follow(db.Model, ReprMixin):
    __tablename__ = 'follows'
    follower_id = db.Column(db.Integer, db.ForeignKey('users.id'),
                            primary_key=True)
    followed_id = db.Column(db.Integer, db.ForeignKey('users.id'),
                            primary_key=True)
    created_time = db.Column(db.Integer(), default=0)

    def __init__(self, follower_id, followed_id):
        super(Follow, self).__init__()
        self.follower_id = follower_id
        self.followed_id = followed_id
        self.created_time = int(time.time())

    def __repr__(self):
        class_name = self.__class__.__name__
        return u'<{} {}:{}>'.format(class_name, self.follower_id, self.followed_id)

    def json(self):
        d = {
        'follower_id': self.follower_id,
        'followed_id': self.followed_id,
        'created_time': self.created_time,
        }
        return d

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def save(self):
        db.session.add(self)
        db.session.commit()
