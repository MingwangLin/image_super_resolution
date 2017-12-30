from . import db
from . import ReprMixin
import time

class Repost(db.Model):
    __tablename__ = 'reposts'
    repost_id = db.Column(db.Integer, db.ForeignKey('tweets.id'),
                            primary_key=True)
    reposted_id = db.Column(db.Integer, db.ForeignKey('tweets.id'),
                            primary_key=True)

    def __init__(self, repost_tweet_id, reposted_tweet_id):
        super(Repost, self).__init__()
        self.repost_id = repost_tweet_id
        self.reposted_id = reposted_tweet_id

    def __repr__(self):
        class_name = self.__class__.__name__
        return u'<{} {}:{}>'.format(class_name, self.repost_id, self.reposted_id)

    def json(self):
        d = {
        'repost_id': self.repost_id,
        'reposted_id': self.reposted_id,
        }
        return d

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def save(self):
        db.session.add(self)
        db.session.commit()
