from .. import db


class ReprMixin(object):
    def __repr__(self):
        class_name = self.__class__.__name__
        return u'<{}: {}>'.format(class_name, self.id)

from .user import User
from .tweet import Tweet, TweetImg
from .comment import Comment
from .notification import At
from .follow import Follow
from .repost import Repost
