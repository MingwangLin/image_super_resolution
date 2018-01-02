from app import init_app
from app import models


# ...

def rebuild_db():
    application = init_app()
    db = models.db
    db.drop_all()
    db.create_all()
    print('auth rebuild database')


def run():
    config = dict(
        debug=True,
    )
    init_app().run(**config)


if __name__ == '__main__':
    # rebuild_db()
    run()
