# -*- coding: utf-8 -*-
from flask import Flask
import time
from flask_sqlalchemy import SQLAlchemy
import os


# 暴露 db 是因为 models 要使用它
# 但是这时候还没有 app 所以要在 app 初始化之后再初始化这个 db
db = SQLAlchemy()


# 把 flask 的初始化放到函数中
# 由外部启动函数来调用

def init_app():
    dir_name = os.path.dirname(os.getcwd())
    db_name = 'db.sqlite'
    db_path = os.path.join(dir_name, db_name)
    # 初始化并配置 flask
    app = Flask(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
    app.secret_key = 'TODO fixme'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(db_path)
    # 初始化 db
    db.init_app(app)
    db.app = app

    @app.template_filter('formatted_time')
    def formatted_time(timestamp):
        t = timestamp
        format = '%Y/%m/%d %H:%M'
        t = time.localtime(timestamp)
        ft = time.strftime(format, t)
        return ft

    # 必须在函数中 import 蓝图
    # 否则循环引用(因为蓝图中 import 了 model, model 引用了这个文件里面目的 db)
    from .api import api as api
    # 注册蓝图
    app.register_blueprint(api)
    # 把 app 引用返回
    return app
