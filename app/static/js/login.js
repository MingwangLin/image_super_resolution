$(document).ready(function () {
    log('ready');
    __main()
});

var __main = function () {
    setup();
    bindActions();
    // select signup
    $('#id-a-signup').click();
};

var setup = function () {
    $('.gua-tab> a').on('click', function () {
        var self = $(this);
        $('.active').removeClass('active');
        self.addClass('active');
    });
    var tabAction = function (showLogin) {

        $('#id-div-login').toggle(showLogin);
        $('#id-div-signup').toggle(!showLogin);
    };

    $('#id-a-signup').on('click', function () {
        var showLogin = false;
        tabAction(showLogin);
    });
    $('#id-a-login').on('click', function () {
        var showLogin = true;
        tabAction(showLogin);
    });


};

var bindActions = function () {
    $('#id-button-login').on('click', function () {
        login();
    });
    $('#id-button-register').on('click', function () {
        register();
    });
    $('#id-input-login-password').on('keypress', function (e) {
        if (e.which == 13) {
            login()
        }
    });
    $('#id-input-password').on('keypress', function (e) {
        if (e.which == 13) {
            register()
        }
    });
    $('#id-button-testuser').on('click', function () {
        var url = '/testuser'
        log('testurl', url)
        get(url, testuser_response)
    });

};

var login = function () {
    var username = $('#id-input-login-username').val();
    var password = $('#id-input-login-password').val();
    log('user', username, password);
    var form = {
        'username': username,
        'password': password
    };
    var url = '/login'
    post($input_box = ``, url, form, response = auth_response, $target = ``)
}

var register = function () {
    var username = $('#id-input-username').val();
    var password = $('#id-input-password').val();
    log('user', username, password);
    var form = {
        'username': username,
        'password': password
    };
    // JSON.stringify 可以把一个 object 转换为字符串
    var url = '/register'
    log('form.content', form)
    post($input_box = $('#id-input-username'), url, form, response = auth_response, $target = ``)
}

var testuser = function (url, response) {
    get(url, form, response)
}

var auth_response = function (r) {
    log('response', r);
    if (r.success) {
        log('成功');
        var url = r.data;
        window.location.href = url;
        $('#id-p-info').text(r.message);
    } else {
        log('失败');
    }
    $('#id-p-info').text(r.message);
};

var testuser_response = function (r) {
    log('response', r);
    if (r.success) {
        log('成功');
        $('#id-p-info').text(r.message);
        var url = 'timeline/' + r.user.username;
        window.location.href = url;
    } else {
        log('失败');
    }
    $('#id-p-info').text(r.message);
}
