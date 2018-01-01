// log
var log = function () {
    console.log(arguments);
};

// form


var ajax = function (url, method, form, response, $target) {
    var request = {
        url: url,
        type: method,
        contentType: 'application/json',
        success: function (r) {
            log('success', url, r);
            response(r, $target);
        },
        error: function (err) {
            r = {
                success: false,
                message: '服务器提了一个问题',
                data: err
            }
            log('err', err)
            log('err', url, err);
            response(r, $target);
        }
    };
    if (method === 'post') {
        var data = JSON.stringify(form);
        request.data = data;
    }
    $.ajax(request);
};

var get = function (url, response, $target) {
    var method = 'get';
    var form = {}
    ajax(url, method, form, response, $target);
};

var post = function ($input_box, url, form, response, $target) {
    if (form.content) {
        var content_filtered = form.content.replace(/ /g, '')
        log('content_filtered', content_filtered)
    }
    // 发送请求前判断用户输入是否为空或是否只有空格
    if (content_filtered == '') {
        $input_box.css({
            "background-color": "#F88E8B",
            "transition": "0.5s",
        });
        setTimeout(function () {
            $input_box.css({"background-color": "white"});
        }, timeout = 1000)
    } else {
        var method = 'post';
        log('url', url)
        ajax(url, method, form, response, $target);
    }
};

var formatted_time = function (timestamp) {
    // multiplied by 1000 so that the argument is in milliseconds, not seconds
    var a = new Date(timestamp * 1000);
    var now = new Date().getTime()
    var now = Math.floor(now / 1000);
    var interval = now - timestamp;
    if (interval <= 0) {
        var time = `now`
    } else if (interval < 60) {
        var time = `${interval}seconds ago`
    } else if (interval >= 60 && interval < 3600) {
        var time = `${Math.floor(interval / 60)}minutes ago`
    } else if (interval >= 3600 && interval < 3600 * 24) {
        var time = `${Math.floor(interval / 3600)}hours ago`
    } else if (interval >= 3600 * 24 && interval < 3600 * 24 * 7) {
        var time = `${Math.floor(interval / (3600 * 24))}days ago`
    } else {
        var year = a.getFullYear();
        var month = a.getMonth();
        var date = a.getDate();
        var hour = a.getHours();
        var min = a.getMinutes();
        if (min < 10) {
            min = '0' + min
        }
        var time = year + '/' + month + '/' + date + ' ' + hour + ':' + min;
    }
    return time;
}

var href_for_personalpage = function (name) {
    // multiplied by 1000 so that the argument is in milliseconds, not seconds
    template = `<a class="font-tweet-name" href="/timeline/${name}">${name}</a>`;
    return template;
}

// API products
var products = function (response) {
    var api = this;
    var path = '/api/products';
    get(path, response);
};

var product_delete = function (product_id, response) {
    var path = '/api/products/delete/' + product_id;
    get(path, response);
};
// API articles
var articles = function (response) {
    var path = '/api/articles';
    get(path, response);
};

var href_for_username = function (content) {
    while (i < 10) {
        text += "The number is " + i;
        i++;
    }
    var i = content.indexof("@")
    var j = content.indexof(":")
    template = `<a href="/timeline/${username}">${username}</a>`
    return template
}
