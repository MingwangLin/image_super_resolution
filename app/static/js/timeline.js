// 请求数据库一页的数据
var mytweets_page = 1;
var followedtweets_page = 1;
var notifications_page = 1;
var comments_page = 1;

$(document).ready(function () {
    __main();
});

var __main = function () {
    setup();
    bindActions();
    $('#id-a-mytweets').click();
};

var pageturning_button_show_hide = function (jquery_page_object, jquery_button_object) {
    var entries_perpage = 10;
    if (jquery_page_object.children().length < 10) {
        jquery_button_object.hide();
    } else if (jquery_page_object.children().length == 10) {
        jquery_button_object.show();
    }
};

var setup = function () {
    // web当前路径
    path = window.location.pathname;
    username_index = 2;
    username = path.split('/')[username_index];
    mytweets_url = '/tweets/' + username + '?page=' + mytweets_page;
    followedtweets_url = '/followedtweets/' + username + '?page=' + followedtweets_page;
    notifications_url = '/notifications/' + username + '?page=' + notifications_page;
    get(mytweets_url, mytweets_response);
    get(followedtweets_url, followedtweets_response);
    get(notifications_url, notifications_response);
    $('.blog-nav> a').on('click', function () {
        var self = $(this);
        $('.active').removeClass('active');
        self.addClass('active');
    });

    var tabAction = function (show_mytweets, show_followedtweets, show_notifications) {
        $('#id-div-mypage').toggle(show_mytweets);
        $('#id-div-followedpage').toggle(show_followedtweets);
        $('#id-div-notificationpage').toggle(show_notifications);
    };

    $('#id-a-mytweets').on('click', function () {
        var show_notifications = false;
        var show_followedtweets = false;
        var show_mytweets = true;
        tabAction(show_mytweets, show_followedtweets, show_notifications);
    });

    $('#id-a-followedtweets').on('click', function () {
        var show_notifications = false;
        var show_followedtweets = true;
        var show_mytweets = false;
        tabAction(show_mytweets, show_followedtweets, show_notifications);
    });

    $('#id-a-notifications').on('click', function () {
        var show_notifications = true;
        var show_followedtweets = false;
        var show_mytweets = false;
        tabAction(show_mytweets, show_followedtweets, show_notifications);
    });

    // 页面加载完毕后js会向服务器发送一个ajax请求一页的微博数，服务器会从数据库拿一页的微博数返回，默认是20条。
    // 如果返回的微博没超过20的话，就不需要显示翻页按钮。
    // 下面的函数会做一个判断 ，决定隐藏还是显示翻页按钮。
    // 需要等页面的第一页微博加载完毕后再执行函数，所以做了一个延时。
    setTimeout(function () {
        pageturning_button_show_hide($('#id-div-mytweets'), $('#id-button-next-mytweets'))
    }, 3000);
    setTimeout(function () {
        pageturning_button_show_hide($('#id-div-followedtweets'), $('#id-button-next-followedtweets'))
    }, 3000);
    setTimeout(function () {
        pageturning_button_show_hide($('#id-div-notification'), $('#id-button-next-notifications'))
    }, 3000)
};

var bindActions = function () {
    $('#id-button-next-mytweets').on('click', function () {
        var increment = 1;
        mytweets_page += increment;
        mytweets_url = '/tweets/' + username + '?page=' + mytweets_page;
        log('m_url', mytweets_url);
        get(mytweets_url, mytweets_response)
    });

    $('#id-button-next-followedtweets').on('click', function () {
        followedtweets_url = '/followedtweets/' + username + '?page=' + followedtweets_page;
        get(followedtweets_url, followedtweets_response);
    });

    $('#id-button-next-notifications').on('click', function () {
        var increment = 1;
        followedtweets_page += increment;
        notifications_url = '/notifications/' + username + '?page=' + notifications_page;
        get(notifications_url, notifications_response);
    });

    $('#id-button-addtweet').on('click', add_newtweet);

    $('#id-div-twitter').on('click', '.button-comments', function () {
        var $single_tweet = $(this).closest(".singletweet");
        var tweet_id = $single_tweet.data('id');
        comments_url = '/tweet/comments/' + tweet_id + '?page=' + comments_page;
        var $interect_area = $(this).closest(".div-interact-area");
        var $repost_area = $interect_area.find(".div-repostarea");
        var $comments_area = $interect_area.find(".div-commentarea");
        var $comments = $interect_area.find(".well-sm");
        log('len', $comments);
        var $comment_button = $interect_area.find(".comment-button-text");
        var $repost_button = $interect_area.find(".repost-button-text");
        if ($comments_area.is(':hidden')) {
            if ($comments.length == 0) {
                get(comments_url, comments_response, $comments_area);
                $interect_area.find(".div-commentarea").slideToggle("slow");
            } else {
                $interect_area.find(".div-commentarea").slideToggle("slow");
            }
            $repost_area.hide();
            $comment_button.text('Hide');
            $repost_button.text('Retweet');
        } else {
            $interect_area.find(".div-commentarea").slideToggle("slow");
            $comment_button.text('Reply' + $comments.length);
        }
    });

    $('#id-div-twitter').on('click', '.button-reposts', function () {
        var $interect_area = $(this).closest(".div-interact-area");
        var $repost_area = $interect_area.find(".div-repostarea");
        var $comment_area = $interect_area.find(".div-commentarea");
        var $comment_button = $interect_area.find(".comment-button-text");
        var $repost_button = $interect_area.find(".repost-button-text");
        var $comments = $interect_area.find(".well-sm");
        log('gh', $repost_area.is(':visible'));
        if ($repost_area.is(':hidden')) {
            $repost_area.slideToggle("slow");
            $comment_area.hide();
            $repost_button.text('Hide');
            $comment_button.text('Reply' + $comments.length);
        } else {
            $repost_area.slideToggle("slow");
            $repost_button.text('Retweet');
        }
    });

    $('#id-div-twitter').on('click', '.button-addcomment', add_newcomment);

    $('#id-div-twitter').on('click', '.button-addrepost', add_newrepost);

    $('#id-button-avatars').on('click', function () {
        $('.file-wrapper').toggle("slow");
    });

    $('#id-button-addpicture').on('click', function () {
        $('.upload-wrapper').toggle("slow");
    });

    $('#id-button-upload-avatar').on('click', function () {
        var fileTag = $('#id-input-file')[0];
        log('fileTag', fileTag);
        var files = fileTag.files;
        log('files', files);
        var numberOfFiles = files.length;
        if (numberOfFiles == 0) {
            alert('还没有选中文件');
        } else {
            for (var i = 0; i < numberOfFiles; i++) {
                var file = files[i];
                console.log('上传的文件: ', file.name);
                upload_avatar(file);
            }
        }
    });

    $('#id-button-upload-picture').on('click', function () {
        var fileTag = $('#id-input-picture')[0];
        log('fileTag', fileTag);
        var files = fileTag.files;
        log('files', files);
        var numberOfFiles = files.length;
        if (numberOfFiles == 0) {
            alert('还没有选中文件');
        } else {
            for (var i = 0; i < numberOfFiles; i++) {
                var file = files[i];
                console.log('上传的文件: ', file.name);
                upload_picture(file);
            }
            $(".upload-wrapper").append(`<p class="upload-info">上传中……</p>`);
        }
    });

    $('#id-div-twitter').on('click', 'img', function () {
        var $modal = $('.modal');
        img_src = $(this).attr("src");
        log('img_src', img_src);
        $modal.find('img').attr("src", img_src);
        log('m', $modal.attr("src"));
        $modal.show();
    });
    // Get the <span> element that closes the modal
    // When the user clicks on <span> (x), close the modal
    $('.close').on('click', function () {
        $('.modal').hide();
    });
};

var mytweets_response = function (data) {
    if (data.success) {
        var increment = 1;
        mytweets_page += increment;
        log('success', data);
        var tweets = data.tweets;
        if (tweets.length == 0 && $('#id-div-mytweets').children().length == 0) {
            $('#id-div-mytweets').append(none_template);
            var timeout = 3000;
            setTimeout(function () {
                $('.none').remove()
            }, timeout)
        } else if (tweets.length == 0 && $('#id-div-mytweets').children().length > 0) {
            $('#id-div-mypage').append(nomore_template);
            var timeout = 1000;
            setTimeout(function () {
                $('p.nomore').remove()
            }, timeout)
        } else {

            var host = data.host;
            var visitor = data.visitor;
            show_tweets_onpage(tweets, host, visitor, $page = $('#id-div-mytweets'));
        }

    } else {
        log('请求失败');
    }
};

var followedtweets_response = function (data) {
    if (data.success) {
        var increment = 1;
        followedtweets_page += increment;
        log('success', data);
        var tweets = data.tweets;
        var host = data.host;
        var visitor = data.visitor;
        show_tweets_onpage(tweets, host, visitor, $page = $('#id-div-followedtweets'));
    } else {
        log('请求失败');
    }
};

var notifications_response = function (data) {
    if (data.success) {
        var increment = 1;
        notifications_page += increment;
        log('at success', data);
        var notifications = data.notifications;
        var host = data.host;
        var visitor = data.visitor;
        show_notifications_onpage(notifications, host, visitor);
    } else {
        log('请求失败');
    }
};

var show_tweets_onpage = function (tweets, host, visitor, $page) {
    var t = tweets;
    for (var i = 0; i < t.length; i++) {
        tweet = t[i];
        var avatar_path = tweet.avatar;
        var comments_length = tweet.comments_length;
        if (comments_length == 0) {
            comments_length = '';
        }
        var template = tweet_template(avatar_path, tweet, comments_length);
        $page.append(template);
    }
};

var show_notifications_onpage = function (notifications, host, visitor) {
    var t = notifications;
    for (var i = 0; i < t.length; i++) {
        notification = t[i];
        tweet = t[i].t;
        var comments_length = tweet.comments_length;
        if (comments_length == 0) {
            comments_length = '';
        }
        var avatar_path = tweet.avatar;
        var template = notification_template(notification, avatar_path, tweet, comments_length);
        $('#id-div-notification').append(template);
    }
};
