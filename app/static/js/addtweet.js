var add_newtweet = function () {
    var content = $('#id-text-content').val();
    var $imgs = $('#id-div-picturearea').find("img");
    log('i', $imgs)
    var imgs_length = $imgs.length;
    var imgs_url = [];
    if (imgs_length != 0) {
        for (var i = 0; i < imgs_length; i++) {
            var img_url = $($imgs[i]).attr("src");
            imgs_url.push(img_url);
        }
        ;
    }
    ;
    // var picture_url =
    var form = {
        'content': content,
        'imgs_url': imgs_url,
    };
    var url = '/tweet/add';
    post($input_box = $('#id-text-content'), url, form, new_tweet);
}

var new_tweet = function (data) {
    if (data.success) {
        $('#id-text-content').val('');
        $('#id-div-picturearea').empty();
        $('.upload-wrapper').hide();
        var tweet = data.tweet;
        var user = data.user;
        var avatar_path = user.avatar;
        var comments_length = '';
        template = tweet_template(avatar_path, tweet, comments_length);
        $(template).hide().prependTo('#id-div-mytweets').show("slow");
    } else {
        log('请求失败');
    }
}
