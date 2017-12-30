var add_newcomment = function () {
    var $tweet = $(this).closest(".singletweet")
    var $interect_area = $tweet.children('.div-interact-area')
    var $comments_input = $interect_area.find('.text-addcomment')
    log('comments_input', $comments_input)
    var content = $comments_input.val();
    var form = {
        'content': content,
    };
    var tweet_id = $tweet.data('id')
    log('tid', tweet_id)
    // JSON.stringify 可以把一个 object 转换为字符串
    var url = '/comment/add' + '/' + tweet_id;
    post($input_box = $comments_input, url, form, response = new_comment, $target = $interect_area);
};

var new_comment = function (data, $target) {
    if (data.success) {
        var $comments_input = $target.find('.text-addcomment')
        $comments_input.val('')
        var t = data.comment
        var u = data.user
        var template = `
                    <hr/>
                    <div class="well well-sm clearfix" data-id="${t.id}">
                        ${u.username} · ${formatted_time(t.created_time)}
                        <br>
                        ${t.content}
                    </div>
                    `
        var $input_area = $target.find(".div-commentarea .input-group")
        $(template).hide().insertAfter($input_area).show("slow")
    } else {
        log('请求失败');
    }
};
