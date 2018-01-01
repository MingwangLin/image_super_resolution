var tweet_template = function (avatar_path, tweet, comments_length) {
    var template =
        `
                <div class="media clearfix">
                <div class="tweetbox">
                  <div class="media-left">
                      <a href="#">
                        <img class="media-object img-circle" src="${avatar_path}" alt="32x32" style="width: 32px; height: 32px;">
                      </a>
                  </div>
                  <div class="media-body clearfix singletweet" data-id="${tweet.id}">
                      <span class="font-tweet-name">
                        ${href_for_personalpage(tweet.user_name)} ·
                        <span class="font-tweet-time">
                        ${formatted_time(tweet.created_time)}
                        </span>
                      </span>
                      <br>
                      ${tweet.content}
                      <div class="tweet-imgs clearfix">
                      ${imgs_thumnail_template(tweet)}
                      </div>
                      ${reposted_tweet_template(tweet.original_tweet)}
                    <div class="div-interact-area">
                      <div class="btn-group bottom-right " role="group" aria-label="...">
                      <button type="button" class="btn btn-default btn-sm button-comments">
                        <span class="glyphicon glyphicon-pencil" aria-hidden="true">
                        </span>
                        <span class="comment-button-text">Reply{comments_length}</span>
                      </button>
                      <button type="button" class="btn btn-default btn-sm button-reposts">
                        <span class="glyphicon glyphicon-share" aria-hidden="true">
                        </span>
                        <span class="repost-button-text">Retweet</span>
                      </button>
                      </div>
                      <div class="clearfix div-repostarea" style="display: none">
                        ${addrepost_textarea_template}
                      </div>
                      <div class="clearfix div-commentarea" style="display: none">
                        ${addcomment_textarea_template}
                      </div>
                    </div>
                    <hr/>
                  </div>
                </div>
                </div>
                `
    return template;
};

var reposted_tweet_template = function (tweet) {
    if (tweet.length != 0) {
        tweet = tweet[0];
        comments_length = tweet.comments_length
        if (comments_length == 0) {
            comments_length = '';
        }
        var template =
            `
              <div class="media repostbox clearfix">
                  <div class="media-left">
                    <a href="#">
                      <img class="media-object img-circle" src="${tweet.avatar}" alt="32x32" style="width: 28px; height: 28px;">
                    </a>
                  </div>
                  <div class="media-body singletweet clearfix" data-id="${tweet.id}">
                    <span class="font-tweet-name">
                      ${href_for_personalpage(tweet.user_name)} ·
                        <span class="font-tweet-time">
                          ${formatted_time(tweet.created_time)}
                        </span>
                    </span>
                    <p>${tweet.content}</p>
                    <div class="tweet-imgs clearfix">
                    ${imgs_thumnail_template(tweet)}
                    </div>
                    <div class="div-interact-area">
                      <div class="btn-group bottom-right" role="group" aria-label="...">
                        <button type="button" class="btn btn-default btn-xs button-comments">
                          <span class="glyphicon glyphicon-pencil" aria-hidden="true">
                          </span>
                          <span class="comment-button-text">Reply${comments_length}</span>
                        </button>
                        <button type="button" class="btn btn-default btn-xs button-reposts">
                          <span class="glyphicon glyphicon-share" aria-hidden="true">
                          </span>
                          <span class="repost-button-text">Retweet</span>
                        </button>
                      </div>
                      <div class="clearfix div-repostarea" style="display: none">
                        ${addrepost_textarea_template}
                      </div>
                      <div class="clearfix div-commentarea" style="display: none">
                        ${addcomment_textarea_template}
                      </div>
                    </div>
                    <hr />
                  </div>
              </div>
                `
        return template;
    } else {
        return template = ``
    }
};

var img_thumnail = function (url, id) {
    var template =
        `
                    <div class="img">
                      <a href="#">
                      <img id="${id}" src="${url}" alt="uploaded-picture" width="300" height="200">
                      </a>
                    </div>
                    `
    return template;
}

var imgs_thumnail_template = function (tweet) {
    var imgs_template = ``;
    var tweet_imgs = tweet.imgs;
    var img_number = tweet_imgs.length;
    if (img_number != 0) {
        for (var i = 0; i < img_number; i++) {
            var img_url = tweet_imgs[i].content;
            var img_id = 'id-img-' + tweet_imgs[i].id;
            var img_template = img_thumnail(img_url, img_id);
            imgs_template = imgs_template.concat(img_template);
        }
        ;
    }
    ;
    return imgs_template;
};

var notification_template = function (notification, avatar_path, tweet, comments_length) {
    var words = 'new@'
    var template = `
    <div class="tweetbox clearfix">
     <span class="font-bold">
      ${notification.sender_name} ${words}
      <hr/>
     </span>
      ${tweet_template(avatar_path, tweet, comments_length)}
    </div>
       `;
    return template;
};

var none_template = `<div class="none tweetbox">
                      <p class="text-center ">
                        no content yet
                      </p>
                    </div>
                    `

var nomore_template = `<p class="nomore text-center">
                    <span class="glyphicon glyphicon-info-sign">
                    </span>
                    nothing more
                    </p>
                    `
var addcomment_textarea_template = `
                      <hr />
                      <div class="input-group">
                      <input type="text" class="form-control text-addcomment" name="content" placeholder="评论点什么">
                      <span class="input-group-btn">
                      <button class="btn btn-default pull-right button-addcomment" type="button">
                      <span class="glyphicon glyphicon-send" aria-hidden="true">
                      </span>
                      Reply
                      </button>
                      </span>
                      </div>
                      `

var addrepost_textarea_template = `
                      <hr />
                      <div class="input-group">
                      <input type="text" class="form-control text-addrepost" name="content" placeholder="说点什么">
                      <span class="input-group-btn">
                      <button class="btn btn-default pull-right button-addrepost" type="button">
                      <span class="glyphicon glyphicon-share" aria-hidden="true">
                      </span>
                      Retweet
                      </button>
                      </span>
                      </div>
                      `
